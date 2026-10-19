"""Timing sweep for the version 2 kernels.

usage: python sweep_v2.py forward|reverse results/v2/sweep_v2_run1.csv
Rows: torch.softmax, B2 (unchanged, the baseline), B3v2 (B3 with the widened block), B4, B5 at six sizes,
B4 at three more block sizes, and B4/B5 with the fast exp2. v1 numbers stay in results/sweep_run*.csv.
"""
import csv
import gc
import os
import sys

import torch

import ext_v2
from softmax import bench, make_input, softmax_torch
from sweep import HEADERS, SIZES, count_kernels, effective_bw_GBps, spin, warm_up_gpu

VARIANT_ID = {'B1': 0, 'B2': 1, 'B3v2': 2, 'B4': 3, 'B5': 4}
HEADERS_V2 = HEADERS + ['block_launched']
SCRATCH_BYTES = 32 * 8


def build_cells() -> list[dict]:
    cells = []
    for S in SIZES:                                                   # 30 cells
        for v in ('torch.softmax', 'B2', 'B3v2', 'B4', 'B5'):
            cells.append({'variant': v, 'S': S, 'block': 256, 'exp_impl': 'expf'})
    for block in (128, 512, 1024):                                    # 3 cells
        cells.append({'variant': 'B4', 'S': 8192, 'block': block, 'exp_impl': 'expf'})
    for v in ('B4', 'B5'):                                            # 4 cells
        for S in (8192, 16384):
            cells.append({'variant': v, 'S': S, 'block': 256, 'exp_impl': 'ex2.approx'})
    return cells                                                      # 37 cells


def make_fn(m, cell: dict):
    if cell['variant'] == 'torch.softmax':
        return softmax_torch
    variant, block = VARIANT_ID[cell['variant']], cell['block']
    fast = cell['exp_impl'] != 'expf'
    return lambda t: m.forward(t, variant, block, fast)


_KPC = {}


def count_all_variants(m, cells) -> None:
    # count once per variant on a small input, before any long call (see sweep.py)
    x = make_input(1024, 0)
    for cell in cells:
        if cell['variant'] not in _KPC:
            _KPC[cell['variant']] = count_kernels(make_fn(m, cell), x)
    del x
    print('kernels per call:', _KPC)


def run_cell(m, cell: dict) -> dict:
    row = {h: '' for h in HEADERS_V2}
    row.update(cell)
    row['status'] = 'ok'
    if cell['variant'] == 'B3v2':
        if 2 * cell['S'] + SCRATCH_BYTES > m.max_smem_optin():
            row['status'] = 'skip_smem'
            return row
        row['block_launched'] = m.b3_block(cell['S'], cell['block'])
    else:
        row['block_launched'] = cell['block']
    x = None
    try:
        x = make_input(cell['S'], 0)
        fn = make_fn(m, cell)
        row['kernels_per_call'] = _KPC[cell['variant']]
        warmup, iters = (50, 200) if cell['S'] <= 4096 else (10, 50)
        if cell['S'] <= 4096:
            spin(fn, x, 0.5)
        row.update(bench(fn, x, warmup=warmup, iters=iters))
        row['eff_bw_GBps'] = effective_bw_GBps(cell['S'], row['t_median_us'])
    except torch.cuda.OutOfMemoryError:
        row['status'] = 'oom'
    del x
    gc.collect()
    torch.cuda.empty_cache()
    return row


def run_sweep(order: str, out_csv: str) -> None:
    assert order in ('forward', 'reverse')
    cells = build_cells()
    if order == 'reverse':
        cells = cells[::-1]
    m = ext_v2.build()
    count_all_variants(m, cells)
    warm_up_gpu()
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=HEADERS_V2)
        w.writeheader()
        for i, cell in enumerate(cells, 1):
            row = run_cell(m, cell)
            w.writerow(row)
            f.flush()
            print(f"[{i}/{len(cells)}] {cell['variant']} S={cell['S']} block={cell['block']} "
                  f"{cell['exp_impl']}: {row['status']} {row['t_median_us']}")


if __name__ == '__main__':
    run_sweep(sys.argv[1], sys.argv[2])
