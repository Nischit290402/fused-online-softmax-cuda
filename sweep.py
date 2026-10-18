import csv
import gc
import os
import sys
import time

import torch
from torch.profiler import profile, ProfilerActivity

import ext
from softmax import bench, make_input, softmax_composed, softmax_torch

HAVE_B3 = True           # variant 2 (B3) is built; rows where the row does not fit get skip_smem
SCRATCH_BYTES = 32 * 8   # float2 smem[32] inside the fused kernels
SIZES = [1024, 2048, 4096, 8192, 16384, 32768]

HEADERS = ['variant', 'S', 'block', 'exp_impl',
           't_median_us', 't_p10_us', 't_p90_us', 'peak_mem_MiB',
           'kernels_per_call', 'eff_bw_GBps',
           'dram_read_bytes', 'dram_write_bytes', 'dram_pct_peak',
           'sm_pct_peak', 'occupancy_pct', 'regs_per_thread', 'status']


def build_cells() -> list[dict]:
    cells = []
    base = ['composed', 'torch.softmax', 'B1', 'B2'] + (['B3'] if HAVE_B3 else [])
    for S in SIZES:                                   # 24 cells (30 with B3)
        for v in base:
            cells.append({'variant': v, 'S': S, 'block': 256, 'exp_impl': 'expf'})
    for v in ('B1', 'B2'):                            # 6 cells
        for block in (128, 512, 1024):
            cells.append({'variant': v, 'S': 8192, 'block': block, 'exp_impl': 'expf'})
    for v in ('B1', 'B2'):                            # 4 cells
        for S in (8192, 16384):
            cells.append({'variant': v, 'S': S, 'block': 256, 'exp_impl': '__expf'})
    return cells                                      # 34 cells (40 with B3)


def make_fn(m, cell: dict):
    v = cell['variant']
    if v == 'composed':
        return softmax_composed
    if v == 'torch.softmax':
        return softmax_torch
    variant = {'B1': 0, 'B2': 1, 'B3': 2}[v]
    block = cell['block']
    fast = cell['exp_impl'] == '__expf'
    return lambda t: m.forward(t, variant, block, fast)


def count_kernels(fn, x) -> int:
    fn(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(x)
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)


def warm_up_gpu(seconds: float = 6.0) -> None:
    # The GPU idles at 300 MHz and needs about 5 s of load to reach its working clocks.
    # Run softmax for a few seconds so the first timed cell does not run during the ramp-up.
    x = make_input(8192, 0)
    count_kernels(softmax_torch, x)      # start torch.profiler now so its set-up does not stall a timed cell
    t_end = time.time() + seconds
    while time.time() < t_end:
        for _ in range(20):
            softmax_torch(x)
        torch.cuda.synchronize()
    del x
    gc.collect()
    torch.cuda.empty_cache()


def spin(fn, x, seconds: float) -> None:
    # keep the GPU busy with fn for a moment so the clocks are settled before timing
    t_end = time.time() + seconds
    while time.time() < t_end:
        for _ in range(20):
            fn(x)
        torch.cuda.synchronize()


def effective_bw_GBps(S: int, t_median_us: float) -> float:
    return 4 * S * S / (t_median_us * 1e-6) / 1e9


def theoretical_bw_GBps(mem_clock_mhz: float, bus_width_bits: int) -> float:
    return mem_clock_mhz * 1e6 * 2 * (bus_width_bits / 8) / 1e9


def run_cell(m, cell: dict) -> dict:
    row = {h: '' for h in HEADERS}
    row.update(cell)
    row['status'] = 'ok'
    if cell['variant'] == 'B3' and 2 * cell['S'] + SCRATCH_BYTES > m.max_smem_optin():
        row['status'] = 'skip_smem'
        return row
    x = None
    try:
        x = make_input(cell['S'], 0)
        fn = make_fn(m, cell)
        row['kernels_per_call'] = count_kernels(fn, x)
        # small S means calls of 60 to 600 us, so use more warm-up and more timed runs
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
    m = ext.build()
    warm_up_gpu()
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader()
        for i, cell in enumerate(cells, 1):
            row = run_cell(m, cell)
            w.writerow(row)
            f.flush()
            print(f"[{i}/{len(cells)}] {cell['variant']} S={cell['S']} block={cell['block']} "
                  f"{cell['exp_impl']}: {row['status']} {row['t_median_us']}")


if __name__ == '__main__':
    run_sweep(sys.argv[1], sys.argv[2])
