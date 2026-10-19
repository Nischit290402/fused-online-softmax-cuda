import gc
import math
import os
import csv
from math import inf, exp

import numpy as np
import torch

flush = torch.empty(16 * 2 ** 20, dtype=torch.uint8, device='cuda')  # 4x the 4 MiB L2

def softmax_safe_3pass(row: list[float])->list[float]:
    m = -inf
    for r in row:
        m = max(m,r)
    if m == -inf: return [0.0]*len(row)
    exp_shift = []
    d = 0
    for r in row:
        x = math.exp(r-m)
        exp_shift.append(x)
        d = d + x
    out = []
    for r in exp_shift:
        out.append(r/d)
    return out

def softmax_online(row: list[float])->list[float]:
    m = -inf
    d = 0
    for t in row:
        mo = m
        m = max(m,t)
        if m==-inf: continue
        d = d*exp(mo-m) + exp(t-m)
    if m == -inf: return [0.0]*len(row)
    return [exp(r - m)/d for r in row]

def make_input(S: int, seed: int=0)->torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(S, S, dtype=torch.float16, device=torch.device('cuda'))

def softmax_oracle(x: torch.Tensor)->torch.Tensor:
    d_x = x.to('cpu',torch.float64)
    return torch.softmax(d_x, dim=-1)

def compare(y: torch.Tensor, x: torch.Tensor)->dict:
    ref = softmax_oracle(x)
    yd = y.to('cpu', torch.float64)
    return {'max_abs_err': (yd - ref).abs().max().item(),
            'max_row_sum_dev': (yd.sum(dim=-1) - 1).abs().max().item()}

def softmax_composed(x: torch.Tensor)->torch.Tensor:
    m = x.amax(dim=-1, keepdim=True)
    s = x - m
    e = torch.exp(s)
    d = e.sum(dim=-1, keepdim=True)
    y = e / d
    return y

def softmax_torch(x: torch.Tensor)->torch.Tensor:
    return torch.softmax(x, dim=-1)

def softmax_unsafe(x: torch.Tensor)->torch.Tensor:
    e = torch.exp(x)
    y = e/e.sum(dim=-1, keepdim=True)
    return y

def bench(fn, x: torch.Tensor, warmup=10, iters=50)->dict:
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    buffer_size = 16 * 1024 * 1024
    buffer = bytearray(buffer_size)
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        flush.zero_()
        start.record()
        fn(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end)*1000) # microseconds

    times.sort()
    peak_mb_MiB = (torch.cuda.max_memory_allocated() - base)/(2**20)
    vals = {}
    # median of times 't_median_us', 't_p10_us', 't_p90_us', 'peak_mem_MiB'
    vals['t_median_us'] = times[len(times) // 2]
    vals['t_p10_us'] = float(np.percentile(times, 10))
    vals['t_p90_us'] = float(np.percentile(times, 90))
    vals['peak_mem_MiB']=peak_mb_MiB

    return vals

def run_cell(name: str, fn, S: int) -> dict:
    row = {'variant': name, 'S': S, 't_median_us': None, 't_p10_us': None,
           't_p90_us': None, 'peak_mem_MiB': None, 'status': 'ok'}
    x = make_input(S, 0)
    try:
        row.update(bench(fn, x))
    except torch.cuda.OutOfMemoryError:
        row['status']='oom'
    del x
    gc.collect()
    torch.cuda.empty_cache()
    return row

def run_baseline(path: str = 'results/baseline.csv') -> None:
    sizes = [1024,2048,4096,8192,16384,32768]
    headers = ['variant','S','t_median_us','t_p10_us','t_p90_us','peak_mem_MiB','status']
    variants = [('composed', softmax_composed), ('torch.softmax', softmax_torch)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for S in sizes:
            for name, fn in variants:
                w.writerow(run_cell(name, fn, S))


def run_tests() -> None:
    # A1
    for fn in (softmax_safe_3pass, softmax_online):
        got = fn([1.0, 2.0, 3.0])
        want = [0.09003057, 0.24472847, 0.66524096]
        assert all(abs(g - w) < 1e-8 for g, w in zip(got, want)), fn.__name__
        assert fn([1000.0, 1000.0]) == [0.5, 0.5]
        assert fn([-inf, 0.0, 0.0]) == [0.0, 0.5, 0.5]
        assert fn([-inf, -inf]) == [0.0, 0.0]
        torch.manual_seed(0)
        r = torch.randn(64, dtype=torch.float64)
        got = torch.tensor(fn(r.tolist()), dtype=torch.float64)
        assert (got - torch.softmax(r, dim=-1)).abs().max().item() < 1e-12
        assert abs(got.sum().item() - 1) < 1e-12

    # A2
    x = make_input(1024)
    o = softmax_oracle(x)
    assert o.dtype == torch.float64 and o.device.type == 'cpu' and o.shape == (1024, 1024)
    assert (o.sum(dim=-1) - 1).abs().max().item() < 1e-12

    # A3 and A4
    for fn in (softmax_composed, softmax_torch):
        y = fn(x)
        c = compare(y, x)
        assert c['max_abs_err'] < 1e-3 and c['max_row_sum_dev'] < 5e-3, fn.__name__
        assert y.dtype == torch.float16 and y.is_cuda and y.shape == (1024, 1024)

    # A5
    x4 = make_input(4096)
    bc = bench(softmax_composed, x4)
    bt = bench(softmax_torch, x4)
    for b in (bc, bt):
        assert set(b) == {'t_median_us', 't_p10_us', 't_p90_us', 'peak_mem_MiB'}
        assert b['t_p10_us'] <= b['t_median_us'] <= b['t_p90_us']
    assert 96 <= bc['peak_mem_MiB'] <= 97
    assert 32 <= bt['peak_mem_MiB'] <= 33
    assert bt['t_median_us'] < bc['t_median_us']

    # A6
    dev = torch.device('cuda')
    a = torch.full((1, 8), 30.0, dtype=torch.float16, device=dev)
    b = torch.tensor([[30.0, 0.0]], dtype=torch.float16, device=dev)
    assert softmax_unsafe(a).isnan().all()
    assert softmax_unsafe(b).cpu()[0, 0].isnan() and softmax_unsafe(b).cpu()[0, 1] == 0
    for fn in (softmax_composed, softmax_torch):
        assert torch.allclose(fn(a).float(), torch.full((1, 8), 0.125, device=dev), atol=1e-3)
        assert torch.allclose(fn(b).float(), torch.tensor([[1.0, 0.0]], device=dev), atol=1e-3)

    # A7
    run_baseline('results/baseline.csv')
    with open('results/baseline.csv') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    print('all tests passed')

if __name__ == '__main__':
    run_tests()