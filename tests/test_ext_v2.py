"""Tests for the version 2 extension (csrc/v2). v1 stays tested by tests/test_ext.py.

variant 0 B1, 1 B2, 2 B3 (block widened to fill the SM), 3 B4 (16-byte loads, per-chunk rescale,
exp2, reciprocal), 4 B5 (one warp per row).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch
import ext_v2
from softmax import bench, compare, make_input, softmax_oracle, softmax_torch
from test_ext import ATOL, CASES, RTOL, Counter, check_compare, expect_raises

VARIANTS = (0, 1, 2, 3, 4)
NAMES = {0: 'B1', 1: 'B2', 2: 'B3', 3: 'B4', 4: 'B5'}
BLOCKS = (128, 256, 512, 1024)


def randn(rows, n, seed):
    torch.manual_seed(seed)
    return torch.randn(rows, n, dtype=torch.float16, device='cuda')


def misaligned(rows, n):
    # contiguous, but the first element sits 2 bytes past a 16-byte boundary, so B4/B5 must use the scalar path
    torch.manual_seed(3)
    buf = torch.randn(rows * n + 1, dtype=torch.float16, device='cuda')
    return buf[1:].view(rows, n)


# extra inputs that take the 16-byte path (N % 8 == 0) or must fall back from it
VEC_CASES = [
    ("v1 N=8, one chunk",          lambda: randn(3, 8, 4)),
    ("v2 N=4096",                  lambda: randn(16, 4096, 5)),
    ("v3 N=4104 (513 chunks)",     lambda: randn(4, 4104, 6)),
    ("v4 N=4096, misaligned",      lambda: misaligned(4, 4096)),
    ("v5 N=2048, values up to 60", lambda: randn(8, 2048, 7) * 20),
]


def main():
    m = ext_v2.build()
    c = Counter()

    print(f"\n=== fast_exp=False: {len(CASES) + len(VEC_CASES)} cases x variant {VARIANTS} x block {BLOCKS} ===")
    for variant in VARIANTS:
        for block in BLOCKS:
            for name, make, check in CASES:
                x = make()
                y = m.forward(x, variant, block, False)
                torch.cuda.synchronize()
                ok, err = check(y, x)
                c.check(ok, f"{NAMES[variant]} block={block} case {name} err={err:.3e}")
            for name, make in VEC_CASES:
                x = make()
                y = m.forward(x, variant, block, False)
                torch.cuda.synchronize()
                ok, err = check_compare(y, x)
                c.check(ok, f"{NAMES[variant]} block={block} case {name} err={err:.3e}")
    print(f"{c.total - c.failed}/{c.total} checks passed")

    print("\n=== fast_exp=True, block 256 (same thresholds) ===")
    for variant in VARIANTS:
        worst = 0.0
        for name, make, check in [CASES[2], CASES[3], CASES[9]] + [(n, f, check_compare) for n, f in VEC_CASES]:
            x = make()
            y = m.forward(x, variant, 256, True)
            torch.cuda.synchronize()
            ok, err = check(y, x)
            worst = max(worst, err)
            c.check(ok, f"fast_exp {NAMES[variant]} case {name} err={err:.3e}")
        print(f"{NAMES[variant]}: worst max_abs_err {worst:.3e}")

    print("\n=== B4 and B5 against B2 on make_input(4096) ===")
    x = make_input(4096)
    y2 = m.forward(x, 1, 256, False).float()
    for variant in (3, 4):
        y = m.forward(x, variant, 256, False).float()
        diff = (y - y2).abs().max().item()
        ok = torch.allclose(y, y2, rtol=2 * RTOL, atol=ATOL)
        c.check(ok, f"{NAMES[variant]} vs B2")
        print(f"{NAMES[variant]}: max |y - y_B2| = {diff:.2e} {'ok' if ok else 'FAIL'}")

    print("\n=== B3 launch: threads per block chosen from shared memory ===")
    for N in (2048, 4096, 8192, 16384):
        print(f"N={N:>6}: block 256 requested -> {m.b3_block(N, 256)} launched")
    if torch.cuda.get_device_capability() == (7, 5):     # 64 KiB smem and 1,024 threads per SM
        for N, want in ((2048, 256), (8192, 320), (16384, 1024)):
            got = m.b3_block(N, 256)
            c.check(got == want, f"b3_block({N}, 256) = {got}, expected {want} on sm_75")

    print("\n=== B3 shared-memory sizes ===")
    limit = m.max_smem_optin()
    scratch = 32 * 8
    for N in (24448, 24576, 30000, (limit - scratch) // 2):
        x = randn(2, N, 2)
        y3 = m.forward(x, 2, 256, False)
        torch.cuda.synchronize()
        ok = torch.allclose(y3.double().cpu(), softmax_oracle(x), rtol=RTOL, atol=ATOL)
        c.check(ok, f"B3 N={N}")
        print(f"N={N:>6} ({2 * N + scratch:,} B smem, {m.b3_block(N, 256)} threads): {'ok' if ok else 'FAIL'}")

    print("\n=== inputs that must raise ===")
    too_big = torch.zeros(1, (limit - scratch) // 2 + 1, dtype=torch.float16, device='cuda')
    print("B3 row too big: ", expect_raises(lambda: m.forward(too_big, 2, 256, False), "B3 row too big", c))
    x = make_input(1024)
    print("fp32 input:     ", expect_raises(lambda: m.forward(x.float(), 3, 256, False), "fp32 input", c))
    print("non-contiguous: ", expect_raises(lambda: m.forward(x.t(), 3, 256, False), "non-contiguous", c))
    print("block 100:      ", expect_raises(lambda: m.forward(x, 4, 100, False), "block 100", c))
    print("variant 5:      ", expect_raises(lambda: m.forward(x, 5, 256, False), "variant 5", c))

    print("\n=== timing (median µs) ===")
    for S in (2048, 8192):
        x = make_input(S)
        line = {NAMES[v]: round(bench(lambda t, v=v: m.forward(t, v, 256, False), x)['t_median_us'], 1) for v in VARIANTS}
        line['torch.softmax'] = round(bench(softmax_torch, x)['t_median_us'], 1)
        print(f"S={S}: {line}")
        del x

    print(f"\nTOTAL: {c.total - c.failed}/{c.total} checks passed")
    if c.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
