import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)              # ext.build() uses paths relative to the project root
sys.path.insert(0, ROOT)

import torch
import ext
from softmax import compare, make_input, bench, softmax_composed, softmax_torch

NEG_INF = float('-inf')


def half(rows):
    return torch.tensor(rows, dtype=torch.float16, device='cuda')


def case3_input():
    torch.manual_seed(0)
    return torch.randn(8, 1001, dtype=torch.float16, device='cuda')


def check_exact(y, expected, tol=1e-3):
    exp = torch.tensor(expected, dtype=torch.float32)
    err = (y.float().cpu() - exp).abs().max().item()
    return err < tol, err


def check_compare(y, x):
    r = compare(y, x)
    return (r['max_abs_err'] < 1e-3 and r['max_row_sum_dev'] < 5e-3), r['max_abs_err']


# Each case: (name, input builder, checker). Checker takes (y, x) and returns (ok, err).
CASES = [
    ("1 single element",  lambda: half([[5.0]]),
     lambda y, x: check_exact(y, [[1.0]])),
    ("2 [1,2,3]",         lambda: half([[1.0, 2.0, 3.0]]),
     lambda y, x: check_exact(y, [[0.0900, 0.2447, 0.6652]])),
    ("3 randn(8,1001)",   case3_input, check_compare),
    ("4 make_input(1024)", lambda: make_input(1024), check_compare),
    ("5 one -inf",        lambda: half([[NEG_INF, 0.0, 0.0]]),
     lambda y, x: check_exact(y, [[0.0, 0.5, 0.5]])),
    ("6 all -inf",        lambda: half([[NEG_INF, NEG_INF, NEG_INF]]),
     lambda y, x: check_exact(y, [[0.0, 0.0, 0.0]])),
    ("7 full(1,8,30)",    lambda: torch.full((1, 8), 30.0, dtype=torch.float16, device='cuda'),
     lambda y, x: check_exact(y, [[0.125] * 8])),
    ("8 [30,0]",          lambda: half([[30.0, 0.0]]),
     lambda y, x: check_exact(y, [[1.0, 0.0]])),
]


def main():
    m = ext.build()
    total = 0
    failed = 0

    print("\n=== fast_exp=False: 8 cases x variant {0,1} x block {128,256,512,1024} ===")
    for variant in (0, 1):
        for block in (128, 256, 512, 1024):
            for name, make, check in CASES:
                x = make()
                y = m.forward(x, variant, block, False)
                torch.cuda.synchronize()
                ok, err = check(y, x)
                total += 1
                if not ok:
                    failed += 1
                    print(f"FAIL variant={variant} block={block} case {name} err={err:.3e}")
    print(f"{total - failed}/{total} checks passed")

    print("\n=== fast_exp=True: cases 3 and 4 (block 256) ===")
    for variant in (0, 1):
        for name, make, check in (CASES[2], CASES[3]):
            x = make()
            y = m.forward(x, variant, 256, True)
            torch.cuda.synchronize()
            r = compare(y, x)
            print(f"variant={variant} case {name}: max_abs_err={r['max_abs_err']:.3e} "
                  f"max_row_sum_dev={r['max_row_sum_dev']:.3e}")

    print("\n=== timing, make_input(4096) ===")
    x = make_input(4096)
    print("B1 (variant 0)  ", bench(lambda t: m.forward(t, 0, 256, False), x))
    print("B2 (variant 1)  ", bench(lambda t: m.forward(t, 1, 256, False), x))
    print("composed        ", bench(softmax_composed, x))
    print("torch.softmax   ", bench(softmax_torch, x))

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
