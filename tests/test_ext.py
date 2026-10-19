import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)              # ext.build() uses paths relative to the project root
sys.path.insert(0, ROOT)

import torch
import ext
from softmax import compare, make_input, bench, softmax_composed, softmax_oracle, softmax_torch

NEG_INF = float('-inf')
VARIANTS = (0, 1, 2)
BLOCKS = (128, 256, 512, 1024)
RTOL, ATOL = 1e-3, 1e-6     # fp16 rounding is at most 2^-11 = 4.9e-4 relative; atol covers fp16 subnormals


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
    # absolute and row-sum thresholds from the lab, plus a relative check against the float64 oracle,
    # because an absolute 1e-3 is loose when typical probabilities are about 1e-3
    r = compare(y, x)
    ref = softmax_oracle(x)
    rel_ok = torch.allclose(y.double().cpu(), ref, rtol=RTOL, atol=ATOL)
    return (r['max_abs_err'] < 1e-3 and r['max_row_sum_dev'] < 5e-3 and rel_ok), r['max_abs_err']


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
    # one block takes the all -inf early return while its neighbours do the normal path
    ("9 mixed rows",      lambda: half([[NEG_INF, NEG_INF], [1.0, 2.0], [NEG_INF, 0.0]]),
     lambda y, x: check_exact(y, [[0.0, 0.0], [0.2689, 0.7311], [0.0, 1.0]])),
    # row longer than every block size, with masked values inside it
    ("10 N=5000, 1/3 masked", lambda: masked_input(), check_compare),
]


def masked_input():
    torch.manual_seed(1)
    x = torch.randn(4, 5000, dtype=torch.float16, device='cuda')
    x[:, ::3] = NEG_INF
    return x


class Counter:
    def __init__(self):
        self.total = 0
        self.failed = 0

    def check(self, ok, label):
        self.total += 1
        if not ok:
            self.failed += 1
            print(f"FAIL {label}")


def expect_raises(fn, label, c):
    try:
        fn()
    except RuntimeError as e:
        c.check(True, label)
        return str(e).splitlines()[0]
    c.check(False, label + " (no error raised)")
    return None


def main():
    m = ext.build()
    c = Counter()

    print(f"\n=== fast_exp=False: {len(CASES)} cases x variant {VARIANTS} x block {BLOCKS} ===")
    for variant in VARIANTS:
        for block in BLOCKS:
            for name, make, check in CASES:
                x = make()
                y = m.forward(x, variant, block, False)
                torch.cuda.synchronize()
                ok, err = check(y, x)
                c.check(ok, f"variant={variant} block={block} case {name} err={err:.3e}")
    print(f"{c.total - c.failed}/{c.total} checks passed")

    print("\n=== fast_exp=True: cases 3, 4 and 10, every variant, block 256 (same thresholds) ===")
    for variant in VARIANTS:
        for name, make, check in (CASES[2], CASES[3], CASES[9]):
            x = make()
            y = m.forward(x, variant, 256, True)
            torch.cuda.synchronize()
            ok, err = check(y, x)
            r = compare(y, x)
            c.check(ok, f"fast_exp variant={variant} case {name}")
            print(f"variant={variant} case {name}: max_abs_err={r['max_abs_err']:.3e} "
                  f"max_row_sum_dev={r['max_row_sum_dev']:.3e} {'ok' if ok else 'FAIL'}")

    print("\n=== B3 shared-memory sizes ===")
    limit = m.max_smem_optin()
    scratch = 32 * 8
    # 24,576 halves = 48 KiB of row: static + dynamic smem crosses 48 KiB here, so the opt-in attribute is needed.
    # 30,000 is well inside the opt-in range. The largest N that fits is (limit - scratch) // 2.
    for N in (24448, 24576, 30000, (limit - scratch) // 2):
        torch.manual_seed(2)
        x = torch.randn(2, N, dtype=torch.float16, device='cuda')
        y3 = m.forward(x, 2, 256, False)
        y2 = m.forward(x, 1, 256, False)
        torch.cuda.synchronize()
        ref = softmax_oracle(x)
        ok = torch.allclose(y3.double().cpu(), ref, rtol=RTOL, atol=ATOL)
        same = (y3.float() - y2.float()).abs().max().item()
        c.check(ok, f"B3 N={N}")
        print(f"N={N:>6} ({2 * N + scratch:,} B smem): {'ok' if ok else 'FAIL'}, max |B3 - B2| = {same:.1e}")

    print("\n=== inputs that must raise ===")
    too_big = torch.zeros(1, (limit - scratch) // 2 + 1, dtype=torch.float16, device='cuda')
    msg = expect_raises(lambda: m.forward(too_big, 2, 256, False), "B3 row larger than shared memory", c)
    print(f"row of {too_big.shape[1]:,} halves, max_smem_optin={limit}: {msg}")
    x = make_input(1024)
    print("fp32 input:     ", expect_raises(lambda: m.forward(x.float(), 0, 256, False), "fp32 input", c))
    print("non-contiguous: ", expect_raises(lambda: m.forward(x.t(), 0, 256, False), "non-contiguous input", c))
    print("CPU tensor:     ", expect_raises(lambda: m.forward(x.cpu(), 0, 256, False), "CPU input", c))
    print("block 100:      ", expect_raises(lambda: m.forward(x, 0, 100, False), "block not a multiple of 32", c))
    print("variant 3:      ", expect_raises(lambda: m.forward(x, 3, 256, False), "unknown variant", c))

    print("\n=== timing, make_input(4096) ===")
    x = make_input(4096)
    print("B1 (variant 0)  ", bench(lambda t: m.forward(t, 0, 256, False), x))
    print("B2 (variant 1)  ", bench(lambda t: m.forward(t, 1, 256, False), x))
    print("B3 (variant 2)  ", bench(lambda t: m.forward(t, 2, 256, False), x))
    print("composed        ", bench(softmax_composed, x))
    print("torch.softmax   ", bench(softmax_torch, x))

    print(f"\nTOTAL: {c.total - c.failed}/{c.total} checks passed")
    if c.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
