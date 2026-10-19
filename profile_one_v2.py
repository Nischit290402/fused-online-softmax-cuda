import sys

import torch

import ext_v2
from softmax import make_input

VARIANT_ID = {'B1': 0, 'B2': 1, 'B3v2': 2, 'B4': 3, 'B5': 4}


def main(variant: str, S: int, block: int = 256) -> None:
    assert variant in VARIANT_ID, f"variant must be one of {list(VARIANT_ID)}"
    m = ext_v2.build()
    v = VARIANT_ID[variant]
    fn = lambda t: m.forward(t, v, block, False)
    x = make_input(S, 0)
    for _ in range(3):                      # warm-up calls stay outside the profiled range
        fn(x)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    fn(x)                                   # the one call ncu records
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 256)
