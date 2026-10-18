import sys

import torch

import ext
from softmax import make_input, softmax_composed, softmax_torch

VARIANTS = ['composed', 'torch.softmax', 'B1', 'B2', 'B3']


def main(variant: str, S: int, block: int = 256) -> None:
    assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
    m = ext.build()
    fns = {
        'composed': softmax_composed,
        'torch.softmax': softmax_torch,
        'B1': lambda t: m.forward(t, 0, block, False),
        'B2': lambda t: m.forward(t, 1, block, False),
        'B3': lambda t: m.forward(t, 2, block, False),
    }
    fn = fns[variant]
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
