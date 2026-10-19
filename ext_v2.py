import os

from torch.utils.cpp_extension import load


def build():
    # in: nothing. out: version 2 module (forward, max_smem_optin, b3_block), built into build_v2/
    os.makedirs('build_v2', exist_ok=True)
    return load(name='fused_softmax_v2',
                sources=['csrc/softmax_bindings_v2.cpp', 'csrc/softmax_kernels_v2.cu'],
                build_directory='build_v2', verbose=True,
                # warning 221: MSVC defines INFINITY as (float)1e+300, which nvcc flags as an overflow
                # at every use of -INFINITY. The value is still +inf, so the warning is noise.
                extra_cuda_cflags=['-O3', '-Xcudafe', '--diag_suppress=221'])
