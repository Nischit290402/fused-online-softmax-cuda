import os

from torch.utils.cpp_extension import load


def build():
    # in: nothing. out: version 2 module (forward, max_smem_optin, b3_block), built into build_v2/
    os.makedirs('build_v2', exist_ok=True)
    return load(name='fused_softmax_v2',
                sources=['csrc/softmax_bindings_v2.cpp', 'csrc/softmax_kernels_v2.cu'],
                build_directory='build_v2', extra_cuda_cflags=['-O3'], verbose=True)
