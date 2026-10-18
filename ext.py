from torch.utils.cpp_extension import load

def build():
    # in: nothing. out: compiled module with forward() and max_smem_optin()
    return load(name='fused_softmax', sources=['csrc/softmax_bindings.cpp', 'csrc/softmax_kernels.cu'],
                build_directory='build', extra_cuda_cflags=['-O3'], verbose=True)