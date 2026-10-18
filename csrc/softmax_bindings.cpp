#include <torch/extension.h>

torch::Tensor softmax_forward(torch::Tensor x, int64_t variant, int64_t block, bool fast_exp);
int64_t max_smem_optin();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_forward);
    m.def("max_smem_optin", &max_smem_optin);
}