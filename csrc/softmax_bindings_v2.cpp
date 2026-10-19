#include <torch/extension.h>

torch::Tensor softmax_forward_v2(torch::Tensor x, int64_t variant, int64_t block, bool fast_exp);
int64_t max_smem_optin_v2();
int64_t b3_block(int64_t N, int64_t block);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_forward_v2,
          "variant 0 B1, 1 B2, 2 B3 (block widened to fill the SM), 3 B4, 4 B5 (one warp per row)");
    m.def("max_smem_optin", &max_smem_optin_v2);
    m.def("b3_block", &b3_block, "threads per block that variant 2 launches with for a row of N halves");
}
