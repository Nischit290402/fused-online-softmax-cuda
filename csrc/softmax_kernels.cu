#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x, expected_type) TORCH_CHECK(x.scalar_type() == expected_type, #x " must be " #expected_type)

#define CHECK_INPUT(x, expected_type) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_TYPE(x, expected_type)

// ---- device helpers ----
__device__ float  warp_reduce_max(float v) {
    // val is our say current lane value
    // nbor is calculated from mask
    int laneMask = 16;
    while (laneMask>0) {
        float nbor = __shfl_xor_sync(0xffffffff, v, laneMask);
        v = fmaxf(v,nbor);
        laneMask>>=1;
    }
    return v;
}
// __shfl_xor_sync, full mask
__device__ float  warp_reduce_sum(float v) {
    int laneMask = 16;
    while (laneMask>0) {
        v += __shfl_xor_sync(0xffffffff, v, laneMask);
        laneMask>>=1;
    }
    return v;
}

__device__ float  block_reduce_max(float v, float* smem) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int idx = warp; // the first lane of each warp will have the warp reduced value so can always just take that
    // for each thread<=>lane we have to perform warp reduce
    float val = warp_reduce_max(v);
    if (lane==0) smem[idx] = val;
    __syncthreads();
    int warps = blockDim.x / 32;
    if (lane < warps) v = smem[lane]; // fills each warp's first 8 values the same as 8 warp's first value!
    else v = -INFINITY;
    float m = warp_reduce_max(v);

    return m;
}

// smem: one float per warp

// (m, d) rule from Section 2
__device__ float  block_reduce_sum(float v, float* smem) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int idx = warp;

    float val = warp_reduce_sum(v);
    __syncthreads();
    if (lane==0) smem[idx] = val;
    __syncthreads();
    int warps = blockDim.x / 32;
    if (lane < warps) v = smem[lane];
    else v = 0;
    float d = warp_reduce_sum(v);

    return d;
}

template <bool FAST_EXP>
__device__ __forceinline__ float exp_f(float v) {
    if constexpr (FAST_EXP) return __expf(v);
    else return expf(v);
}
template <bool FAST_EXP>
__device__ float2 md_combine(float2 a, float2 b) {
    float ma = a.x;
    float mb = b.x;
    float da = a.y;
    float db = b.y;
    if (ma==-INFINITY && mb==-INFINITY) return make_float2(-INFINITY, 0);
    float2 c = a;
    c.x = fmaxf(ma,mb);
    c.y = da * exp_f<FAST_EXP>(ma-c.x) + db * exp_f<FAST_EXP>(mb-c.x);
    return c;
}

template <bool FAST_EXP>
__device__ float2 warp_reduce_md(float2 v) {
    int laneMask = 16;
    while (laneMask>0) {
        float nbor_x = __shfl_xor_sync(0xffffffffu, v.x, laneMask);
        float nbor_y = __shfl_xor_sync(0xffffffffu, v.y, laneMask);
        v = md_combine<FAST_EXP>(v, make_float2(nbor_x, nbor_y));
        laneMask>>=1;
    }
    return v;
}

template <bool FAST_EXP>
__device__ float2 block_reduce_md(float2 v, float2* smem) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int idx = warp;
    float2 val = warp_reduce_md<FAST_EXP>(v);
    if (lane==0) smem[idx] = val;
    __syncthreads();
    int warps = blockDim.x / 32;
    if (lane<warps) v = smem[lane];
    else {
        v.x = -INFINITY;
        v.y = 0;
    }
    float2 md = warp_reduce_md<FAST_EXP>(v);

    return md;
}

// ---- kernels: grid = rows, block = `block` threads, strided loop over N ----
template <bool FAST_EXP> __global__ void softmax_b1_kernel(const __half* x, __half* y, int64_t N) {
    // here x is one row of len N
    // y is the calculated exp(x-M)/d which is also a row of len N
    // Must use the above functions to parallelize
    // we have to read the row x in a strided manner to parallelize
    int64_t offset = blockIdx.x * N;
    int64_t s = blockDim.x;
    float maxi = -INFINITY;
    __shared__ float smem[32];
    for (int64_t i=offset+threadIdx.x;i<offset+N;i+=s) {
        maxi = fmaxf(maxi,__half2float(x[i]));
    }
    float m = block_reduce_max(maxi,smem);
    if (m==-INFINITY) {
        for (int64_t i=offset+threadIdx.x;i<offset+N;i+=s) {
            y[i]=__float2half(0.0f);
        }
        return;
    }
    float d = 0;
    for (int64_t i=offset+threadIdx.x;i<offset+N;i+=s) {
        d += exp_f<FAST_EXP>(__half2float(x[i])-m);
    }
    d = block_reduce_sum(d,smem);
    for (int64_t i=offset+threadIdx.x;i<offset+N;i+=s) {
        y[i] = __float2half(exp_f<FAST_EXP>(__half2float(x[i])-m)/d);
    }
}  // 3 passes

template <bool FAST_EXP> __global__ void softmax_b2_kernel(const __half* x, __half* y, int64_t N) {
    int64_t offset = blockIdx.x * N;
    int64_t s = blockDim.x;
    __shared__ float2 smem[32];
    float2 md = make_float2(-INFINITY,0.0f);
    for (int64_t i = offset+threadIdx.x;i<offset+N;i+=s) {
        md = md_combine<FAST_EXP>(make_float2(__half2float(x[i]),1.0f), md);
    }
    md = block_reduce_md<FAST_EXP>(md,smem);
    if (md.x==-INFINITY) {
        for (int64_t i=offset+threadIdx.x;i<offset+N;i+=s) {
            y[i]=__float2half(0.0f);
        }
        return;
    }

    for (int64_t i=offset+threadIdx.x;i<offset+N;i+=s) {
        y[i] = __float2half(exp_f<FAST_EXP>(__half2float(x[i])-md.x)/md.y);
    }

}  // 2 passes

template <bool FAST_EXP> __global__ void softmax_b3_kernel(const __half* x, __half* y, int64_t N);  // 1 pass, optional

// ---- called from Python ----
torch::Tensor softmax_forward(torch::Tensor x, int64_t variant, int64_t block, bool fast_exp) {
    // 1 check: x is CUDA, Half, contiguous, 2-D
    // 2 y = at::empty_like(x); rows = x.size(0); N = x.size(1)
    // 3 stream = at::cuda::getCurrentCUDAStream()
    // 4 launch kernel<<<rows, block, smem_bytes, stream>>>(...) chosen by variant and fast_exp
    // 5 C10_CUDA_KERNEL_LAUNCH_CHECK(); return y
    if (variant == 0 || variant == 1) {
        CHECK_INPUT(x, at::kHalf);
        TORCH_CHECK(x.dim() == 2, "x must be 2-D (rows, N)");
        TORCH_CHECK(block > 0 && block <= 1024 && block % 32 == 0,
                    "block must be a multiple of 32 and at most 1024");
        torch::Tensor y = torch::empty_like(x);
        int64_t rows = x.size(0);
        int64_t N = x.size(1);
        TORCH_CHECK(N > 0, "N must be positive");
        if (rows == 0) return y;

        auto stream = at::cuda::getCurrentCUDAStream();
        const __half* x_ptr = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
        __half* y_ptr = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
        dim3 grid(static_cast<unsigned int>(rows));
        dim3 blk(static_cast<unsigned int>(block));

        if (variant == 0) {
            if (fast_exp) {
                softmax_b1_kernel<true><<<grid, blk, 0, stream>>>(x_ptr, y_ptr, N);
            } else {
                softmax_b1_kernel<false><<<grid, blk, 0, stream>>>(x_ptr, y_ptr, N);
            }
        } else {
            if (fast_exp) {
                softmax_b2_kernel<true><<<grid, blk, 0, stream>>>(x_ptr, y_ptr, N);
            } else {
                softmax_b2_kernel<false><<<grid, blk, 0, stream>>>(x_ptr, y_ptr, N);
            }
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    }
    return x;
}
int64_t max_smem_optin() { return 0; }   // cudaDeviceGetAttribute(..., cudaDevAttrMaxSharedMemoryPerBlockOptin, dev)