// Fused softmax, version 2 (module fused_softmax_v2, built by ext_v2.py).
// B1, B2 and B3 kernels: copied unchanged from csrc/softmax_kernels.cu (v1).
// New in v2, written with an AI assistant after a review against FlashAttention:
//   B3 launch: the block is widened until the blocks that fit in shared memory fill the SM.
//   B4 (variant 3): B2's two passes with 16-byte loads, one (m, d) rescale per 8 values, exp2, reciprocal.
//   B5 (variant 4): B4's math with one warp per row (shuffles only, no shared memory, no __syncthreads).
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <math.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>   // std::max, std::min for the B3 block size
#include <cstdint>     // uintptr_t for the 16-byte alignment check

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
    __syncthreads();
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

constexpr float kLog2e = 1.4426950408889634f;

// BASE2 = false: e^v (B1, B2, B3, unchanged). BASE2 = true: 2^v (B4, B5 work in log2 units).
template <bool FAST_EXP, bool BASE2 = false>
__device__ __forceinline__ float exp_f(float v) {
    if constexpr (BASE2) {
        if constexpr (FAST_EXP) {
            float r;
            asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v));   // one MUFU.EX2
            return r;
        } else {
            return exp2f(v);                                        // 2 ulp
        }
    } else {
        if constexpr (FAST_EXP) return __expf(v);
        else return expf(v);
    }
}

template <bool FAST_EXP, bool BASE2 = false>
__device__ float2 md_combine(float2 a, float2 b) {
    float ma = a.x;
    float mb = b.x;
    float da = a.y;
    float db = b.y;
    if (ma==-INFINITY && mb==-INFINITY) return make_float2(-INFINITY, 0);
    float2 c = a;
    c.x = fmaxf(ma,mb);
    c.y = da * exp_f<FAST_EXP, BASE2>(ma-c.x) + db * exp_f<FAST_EXP, BASE2>(mb-c.x);
    return c;
}

template <bool FAST_EXP, bool BASE2 = false>
__device__ float2 warp_reduce_md(float2 v) {
    int laneMask = 16;
    while (laneMask>0) {
        float nbor_x = __shfl_xor_sync(0xffffffffu, v.x, laneMask);
        float nbor_y = __shfl_xor_sync(0xffffffffu, v.y, laneMask);
        v = md_combine<FAST_EXP, BASE2>(v, make_float2(nbor_x, nbor_y));
        laneMask>>=1;
    }
    return v;
}

template <bool FAST_EXP, bool BASE2 = false>
__device__ float2 block_reduce_md(float2 v, float2* smem) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int idx = warp;
    float2 val = warp_reduce_md<FAST_EXP, BASE2>(v);
    __syncthreads();
    if (lane==0) smem[idx] = val;
    __syncthreads();
    int warps = blockDim.x / 32;
    if (lane<warps) v = smem[lane];
    else {
        v.x = -INFINITY;
        v.y = 0;
    }
    float2 md = warp_reduce_md<FAST_EXP, BASE2>(v);

    return md;
}

// ---- B4 / B5 helpers: 16-byte loads, one (m, d) rescale per 8 values ----

// 8 halves (16 bytes) <-> 8 floats
__device__ __forceinline__ void unpack8(uint4 raw, float (&f)[8]) {
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        float2 p = __half22float2(h2[k]);
        f[2 * k] = p.x;
        f[2 * k + 1] = p.y;
    }
}

__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
    uint4 raw;
    __half2* h2 = reinterpret_cast<__half2*>(&raw);
    #pragma unroll
    for (int k = 0; k < 4; ++k) h2[k] = __floats2half2_rn(f[2 * k], f[2 * k + 1]);
    return raw;
}

// (m, d) of one 8-value chunk in log2 units: max first, then 8 exps against that max.
// This is FlashAttention's per-tile step: 1 exp per value, plus 2 in the md_combine that merges the chunk.
// B2 pays 2 exps per value, because it merges every value through md_combine.
template <bool FAST_EXP>
__device__ __forceinline__ float2 chunk_md(float (&f)[8]) {
    float cm = -INFINITY;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        f[k] *= kLog2e;
        cm = fmaxf(cm, f[k]);
    }
    if (cm == -INFINITY) return make_float2(-INFINITY, 0.0f);   // 8 masked values add nothing
    float cs = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) cs += exp_f<FAST_EXP, true>(f[k] - cm);
    return make_float2(cm, cs);
}

// Pass 1 for one thread (B4: t = threadIdx.x, nt = blockDim.x) or one lane (B5: t = lane, nt = 32).
template <bool FAST_EXP>
__device__ __forceinline__ float2 row_md(const __half* xr, int64_t N, bool vec, int t, int nt) {
    float2 md = make_float2(-INFINITY, 0.0f);
    if (vec) {
        const uint4* xv = reinterpret_cast<const uint4*>(xr);
        for (int64_t v = t; v < N / 8; v += nt) {
            float f[8];
            unpack8(__ldg(xv + v), f);
            md = md_combine<FAST_EXP, true>(md, chunk_md<FAST_EXP>(f));
        }
    } else {                                    // N not a multiple of 8, or unaligned pointer
        for (int64_t j = t; j < N; j += nt)
            md = md_combine<FAST_EXP, true>(md, make_float2(__half2float(xr[j]) * kLog2e, 1.0f));
    }
    return md;
}

// Pass 2: y = 2^(x·log2e − m) · (1/d). One FFMA + one EX2 + one FMUL per value, no division.
template <bool FAST_EXP>
__device__ __forceinline__ void write_row(const __half* xr, __half* yr, int64_t N, bool vec,
                                          int t, int nt, float2 md) {
    if (md.x == -INFINITY) {                    // all -inf row: zeros, lab convention
        for (int64_t j = t; j < N; j += nt) yr[j] = __float2half(0.0f);
        return;
    }
    const float inv_d = 1.0f / md.y;
    if (vec) {
        const uint4* xv = reinterpret_cast<const uint4*>(xr);
        uint4* yv = reinterpret_cast<uint4*>(yr);
        for (int64_t v = t; v < N / 8; v += nt) {
            float f[8];
            unpack8(__ldg(xv + v), f);
            #pragma unroll
            for (int k = 0; k < 8; ++k) f[k] = exp_f<FAST_EXP, true>(fmaf(f[k], kLog2e, -md.x)) * inv_d;
            yv[v] = pack8(f);
        }
    } else {
        for (int64_t j = t; j < N; j += nt)
            yr[j] = __float2half(exp_f<FAST_EXP, true>(fmaf(__half2float(xr[j]), kLog2e, -md.x)) * inv_d);
    }
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

template <bool FAST_EXP> __global__ void softmax_b3_kernel(const __half* x, __half* y, int64_t N) {
    extern __shared__ __half row[];
    int64_t offset = blockIdx.x * N;
    int64_t s = blockDim.x;
    __shared__ float2 smem[32];
    float2 md = make_float2(-INFINITY,0.0f);

    for (int64_t j = threadIdx.x; j < N; j += s) {
        __half h = x[offset + j];
        row[j] = h;
        md = md_combine<FAST_EXP>(make_float2(__half2float(h), 1.0f), md);
    }
    md = block_reduce_md<FAST_EXP>(md, smem);

    if (md.x == -INFINITY) {               // all -inf row: write zeros, same rule as B1 and B2
        for (int64_t j = threadIdx.x; j < N; j += s) {
            y[offset + j] = __float2half(0.0f);
        }
        return;
    }

    // second pass reads shared memory, not DRAM
    for (int64_t j = threadIdx.x; j < N; j += s) {
        y[offset + j] = __float2half(exp_f<FAST_EXP>(__half2float(row[j]) - md.x) / md.y);
    }
}  // 1 pass, optional

// B4: B2's two passes with FlashAttention's inner-loop choices.
template <bool FAST_EXP>
__global__ void softmax_b4_kernel(const __half* __restrict__ x, __half* __restrict__ y, int64_t N, bool vec) {
    __shared__ float2 smem[32];
    const __half* xr = x + static_cast<int64_t>(blockIdx.x) * N;
    __half* yr = y + static_cast<int64_t>(blockIdx.x) * N;
    float2 md = row_md<FAST_EXP>(xr, N, vec, threadIdx.x, blockDim.x);
    md = block_reduce_md<FAST_EXP, true>(md, smem);
    write_row<FAST_EXP>(xr, yr, N, vec, threadIdx.x, blockDim.x, md);
}  // 2 passes, 16-byte loads

// B5: one warp per row. Shuffles only: no shared memory, no __syncthreads.
template <bool FAST_EXP>
__global__ void softmax_b5_kernel(const __half* __restrict__ x, __half* __restrict__ y,
                                  int64_t rows, int64_t N, bool vec) {
    const int lane = threadIdx.x & 31;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;          // all 32 lanes share `row`, so the whole warp leaves together
    const __half* xr = x + row * N;
    __half* yr = y + row * N;
    float2 md = row_md<FAST_EXP>(xr, N, vec, lane, 32);
    md = warp_reduce_md<FAST_EXP, true>(md);
    write_row<FAST_EXP>(xr, yr, N, vec, lane, 32, md);
}  // 2 passes, a row per warp; best for short rows

int64_t max_smem_optin_v2();              // defined at the end of the file
int64_t b3_block(int64_t N, int64_t block);  // defined at the end of the file

// ---- called from Python ----
torch::Tensor softmax_forward_v2(torch::Tensor x, int64_t variant, int64_t block, bool fast_exp) {
    const c10::cuda::CUDAGuard guard(x.device());

    CHECK_INPUT(x, at::kHalf);
    TORCH_CHECK(x.dim() == 2, "x must be 2-D (rows, N)");
    TORCH_CHECK(block > 0 && block <= 1024 && block % 32 == 0,
                "block must be a multiple of 32 and at most 1024");
    TORCH_CHECK(variant >= 0 && variant <= 4, "variant must be 0 to 4");
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

    const bool vec = (N % 8 == 0)
    && (reinterpret_cast<uintptr_t>(x_ptr) % 16 == 0)
    && (reinterpret_cast<uintptr_t>(y_ptr) % 16 == 0);

    if (variant == 0) {
        if (fast_exp) {
            softmax_b1_kernel<true><<<grid, blk, 0, stream>>>(x_ptr, y_ptr, N);
        } else {
            softmax_b1_kernel<false><<<grid, blk, 0, stream>>>(x_ptr, y_ptr, N);
        }
    } else if (variant == 1) {
        if (fast_exp) {
            softmax_b2_kernel<true><<<grid, blk, 0, stream>>>(x_ptr, y_ptr, N);
        } else {
            softmax_b2_kernel<false><<<grid, blk, 0, stream>>>(x_ptr, y_ptr, N);
        }
    } else if (variant == 2) {
        int64_t bytes = N * static_cast<int64_t>(sizeof(__half));      // the row
        int64_t scratch = 32 * static_cast<int64_t>(sizeof(float2));   // smem[32] inside the kernel
        TORCH_CHECK(bytes + scratch <= max_smem_optin_v2(),
                    "variant 2: row does not fit in shared memory");
        dim3 blk3(static_cast<unsigned int>(b3_block(N, block)));   // widened block, see b3_block
        if (fast_exp) {
            if (bytes + scratch > 48 * 1024)
                C10_CUDA_CHECK(cudaFuncSetAttribute(softmax_b3_kernel<true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
            softmax_b3_kernel<true><<<grid, blk3, bytes, stream>>>(x_ptr, y_ptr, N);
        } else {
            if (bytes + scratch > 48 * 1024)
                C10_CUDA_CHECK(cudaFuncSetAttribute(softmax_b3_kernel<false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
            softmax_b3_kernel<false><<<grid, blk3, bytes, stream>>>(x_ptr, y_ptr, N);
        }
    } else if (variant == 3) {
        if (fast_exp) softmax_b4_kernel<true><<<grid, blk, 0, stream>>>(x_ptr, y_ptr, N, vec);
        else          softmax_b4_kernel<false><<<grid, blk, 0, stream>>>(x_ptr, y_ptr, N, vec);
    } else {
        // B5: block / 32 rows per block, one warp each
        const int64_t rows_per_block = block / 32;
        dim3 grid5(static_cast<unsigned int>((rows + rows_per_block - 1) / rows_per_block));
        if (fast_exp) softmax_b5_kernel<true><<<grid5, blk, 0, stream>>>(x_ptr, y_ptr, rows, N, vec);
        else          softmax_b5_kernel<false><<<grid5, blk, 0, stream>>>(x_ptr, y_ptr, rows, N, vec);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

int64_t max_smem_optin_v2() {
    int dev;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    int v;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    return v;
}

// B3 threads per block: keep the number of blocks per SM that shared memory allows, and widen each block
// until those blocks fill the SM's threads. `block` is a lower bound. On sm_75 (64 KiB smem, 1,024 threads
// per SM): S = 16384 -> 1 block of 1024; S = 8192 -> 3 blocks of 320; S <= 4096 -> `block` unchanged.
int64_t b3_block(int64_t N, int64_t block) {
    int dev;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    int smem_per_sm, threads_per_sm;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
    const int64_t bytes = N * static_cast<int64_t>(sizeof(__half)) + 32 * static_cast<int64_t>(sizeof(float2));
    const int64_t fit = std::max<int64_t>(1, smem_per_sm / bytes);              // blocks per SM by smem
    const int64_t wide = (std::min<int64_t>(1024, threads_per_sm) / fit) / 32 * 32;
    return std::min<int64_t>(1024, std::max<int64_t>(block, wide));
}
