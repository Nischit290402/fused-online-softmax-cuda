// Fused softmax, version 2 (module fused_softmax_v2, built by ext_v2.py).
// B1, B2 and B3 kernels: copied unchanged from csrc/softmax_kernels.cu (v1).
// New in v2, written with an AI assistant after a review against FlashAttention:
//   B3 launch: the block is widened until the blocks that fit in shared memory fill the SM.
//   B4 (variant 3): B2's two passes with 16-byte loads, one (m, d) rescale per 8 values, exp2, reciprocal.
//   B5 (variant 4): B4's math with one warp per row (shuffles only, no shared memory, no __syncthreads).
// Study notes with worked numbers are at the end of this file.
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

/* =====================================================================================================
   STUDY NOTES FOR VERSION 2 (read these when you come back to this file)
   =====================================================================================================

   0. Map of this file
      B1 (variant 0), B2 (variant 1), B3 kernel (variant 2): copied unchanged from v1, written by hand.
      B3 launch: b3_block() at the end widens the block.                          See note 6.
      B4 (variant 3): softmax_b4_kernel = row_md + block_reduce_md + write_row.   See notes 1 to 5.
      B5 (variant 4): softmax_b5_kernel = row_md + warp_reduce_md + write_row.    See note 7.
      Helpers in order: exp_f (base e or base 2), md_combine / warp_reduce_md / block_reduce_md
      (BASE2 template flag), unpack8 / pack8, chunk_md, row_md, write_row.

   1. Why B4 loads 8 values at a time (uint4 = 16 bytes)
      v1 kernels read one __half per thread per load: a warp load moves 32 x 2 = 64 bytes.
      B4 reads one uint4 per thread: a warp load moves 32 x 16 = 512 bytes, 8x more per instruction.
      Nsight showed B1/B2 stalled on long_scoreboard (waiting for global loads), with DRAM only 25-41%
      busy. That means too few bytes were in flight. More bytes per load instruction is the direct fix.
      torch.softmax does the same: the 8 in cunn_SoftMaxForwardReg<..., 8> is 8 halves per load.

   2. chunk_md: one rescale per 8 values instead of one per value
      B2 folds every value into (m, d) through md_combine, which calls exp twice:
          d = d_a * e^(m_a - m) + d_b * e^(m_b - m)      -> 2 exps per value
      chunk_md first finds the chunk max cm, then sums 8 exps against cm, then merges the chunk once:
          8 exps + 2 exps (one md_combine) per 8 values  -> 1.25 exps per value
      This is the per-tile step of FlashAttention: tile max, exp of the tile, one rescale per tile.
      Worked example, base e for readability, two chunks seen by one thread:
          chunk A = [1, 3, 2, ...]  -> cm = 3, d = e^(1-3) + e^(3-3) + e^(2-3) + ...
          chunk B = [5, 0, 4, ...]  -> cm = 5, d = e^(5-5) + e^(0-5) + e^(4-5) + ...
          merge:  m = 5, d = d_A * e^(3-5) + d_B * e^(5-5)
      The result equals the one-value-at-a-time result exactly (in real arithmetic); only the number of
      exps changes. Masked chunk (all -inf): chunk_md returns (-inf, 0), md_combine skips it.

   3. Why the log2 domain (exp2 instead of exp)
      The GPU's special-function unit computes 2^x (instruction MUFU.EX2). e^x is rewritten as
          e^x = 2^(x * log2(e)),    log2(e) = 1.4426950408889634
      expf adds range-reduction instructions around that EX2. B4/B5 multiply every input by log2(e) once
      (in chunk_md, or in the scalar fallback), keep m in log2 units, and call 2^t directly.
      In write_row, x * log2(e) - m is a single fmaf (one FFMA instruction), then EX2, then one multiply.
      FlashAttention-2 does the same: it folds log2(e) into the softmax scale.
      Fast path: ex2.approx.ftz.f32 via inline PTX (what __expf uses internally). Accurate path: exp2f.

   4. Why multiply by 1/d instead of dividing by d
      Without --use_fast_math, fp32 division is IEEE-correct and costs several instructions per value.
      inv_d = 1.0f / md.y is computed once per row; each value then needs one FMUL.
      The difference is at most about 1 fp32 ulp (relative 2^-23), far below fp16 output rounding
      (relative 2^-11), so the fp16 result is the same in practice.

   5. When the 16-byte path is used (vec)
      vec = (N % 8 == 0) and x_ptr, y_ptr both 16-byte aligned. Every row then starts on a 16-byte
      boundary, because row r starts at r * N * 2 bytes and N * 2 is a multiple of 16.
      Otherwise the kernels take the scalar loop (one __half per thread, md_combine per value).
      Example of an unaligned but contiguous input: buf = randn(4 * 4096 + 1); x = buf[1:].view(4, 4096).
      x starts 2 bytes past an aligned address, so vec is false. tests/test_ext_v2.py covers this case.

   6. B3: why the launch widens the block (b3_block)
      B3 keeps the whole row in shared memory: 2 * N bytes + 256 bytes of reduction scratch per block.
      RTX 2070 Max-Q (sm_75): 64 KiB shared memory and 1,024 threads per SM.
          S = 16384: 32,768 + 256 = 33,024 B per block -> 1 block per SM fits.
                     v1 launched 256 threads = 8 warps of 32 possible -> 25% occupancy (ncu measured 25%).
                     b3_block: 1024 / 1 = 1024 threads -> 32 warps -> 100%.
          S = 8192:  16,640 B per block -> 3 blocks fit (4 x 16,640 > 65,536).
                     v1: 3 x 8 warps = 24 of 32 = 75% (ncu measured 74%).
                     b3_block: 1024 / 3 = 341 -> rounded down to 320 (a multiple of 32) -> 3 x 10 = 30 warps.
          S <= 4096: the requested block already fills the SM, so nothing changes.
      torch.softmax's cunn_SoftMaxForwardSmem runs 1,024 threads per block at S = 16384 for this reason.
      B3's kernel is unchanged. Each thread still re-reads only the row[j] values it wrote itself, so the
      second loop needs no extra __syncthreads.

   7. B5: one warp per row
      B1 to B4 split one row across 8 warps, so each reduction needs shared memory and two __syncthreads.
      B5 gives each row to one warp: 32 lanes stride over the row, then warp_reduce_md (5 shuffles,
      offsets 16, 8, 4, 2, 1) leaves the result in every lane. No shared memory, no block barrier.
      Launch: block / 32 rows per block (block 256 -> 8 rows per block), grid = ceil(rows / 8).
      `if (row >= rows) return;` is safe: row is the same for all 32 lanes of a warp, so the whole warp
      exits together, and there is no __syncthreads anywhere in the kernel that it could skip.
      Expected: helps short rows (S <= 2048, where torch's softmax_warp_forward beat B1: 92 us vs 147 us
      under ncu). Long rows (16384) give one warp a lot of serial work, so B4 should win there.
      FlashAttention-2 uses the same idea: rows go to warps, instead of splitting one row's reduction
      across warps.

   8. Edge cases (same rules as v1)
      All -inf row: block (or warp) result m = -inf, write_row writes zeros. Lab convention;
      torch.softmax returns NaN. The branch is uniform: every thread of the block (B4) or warp (B5)
      holds the same reduced md.
      Masked values inside a row: 2^(-inf - m) = 0, so they get probability 0.
      NaN: fmaxf skips NaN, but the NaN reaches the sum, so the row comes out NaN.

   9. What FlashAttention does that this file still does not
      * It fuses QK^T, softmax and PV, so S and P never reach DRAM. Here S is read from DRAM and P is
        written back: 4 bytes per element at best.
      * It keeps O unnormalized and divides by l once at the end. Here the output is P itself, so a
        second read of x (B4, B5) or an on-chip copy of the row (B3) is unavoidable.
      * It stores logsumexp = m + log(l) per row for the backward pass. Not done here.

   10. Check yourself (answers are in the notes above)
      Q: B2 calls exp how many times per value in pass 1? B4?             (2; 1.25)
      Q: Why does B3 at S = 8192 launch 320 threads and not 341?          (multiple of 32 = whole warps)
      Q: Why is B5's early return safe without a barrier?                  (uniform per warp, no __syncthreads)
      Q: What makes vec false for a contiguous tensor?                     (N % 8 != 0 or a 16-B misalignment)
      Q: Why fmaf(x, log2e, -m) instead of (x - m) * log2e?                (one FFMA; m is already in log2 units)
      Q: Which Nsight metric showed the load problem in v1?                (long_scoreboard stalls, DRAM 25-41%)
   ===================================================================================================== */
