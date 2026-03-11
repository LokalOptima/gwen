#include "gwen/kernels.h"
#include "gwen/ggml_quants.h"

namespace gwen {

// ============================================================
// SiLU: y = x * sigmoid(x)
// ============================================================
__global__ void __launch_bounds__(256)
kernel_silu(const half* __restrict__ x, half* __restrict__ y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(x[idx]);
        float sig = 1.0f / (1.0f + expf(-val));
        y[idx] = __float2half(val * sig);
    }
}

__global__ void __launch_bounds__(256)
kernel_silu_inplace(half* __restrict__ x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(x[idx]);
        float sig = 1.0f / (1.0f + expf(-val));
        x[idx] = __float2half(val * sig);
    }
}

// ============================================================
// SwiGLU: y = SiLU(gate) * up
// ============================================================
__global__ void __launch_bounds__(256)
kernel_swiglu(const half* __restrict__ gate, const half* __restrict__ up,
              half* __restrict__ y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float sig = 1.0f / (1.0f + expf(-g));
        y[idx] = __float2half(g * sig * u);
    }
}

// ============================================================
// Fused SwiGLU + Q8_1 Quantize: computes SwiGLU and quantizes directly to Q8_1
// ============================================================
// Each warp handles one Q8_1 block (32 elements).
// Skips the FP16 intermediate entirely.
__global__ void __launch_bounds__(256)
kernel_swiglu_quantize_q8_1(const half* __restrict__ gate,
                             const half* __restrict__ up,
                             block_q8_1* __restrict__ y_q8,
                             int n_blocks) {
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_global >= n_blocks) return;

    int base = warp_global * 32 + lane;

    // Compute SwiGLU in FP32
    float g = __half2float(gate[base]);
    float u = __half2float(up[base]);
    float sig = 1.0f / (1.0f + expf(-g));
    float val = g * sig * u;

    // Warp reduction for amax
    float amax = fabsf(val);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));

    float d = amax / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    int8_t q = (int8_t)roundf(val * id);

    float sum = (float)q;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);

    y_q8[warp_global].qs[lane] = q;
    if (lane == 0)
        y_q8[warp_global].ds = __halves2half2(__float2half(d), __float2half(sum));
}

// ============================================================
// Sigmoid: y = 1 / (1 + exp(-x))
// ============================================================
__global__ void __launch_bounds__(256)
kernel_sigmoid(const half* __restrict__ x, half* __restrict__ y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(x[idx]);
        y[idx] = __float2half(1.0f / (1.0f + expf(-val)));
    }
}

// ============================================================
// Element-wise multiply: y = a * b
// ============================================================
__global__ void __launch_bounds__(256)
kernel_mul(const half* __restrict__ a, const half* __restrict__ b,
           half* __restrict__ y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = __hmul(a[idx], b[idx]);
    }
}

// ============================================================
// Element-wise add: y = a + b
// ============================================================
__global__ void __launch_bounds__(256)
kernel_add(const half* __restrict__ a, const half* __restrict__ b,
           half* __restrict__ y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = __hadd(a[idx], b[idx]);
    }
}

// Residual add inplace: x += residual
__global__ void __launch_bounds__(256)
kernel_add_inplace(half* __restrict__ x, const half* __restrict__ residual, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = __hadd(x[idx], residual[idx]);
    }
}

// ============================================================
// Fused sigmoid-mul: y = a * sigmoid(b)
// ============================================================
__global__ void __launch_bounds__(256)
kernel_sigmoid_mul(const half* __restrict__ a, const half* __restrict__ b,
                   half* __restrict__ y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        float sig = 1.0f / (1.0f + expf(-vb));
        y[idx] = __float2half(va * sig);
    }
}

// ============================================================
// Launch wrappers
// ============================================================

void gwen_silu(const half* x, half* y, int n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    kernel_silu<<<blocks, 256, 0, stream>>>(x, y, n);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_silu_inplace(half* x, int n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    kernel_silu_inplace<<<blocks, 256, 0, stream>>>(x, n);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_swiglu(const half* gate, const half* up, half* y, int n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    kernel_swiglu<<<blocks, 256, 0, stream>>>(gate, up, y, n);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_swiglu_quantize_q8_1(const half* gate, const half* up, void* y_q8, int n, cudaStream_t stream) {
    int n_blocks = n / 32;
    int grid = (n_blocks + 7) / 8;  // 8 warps per thread block
    kernel_swiglu_quantize_q8_1<<<grid, 256, 0, stream>>>(
        gate, up, static_cast<block_q8_1*>(y_q8), n_blocks);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_sigmoid(const half* x, half* y, int n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    kernel_sigmoid<<<blocks, 256, 0, stream>>>(x, y, n);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_mul(const half* a, const half* b, half* y, int n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    kernel_mul<<<blocks, 256, 0, stream>>>(a, b, y, n);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_add(const half* a, const half* b, half* y, int n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    kernel_add<<<blocks, 256, 0, stream>>>(a, b, y, n);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_add_inplace(half* x, const half* residual, int n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    kernel_add_inplace<<<blocks, 256, 0, stream>>>(x, residual, n);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_sigmoid_mul(const half* a, const half* b, half* y, int n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    kernel_sigmoid_mul<<<blocks, 256, 0, stream>>>(a, b, y, n);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

} // namespace gwen
