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

// Batch2 SwiGLU: process two tokens in one launch (blockIdx.y selects token)
__global__ void __launch_bounds__(256)
kernel_swiglu_batch2(
    const half* __restrict__ gate0, const half* __restrict__ gate1,
    const half* __restrict__ up0, const half* __restrict__ up1,
    half* __restrict__ y0, half* __restrict__ y1, int n) {
    const half* gate = (blockIdx.y == 0) ? gate0 : gate1;
    const half* up = (blockIdx.y == 0) ? up0 : up1;
    half* y = (blockIdx.y == 0) ? y0 : y1;
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float sig = 1.0f / (1.0f + expf(-g));
        y[idx] = __float2half(g * sig * u);
    }
}

void gwen_swiglu_batch2(const half* gate0, const half* gate1,
                         const half* up0, const half* up1,
                         half* y0, half* y1,
                         int n, cudaStream_t stream) {
    dim3 grid((n + 255) / 256, 2);
    kernel_swiglu_batch2<<<grid, 256, 0, stream>>>(gate0, gate1, up0, up1, y0, y1, n);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_swiglu_quantize_q8_1(const half* gate, const half* up, void* y_q8, int n, cudaStream_t stream) {
    int n_blocks = n / 32;
    int grid = (n_blocks + 7) / 8;  // 8 warps per thread block
    kernel_swiglu_quantize_q8_1<<<grid, 256, 0, stream>>>(
        gate, up, static_cast<block_q8_1*>(y_q8), n_blocks);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

// Batch2: process two independent SwiGLU+Q8_1 in a single launch
// blockIdx.y selects token A (0) or B (1)
__global__ void __launch_bounds__(256)
kernel_swiglu_quantize_q8_1_batch2(
        const half* __restrict__ gate_a, const half* __restrict__ gate_b,
        const half* __restrict__ up_a, const half* __restrict__ up_b,
        block_q8_1* __restrict__ y_q8_a, block_q8_1* __restrict__ y_q8_b,
        int n_blocks) {
    const half* gate = (blockIdx.y == 0) ? gate_a : gate_b;
    const half* up = (blockIdx.y == 0) ? up_a : up_b;
    block_q8_1* y_q8 = (blockIdx.y == 0) ? y_q8_a : y_q8_b;

    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_global >= n_blocks) return;

    int base = warp_global * 32 + lane;

    float g = __half2float(gate[base]);
    float u = __half2float(up[base]);
    float sig = 1.0f / (1.0f + expf(-g));
    float val = g * sig * u;

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

void gwen_swiglu_quantize_q8_1_batch2(
        const half* gate_a, const half* gate_b,
        const half* up_a, const half* up_b,
        void* y_q8_a, void* y_q8_b,
        int n, cudaStream_t stream) {
    int n_blocks = n / 32;
    int grid_x = (n_blocks + 7) / 8;
    dim3 grid(grid_x, 2);
    kernel_swiglu_quantize_q8_1_batch2<<<grid, 256, 0, stream>>>(
        gate_a, gate_b, up_a, up_b,
        static_cast<block_q8_1*>(y_q8_a), static_cast<block_q8_1*>(y_q8_b),
        n_blocks);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

// Batch2: quantize two independent FP16 vectors to Q8_1 in a single launch
__global__ void __launch_bounds__(256)
kernel_quantize_q8_1_batch2(
        const half* __restrict__ x_a, const half* __restrict__ x_b,
        block_q8_1* __restrict__ y_a, block_q8_1* __restrict__ y_b,
        int n_blocks) {
    const half* x = (blockIdx.y == 0) ? x_a : x_b;
    block_q8_1* y = (blockIdx.y == 0) ? y_a : y_b;

    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_global >= n_blocks) return;

    int base = warp_global * 32 + lane;
    float val = __half2float(x[base]);

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

    y[warp_global].qs[lane] = q;
    if (lane == 0)
        y[warp_global].ds = __halves2half2(__float2half(d), __float2half(sum));
}

void gwen_quantize_q8_1_batch2(
        const half* x_a, const half* x_b,
        void* y_q8_a, void* y_q8_b,
        int n, cudaStream_t stream) {
    int n_blocks = n / 32;
    int grid_x = (n_blocks + 7) / 8;
    dim3 grid(grid_x, 2);
    kernel_quantize_q8_1_batch2<<<grid, 256, 0, stream>>>(
        x_a, x_b,
        static_cast<block_q8_1*>(y_q8_a), static_cast<block_q8_1*>(y_q8_b),
        n_blocks);
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
