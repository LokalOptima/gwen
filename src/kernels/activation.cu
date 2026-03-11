#include "gwen/kernels.h"

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
