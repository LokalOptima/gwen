#include "gwen/kernels.h"
#include <cublas_v2.h>

namespace gwen {

static cublasHandle_t g_cublas_handle = nullptr;

void gwen_cublas_init(cudaStream_t stream) {
    if (!g_cublas_handle) {
        cublasCreate(&g_cublas_handle);
    }
    cublasSetStream(g_cublas_handle, stream);
}

void gwen_cublas_destroy() {
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
}

// GEMM: y = W * x^T where W is [out, in] and x is [seq_len, in]
// Result y is [seq_len, out]
// W_quant: quantized weight, temp_w: scratch for dequantized FP16 weight
// x: [seq_len, in_features] row-major, y: [seq_len, out_features] row-major
void gwen_gemm(const void* W_quant, GGMLType type,
               half* temp_w,
               const half* x, half* y,
               int out_features, int in_features, int seq_len,
               cudaStream_t stream) {
    // Step 1: Dequantize weight to FP16 in temp_w [out_features, in_features]
    int n_elements = out_features * in_features;
    gwen_dequant(W_quant, temp_w, n_elements, type, stream);

    // Step 2: cuBLAS HGEMM
    // We want: y[seq_len, out] = x[seq_len, in] * W^T[in, out]
    // cuBLAS uses column-major. With row-major matrices:
    //   C = A * B^T  in row-major
    //   = B * A^T  in column-major (cuBLAS)
    //
    // A = x (seq_len × in_features), row-major → col-major: in_features × seq_len
    // B = W (out_features × in_features), row-major → col-major: in_features × out_features
    // C = y (seq_len × out_features), row-major → col-major: out_features × seq_len
    //
    // We need: C_col = B_col^T * A_col
    //   i.e. out×seq = (in×out)^T * (in×seq)
    //   = out×in * in×seq = out×seq ✓
    //
    // So: cublas call is C = alpha * op(B) * op(A) + beta * C
    //   op(B) = B^T (CUBLAS_OP_T on B_col gives B_col^T = W row-major)
    //   op(A) = A (CUBLAS_OP_N on A_col = x col-major)
    //   m = out_features, n = seq_len, k = in_features
    //   lda = in_features (leading dim of A_col = x in col-major)
    //   ldb = in_features (leading dim of B_col = W in col-major)
    //   ldc = out_features (leading dim of C_col = y in col-major)

    gwen_cublas_init(stream);

    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);

    cublasHgemm(g_cublas_handle,
                CUBLAS_OP_T,    // op on W (B in col-major)
                CUBLAS_OP_N,    // op on x (A in col-major)
                out_features,   // m
                seq_len,        // n
                in_features,    // k
                &alpha_h,
                temp_w,         // W in col-major (= W row-major transposed)
                in_features,    // lda (leading dim of W col-major)
                x,              // x in col-major
                in_features,    // ldb (leading dim of x col-major)
                &beta_h,
                y,              // output
                out_features);  // ldc
}

} // namespace gwen
