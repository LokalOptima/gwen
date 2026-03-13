#include "gwen/kernels.h"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>

namespace gwen {

// CUTLASS 2.x GEMM targeting Sm80 (generates mma.sync, works on Sm120)
// Drop-in replacement for cuBLAS HGEMM.
//
// Layout mapping:
//   We want: y[seq_len, out] = x[seq_len, in] * W^T[in, out]
//   Equivalently: C[out, seq] = A[out, in] * B[in, seq]
//   A = temp_w [out_features, in_features] RowMajor
//   B = x [in_features, seq_len] ColumnMajor (i.e. x is [seq_len, in] row-major)
//   C = y [out_features, seq_len] ColumnMajor (i.e. y is [seq_len, out] row-major)

using CutlassGemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::RowMajor,                 // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    cutlass::half_t,                           // ElementC
    cutlass::layout::ColumnMajor,              // LayoutC
    float,                                     // ElementAccumulator
    cutlass::arch::OpClassTensorOp,            // OperatorClass
    cutlass::arch::Sm80,                       // ArchTag (mma.sync — works on Sm120)
    cutlass::gemm::GemmShape<128, 128, 32>,    // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 32>,      // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>        // InstructionShape
>;

void gwen_gemm(const void* W_quant, GGMLType type,
               half* temp_w,
               const half* x, half* y,
               int out_features, int in_features, int seq_len,
               cudaStream_t stream) {
    // Step 1: Dequantize weight to FP16 in temp_w [out_features, in_features]
    int n_elements = out_features * in_features;
    gwen_dequant(W_quant, temp_w, n_elements, type, stream);

    // Step 2: CUTLASS GEMM
    // C[M,N] = A[M,K] * B[K,N]
    // M = out_features, N = seq_len, K = in_features
    int M = out_features;
    int N = seq_len;
    int K = in_features;

    CutlassGemm gemm_op;
    CutlassGemm::Arguments args(
        {M, N, K},                                              // problem size
        {reinterpret_cast<cutlass::half_t*>(temp_w), K},        // A: [M, K] RowMajor, lda=K
        {reinterpret_cast<const cutlass::half_t*>(x), K},       // B: [K, N] ColumnMajor, ldb=K
        {reinterpret_cast<cutlass::half_t*>(y), M},             // C: [M, N] ColumnMajor, ldc=M
        {reinterpret_cast<cutlass::half_t*>(y), M},             // D: [M, N] ColumnMajor, ldd=M
        {1.0f, 0.0f}                                            // alpha, beta
    );

    cutlass::Status status = gemm_op(args, nullptr, stream);
    GWEN_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM failed");
}

// F32 output variant: same computation but stores F32 result without FP16 truncation.
// Used by the verified reference path (GWEN_GEMM_DECODE=1) for correctness vs llama.cpp.
using CutlassGemmF32Out = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // ElementA (dequantized weights)
    cutlass::layout::RowMajor,
    cutlass::half_t,                           // ElementB (activations)
    cutlass::layout::ColumnMajor,
    float,                                     // ElementC — F32 output!
    cutlass::layout::ColumnMajor,
    float,                                     // ElementAccumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>
>;

void gwen_gemm_f32out(const void* W_quant, GGMLType type,
                       half* temp_w,
                       const half* x, float* y,
                       int out_features, int in_features, int seq_len,
                       cudaStream_t stream) {
    int n_elements = out_features * in_features;
    gwen_dequant(W_quant, temp_w, n_elements, type, stream);

    int M = out_features;
    int N = seq_len;
    int K = in_features;

    CutlassGemmF32Out gemm_op;
    CutlassGemmF32Out::Arguments args(
        {M, N, K},
        {reinterpret_cast<cutlass::half_t*>(temp_w), K},
        {reinterpret_cast<const cutlass::half_t*>(x), K},
        {y, M},                                                    // C: float*
        {y, M},                                                    // D: float*
        {1.0f, 0.0f}
    );

    cutlass::Status status = gemm_op(args, nullptr, stream);
    GWEN_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM F32 output failed");
}

} // namespace gwen
