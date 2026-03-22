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

// Tile config A: 128×128×32 — good for large M and N
using CutlassGemm_128x128 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>
>;

// Tile config B: 128×64×32 — 2x more N-tiles for small seq_len (N≤512)
using CutlassGemm_128x64 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>
>;

// FP16 weight GEMM — weights already in FP16 (no dequant).
// Used for restricted_embed and fine-tuned lm_head.
template<typename GemmType>
static void run_gemm_fp16(const half* W_fp16, const half* x, half* y,
                           int M, int K, int N, cudaStream_t stream) {
    GemmType gemm_op;
    typename GemmType::Arguments args(
        {M, N, K},
        {reinterpret_cast<const cutlass::half_t*>(W_fp16), K},
        {reinterpret_cast<const cutlass::half_t*>(x), K},
        {reinterpret_cast<cutlass::half_t*>(y), M},
        {reinterpret_cast<cutlass::half_t*>(y), M},
        {1.0f, 0.0f}
    );
    cutlass::Status status = gemm_op(args, nullptr, stream);
    GWEN_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM FP16 failed");
}

void gwen_gemm_fp16(const half* W_fp16, const half* x, half* y,
                     int out_features, int in_features, int seq_len,
                     cudaStream_t stream) {
    int M = out_features, K = in_features, N = seq_len;
    // Auto-select tile config based on problem size.
    // For small GEMMs (few CTA tiles), use 128×64 for better parallelism.
    // For large GEMMs, 128×128 has better arithmetic intensity.
    int n_blocks_large = ((M + 127) / 128) * ((N + 127) / 128);
    if (n_blocks_large < 140) {
        run_gemm_fp16<CutlassGemm_128x64>(W_fp16, x, y, M, K, N, stream);
    } else {
        run_gemm_fp16<CutlassGemm_128x128>(W_fp16, x, y, M, K, N, stream);
    }
}

} // namespace gwen
