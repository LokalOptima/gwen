#pragma once

#include "gwen/common.h"
#include <cuda_fp16.h>

namespace gwen {

// ============================================================
// Dequantization: quantized blocks → FP16 output
// ============================================================

// Dequantize Q4_K blocks to FP16
// src: device pointer to Q4_K data, dst: device pointer to half output
// n: number of elements (must be multiple of 256)
void gwen_dequant_q4_k(const void* src, half* dst, int n, cudaStream_t stream = 0);

// Dequantize Q5_K blocks to FP16
void gwen_dequant_q5_k(const void* src, half* dst, int n, cudaStream_t stream = 0);

// Dequantize Q6_K blocks to FP16
void gwen_dequant_q6_k(const void* src, half* dst, int n, cudaStream_t stream = 0);

// Dequantize Q8_0 blocks to FP16
void gwen_dequant_q8_0(const void* src, half* dst, int n, cudaStream_t stream = 0);

// Dequantize F32 to FP16
void gwen_dequant_f32(const float* src, half* dst, int n, cudaStream_t stream = 0);

// Generic dequant dispatch by type
void gwen_dequant(const void* src, half* dst, int n, GGMLType type, cudaStream_t stream = 0);

// ============================================================
// Fused Dequant + GEMV: y = W * x (decode path)
// ============================================================
// W is [out_features, in_features] in quantized format
// x is [in_features] in FP16
// y is [out_features] in FP16

void gwen_gemv_q4_k(const void* W, const half* x, half* y,
                     int out_features, int in_features, cudaStream_t stream = 0);

void gwen_gemv_q5_k(const void* W, const half* x, half* y,
                     int out_features, int in_features, cudaStream_t stream = 0);

void gwen_gemv_q6_k(const void* W, const half* x, half* y,
                     int out_features, int in_features, cudaStream_t stream = 0);

void gwen_gemv_q8_0(const void* W, const half* x, half* y,
                     int out_features, int in_features, cudaStream_t stream = 0);

// Generic GEMV dispatch
void gwen_gemv(const void* W, const half* x, half* y,
               int out_features, int in_features, GGMLType type, cudaStream_t stream = 0);

// ============================================================
// dp4a-accelerated GEMV (quantize input to Q8_1, use integer SIMD)
// ============================================================

// Quantize FP16 input vector to Q8_1 blocks for dp4a GEMV
// x: [n] in FP16, x_q8: [n/32] block_q8_1 output
// n must be a multiple of 32
void gwen_quantize_q8_1(const half* x, void* x_q8, int n, cudaStream_t stream = 0);

// dp4a GEMV: y = W * x, where x is pre-quantized to Q8_1
void gwen_gemv_dp4a(const void* W, const void* x_q8, half* y,
                    int out_features, int in_features, GGMLType type, cudaStream_t stream = 0);

// dp4a GEMV with fused residual add: y = W * x + residual
void gwen_gemv_dp4a_residual(const void* W, const void* x_q8, half* y, const half* residual,
                              int out_features, int in_features, GGMLType type, cudaStream_t stream = 0);

// Batch-2 dp4a GEMV: read quantized weights ONCE, produce 2 outputs
// y0 = W * x_q8_0 [+ res0], y1 = W * x_q8_1 [+ res1]
void gwen_gemv_dp4a_batch2(const void* W,
                            const void* x_q8_0, const void* x_q8_1,
                            half* y0, half* y1,
                            int out_features, int in_features,
                            GGMLType type, cudaStream_t stream = 0);

void gwen_gemv_dp4a_residual_batch2(const void* W,
                                     const void* x_q8_0, const void* x_q8_1,
                                     half* y0, half* y1,
                                     const half* res0, const half* res1,
                                     int out_features, int in_features,
                                     GGMLType type, cudaStream_t stream = 0);

// ============================================================
// FP16 GEMV (for MTP weights stored as FP16)
// ============================================================

// y = W * x, where W is [out_features, in_features] in FP16
void gwen_gemv_fp16(const half* W, const half* x, half* y,
                    int out_features, int in_features, cudaStream_t stream = 0);

// y = W * x + residual (fused residual add)
void gwen_gemv_fp16_residual(const half* W, const half* x, half* y, const half* residual,
                              int out_features, int in_features, cudaStream_t stream = 0);

// ============================================================
// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
// ============================================================
void gwen_rmsnorm(const half* x, const half* weight, half* y,
                  int dim, float eps, cudaStream_t stream = 0);

// Fused RMSNorm for F32 weights (common for norm layers stored as F32)
void gwen_rmsnorm_f32w(const half* x, const float* weight, half* y,
                       int dim, float eps, cudaStream_t stream = 0);

// Fused RMSNorm + Q8_1 quantize (with optional FP16 output)
// y_fp16 may be nullptr if FP16 output not needed
void gwen_rmsnorm_quantize_q8_1(const half* x, const float* weight, void* y_q8, half* y_fp16,
                                  int dim, float eps, cudaStream_t stream = 0);

// Batch2: fused RMSNorm + Q8_1 for two independent vectors in one launch
void gwen_rmsnorm_quantize_q8_1_batch2(
        const half* x_a, const half* x_b, const float* weight,
        void* y_q8_a, void* y_q8_b, half* y_fp16_a, half* y_fp16_b,
        int dim, float eps, cudaStream_t stream = 0);

// Batched RMSNorm: normalize n_vecs vectors of length dim
// x: [n_vecs, dim], y: [n_vecs, dim], weight: [dim] (shared across vectors)
void gwen_rmsnorm_batched_f32w(const half* x, const float* weight, half* y,
                               int n_vecs, int dim, float eps, cudaStream_t stream = 0);

// ============================================================
// Activations
// ============================================================

// SiLU: y = x * sigmoid(x)
void gwen_silu(const half* x, half* y, int n, cudaStream_t stream = 0);

// SiLU inplace: x = x * sigmoid(x)
void gwen_silu_inplace(half* x, int n, cudaStream_t stream = 0);

// SwiGLU: y = SiLU(gate) * up (fused)
void gwen_swiglu(const half* gate, const half* up, half* y, int n, cudaStream_t stream = 0);

// Fused SwiGLU + Q8_1 quantize: computes SwiGLU and writes Q8_1 directly (skips FP16 intermediate)
void gwen_swiglu_quantize_q8_1(const half* gate, const half* up, void* y_q8, int n, cudaStream_t stream = 0);

// Batch2: fused SwiGLU + Q8_1 for two independent inputs in one launch
void gwen_swiglu_quantize_q8_1_batch2(
        const half* gate_a, const half* gate_b,
        const half* up_a, const half* up_b,
        void* y_q8_a, void* y_q8_b,
        int n, cudaStream_t stream = 0);

// Batch2: Q8_1 quantize two independent FP16 vectors in one launch
void gwen_quantize_q8_1_batch2(
        const half* x_a, const half* x_b,
        void* y_q8_a, void* y_q8_b,
        int n, cudaStream_t stream = 0);

// Sigmoid: y = 1 / (1 + exp(-x))
void gwen_sigmoid(const half* x, half* y, int n, cudaStream_t stream = 0);

// Element-wise multiply: y = a * b
void gwen_mul(const half* a, const half* b, half* y, int n, cudaStream_t stream = 0);

// Fused sigmoid-mul: y = a * sigmoid(b)
void gwen_sigmoid_mul(const half* a, const half* b, half* y, int n, cudaStream_t stream = 0);

// Element-wise add: y = a + b
void gwen_add(const half* a, const half* b, half* y, int n, cudaStream_t stream = 0);

// Residual add inplace: x += residual
void gwen_add_inplace(half* x, const half* residual, int n, cudaStream_t stream = 0);

// ============================================================
// Softmax
// ============================================================

// Online softmax over last dimension
// x: [rows, cols], y: [rows, cols]
void gwen_softmax(const half* x, half* y, int rows, int cols, cudaStream_t stream = 0);

// Causal masked softmax (for attention)
// x: [n_heads, seq_len, seq_len], y: same shape
// Each row is masked: positions > current_pos are -inf before softmax
void gwen_causal_softmax(const float* x, half* y, int n_heads, int seq_len,
                         cudaStream_t stream = 0);

// ============================================================
// RoPE (Rotary Position Encoding)
// ============================================================

// imRoPE: interleaved multi-axis RoPE for Qwen3.5
// q: [n_heads, head_dim], k: [n_kv_heads, head_dim]
// sections: [s0, s1, s2, s3] — how many pairs per axis
// Only first rope_dim dims of head_dim are rotated
// d_pos: device pointer to position (for CUDA graph compatibility)
void gwen_rope(half* q, half* k,
               int n_heads, int n_kv_heads, int head_dim,
               const int* d_pos, float theta,
               const int* sections, int rope_dim,
               cudaStream_t stream = 0);

// ============================================================
// Embedding lookup
// ============================================================

// Look up token embedding: y = embedding_table[token_id]
// table: quantized embedding [n_vocab, dim]
// d_token_id: device pointer to token ID (for CUDA graph compatibility)
void gwen_embed_lookup(const void* table, GGMLType table_type,
                       const int* d_token_id, half* y, int dim,
                       cudaStream_t stream = 0);

// ============================================================
// L2 Normalization
// ============================================================

// L2-normalize vectors: y = (x / ||x||_2) * extra_scale
// x: [n_vecs, dim], y: [n_vecs, dim]
// extra_scale folds in e.g. 1/sqrt(d_k) for DeltaNet Q scaling
void gwen_l2_normalize(const half* x, half* y, int n_vecs, int dim,
                       float extra_scale = 1.0f, cudaStream_t stream = 0);

// ============================================================
// DeltaNet recurrence (exposed for profiling)
// ============================================================
void gwen_deltanet_decode(float* S, const half* q, const half* k, const half* v,
                          const float* alpha, const float* beta, half* output,
                          int n_heads, int dk, int dv, cudaStream_t stream = 0);

// ============================================================
// GEMM (for prefill — batch of tokens)
// ============================================================

// Dequant + CUTLASS GEMM: y[seq_len, out] = x[seq_len, in] * W^T[in, out]
// temp_w: scratch for dequantized FP16 weights [out_features * in_features]
void gwen_gemm(const void* W_quant, GGMLType type,
               half* temp_w,
               const half* x, half* y,
               int out_features, int in_features, int seq_len,
               cudaStream_t stream = 0);

// F32 output variant: same as gwen_gemm but writes F32 output (no FP16 truncation)
void gwen_gemm_f32out(const void* W_quant, GGMLType type,
                       half* temp_w,
                       const half* x, float* y,
                       int out_features, int in_features, int seq_len,
                       cudaStream_t stream = 0);

// ============================================================
// Row dequantization (selective rows from quantized matrix)
// ============================================================

// Dequantize K specific rows from a Q6_K embedding table to FP16.
// table: quantized embedding [n_vocab, dim] in Q6_K format
// row_ids: [K] int32 token IDs (device pointer) specifying which rows to extract
// dst: [K, dim] FP16 output (device pointer, contiguous)
// K: number of rows to dequantize
// dim: embedding dimension (must be multiple of 256)
void gwen_dequant_rows_q6k(const void* table, const int* row_ids, half* dst,
                            int K, int dim, cudaStream_t stream = 0);

// ============================================================
// FP16 GEMM (no dequant — weights already in FP16)
// ============================================================

// y[seq_len, out] = x[seq_len, in] * W^T[in, out]
// W_fp16: [out_features, in_features] FP16 (already dequantized)
void gwen_gemm_fp16(const half* W_fp16, const half* x, half* y,
                     int out_features, int in_features, int seq_len,
                     cudaStream_t stream = 0);

// ============================================================
// Reduction: logsumexp + p_idk (for training server p_idk)
// ============================================================

// Row-wise logsumexp: x[n_rows, n_cols] FP16 → log_Z[n_rows] F32
// One block (256 threads) per row. Used for 248K vocab logits.
void gwen_logsumexp_rows(const half* x, float* log_Z,
                          int n_rows, int n_cols, cudaStream_t stream = 0);

// Compute p_idk from restricted logits + log_Z (partition function)
// restricted_logits[n_rows, K] FP16, log_Z[n_rows] F32 → p_idk[n_rows] F32
// p_idk = clamp(1 - sum(exp(restricted_logits - log_Z)), 0, 1)
void gwen_p_idk_from_logits(const half* restricted_logits, const float* log_Z,
                              float* p_idk, int n_rows, int K, cudaStream_t stream = 0);

// ============================================================
// Top-k selection (for sparse distillation)
// ============================================================

// Select top-k values and indices from each row.
// logits: [n_rows, K] FP16 input
// topk_indices: [n_rows, k] uint16 output (column indices)
// topk_values: [n_rows, k] FP16 output (corresponding values)
// Uses bitonic sort in shared memory (K * 4 bytes shared mem).
void gwen_topk(const half* logits, uint16_t* topk_indices, half* topk_values,
               int n_rows, int K, int k, cudaStream_t stream = 0);


} // namespace gwen
