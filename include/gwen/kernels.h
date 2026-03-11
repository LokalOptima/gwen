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
// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
// ============================================================
void gwen_rmsnorm(const half* x, const half* weight, half* y,
                  int dim, float eps, cudaStream_t stream = 0);

// Fused RMSNorm for F32 weights (common for norm layers stored as F32)
void gwen_rmsnorm_f32w(const half* x, const float* weight, half* y,
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
// GEMM (for prefill — batch of tokens)
// ============================================================

// Initialize cuBLAS (called once)
void gwen_cublas_init(cudaStream_t stream = 0);
void gwen_cublas_destroy();

// Dequant + cuBLAS HGEMM: y[seq_len, out] = x[seq_len, in] * W^T[in, out]
// temp_w: scratch for dequantized FP16 weights [out_features * in_features]
void gwen_gemm(const void* W_quant, GGMLType type,
               half* temp_w,
               const half* x, half* y,
               int out_features, int in_features, int seq_len,
               cudaStream_t stream = 0);

} // namespace gwen
