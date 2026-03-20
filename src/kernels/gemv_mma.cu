// Phase B: Marlin-style tensor core GEMV for Q4_K weights
//
// Key ideas:
// - mma.sync.m16n8k16.f32.f16.f16.f32 as the compute primitive
// - Just-in-time Q4_K→FP16 dequantization in registers
// - Each block processes TILE_M=64 output rows (4 warps × 16 rows/warp)
// - Weight data loaded with streaming hints (L2::evict_first)
// - Activation vector loaded with caching hints (stays in L2)
//
// Unlike the dp4a path, this kernel:
// - Takes FP16 activation input directly (no Q8_1 quantization needed)
// - Processes multiple output rows per block (amortizing activation loads)
// - Uses tensor cores for the multiply-accumulate

#include "gwen/kernels.h"
#include "gwen/ggml_quants.h"

namespace gwen {

// Pack two half values into a uint32_t register for mma fragments
__device__ __forceinline__
uint32_t pack_half2(half a, half b) {
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};\n" : "=r"(result) : "h"(*reinterpret_cast<unsigned short*>(&a)),
                                                     "h"(*reinterpret_cast<unsigned short*>(&b)));
    return result;
}

// ============================================================
// Constants
// ============================================================
static constexpr int QK_K = 256;   // elements per Q4_K super-block
static constexpr int TILE_M = 64;  // output rows per block (4 warps × 16)
static constexpr int MMA_M = 16;   // mma.sync M dimension
static constexpr int MMA_N = 8;    // mma.sync N dimension (N=1 for GEMV, padded to 8)
static constexpr int MMA_K = 16;   // mma.sync K dimension

// ============================================================
// Q4_K mma.sync GEMV kernel
// ============================================================
//
// Geometry:
//   Grid: ceil(out_features / TILE_M) blocks
//   Block: 128 threads (4 warps)
//   Each warp: processes 16 output rows via mma.sync M=16
//   Each block: 4 warps × 16 rows = 64 output rows
//
// Inner loop over K dimension:
//   For each Q4_K sub-block (32 elements each, 8 sub-blocks per super-block):
//     1. Load 16 bytes of quantized nibbles per row from global memory
//     2. Dequantize to FP16 in registers
//     3. Load activation FP16 values (shared across all rows)
//     4. Execute mma.sync.m16n8k16
//     5. Accumulate in FP32
//
// Since we're doing GEMV (N=1), we pad the activation to N=8 (only column 0 is non-zero).
// The mma.sync still computes 16×8 output, but we only use column 0.

__global__ void __launch_bounds__(128)
kernel_gemv_q4_k_mma(const block_q4_k* __restrict__ W,
                      const half* __restrict__ x,
                      half* __restrict__ y,
                      const half* __restrict__ residual,
                      int out_features, int in_features) {
    const int warp_id = threadIdx.x / 32;  // 0..3
    const int lane = threadIdx.x % 32;
    const int block_row_start = blockIdx.x * TILE_M;
    const int warp_row_start = block_row_start + warp_id * MMA_M;

    // Early exit if this warp is entirely out of bounds
    if (warp_row_start >= out_features) return;

    const int blocks_per_row = in_features / QK_K;

    // Shared memory for activation vector tile (16 FP16 values at a time)
    // All 4 warps share the same activation data
    __shared__ half x_smem[MMA_K];

    // FP32 accumulator for mma output — each thread owns 4 values
    // from the 16×8 output tile (but we only use column 0)
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // mma fragment registers
    uint32_t a_frag[4];  // A matrix: 16×16 (4 registers per thread)
    uint32_t b_frag[2];  // B matrix: 16×8  (2 registers per thread)

    // Iterate over K dimension: each Q4_K super-block has 256 elements = 8 sub-blocks of 32
    // Each mma processes 16 elements, so we need 2 mma ops per sub-block
    for (int blk_idx = 0; blk_idx < blocks_per_row; blk_idx++) {
        // Process 8 sub-blocks per Q4_K block, 2 mma passes per sub-block (32 elements = 2×16)
        for (int sb = 0; sb < 8; sb++) {
            // Each sub-block has 32 elements. We process them in two 16-element chunks.
            for (int half_sb = 0; half_sb < 2; half_sb++) {
                int k_offset = blk_idx * QK_K + sb * 32 + half_sb * 16;

                // --- Load activation tile into shared memory ---
                // 16 elements, loaded cooperatively by first 16 threads of each warp
                // (all warps need the same data so only warp 0 loads)
                if (warp_id == 0 && lane < 16) {
                    if (k_offset + lane < in_features)
                        x_smem[lane] = x[k_offset + lane];
                    else
                        x_smem[lane] = __float2half(0.0f);
                }
                __syncthreads();

                // --- Prepare B fragment (activation vector padded to 16×8) ---
                // For GEMV, B is 16×1 padded to 16×8. Column 0 = activation, rest = 0.
                // B fragment layout for mma.sync.m16n8k16:
                //   Thread t holds B[t%4*2 : t%4*2+1, t/4] (col-major perspective)
                //   But actually B is K×N where K=16, N=8.
                //   For .col layout: B[k, n] with k changing faster.
                //
                // We want B[:,0] = x_smem[:], B[:,1:7] = 0
                // After ldmatrix.trans, each thread gets the right fragment.
                //
                // Simpler: construct B fragments manually.
                // For mma.sync.m16n8k16.row.col, B matrix is K=16 × N=8 in col-major.
                // Thread t owns:
                //   b_frag[0] = {B[t%4*2, t/4], B[t%4*2+1, t/4]}  (half2)
                //   b_frag[1] = {B[t%4*2+8, t/4], B[t%4*2+1+8, t/4]}  (half2)
                // where B[k, n] = x[k] if n==0, else 0.
                {
                    int b_k0 = (lane % 4) * 2;
                    int b_n  = lane / 4;  // 0..7

                    half bv0, bv1, bv2, bv3;
                    if (b_n == 0) {
                        bv0 = x_smem[b_k0];
                        bv1 = x_smem[b_k0 + 1];
                        bv2 = x_smem[b_k0 + 8];
                        bv3 = x_smem[b_k0 + 1 + 8];
                    } else {
                        bv0 = __float2half(0.0f);
                        bv1 = bv0;
                        bv2 = bv0;
                        bv3 = bv0;
                    }
                    b_frag[0] = pack_half2(bv0, bv1);
                    b_frag[1] = pack_half2(bv2, bv3);
                }

                // --- Dequantize weights for this warp's 16 rows × 16 K elements ---
                // Correct mma.sync.m16n8k16 A-fragment register mapping:
                //   a_frag[0] = {A[t/4,   (t%4)*2],   A[t/4,   (t%4)*2+1]}    row 0-7,  K cols 0-7
                //   a_frag[1] = {A[t/4+8, (t%4)*2],   A[t/4+8, (t%4)*2+1]}    row 8-15, K cols 0-7
                //   a_frag[2] = {A[t/4,   (t%4)*2+8], A[t/4,   (t%4)*2+9]}    row 0-7,  K cols 8-15
                //   a_frag[3] = {A[t/4+8, (t%4)*2+8], A[t/4+8, (t%4)*2+9]}    row 8-15, K cols 8-15
                // Note: registers interleave row groups and column groups!
                {
                    int a_row_local = lane / 4;        // 0..7
                    int a_col_base  = (lane % 4) * 2;  // 0,2,4,6

                    for (int rg = 0; rg < 2; rg++) {
                        int global_row = warp_row_start + a_row_local + rg * 8;

                        half vals[2][2]; // [col_group 0..1][within_pair 0..1]

                        if (global_row < out_features) {
                            const block_q4_k& blk = W[global_row * blocks_per_row + blk_idx];
                            float d = __half2float(blk.d);
                            float dmin = __half2float(blk.dmin);

                            int sc_val, m_val;
                            if (sb < 4) {
                                sc_val = blk.scales[sb] & 0x3F;
                                m_val  = blk.scales[sb + 4] & 0x3F;
                            } else {
                                sc_val = (blk.scales[sb + 4] & 0xF) | ((blk.scales[sb - 4] >> 6) << 4);
                                m_val  = (blk.scales[sb + 4] >> 4) | ((blk.scales[sb] >> 6) << 4);
                            }

                            float scale = d * sc_val;
                            float offset = dmin * m_val;

                            for (int cg = 0; cg < 2; cg++) {
                                for (int ci = 0; ci < 2; ci++) {
                                    int k_local = a_col_base + cg * 8 + ci;
                                    int abs_elem = sb * 32 + half_sb * 16 + k_local;

                                    int qs_byte_idx = (abs_elem / 64) * 32 + (abs_elem % 32);
                                    bool is_high = (abs_elem % 64) >= 32;
                                    int q_val = is_high
                                        ? (blk.qs[qs_byte_idx] >> 4)
                                        : (blk.qs[qs_byte_idx] & 0xF);

                                    vals[cg][ci] = __float2half(scale * q_val - offset);
                                }
                            }
                        } else {
                            vals[0][0] = vals[0][1] = vals[1][0] = vals[1][1] = __float2half(0.0f);
                        }

                        // Register assignment: a_frag[rg + cg*2]
                        // rg=0,cg=0 → frag[0], rg=0,cg=1 → frag[2]
                        // rg=1,cg=0 → frag[1], rg=1,cg=1 → frag[3]
                        a_frag[rg]     = pack_half2(vals[0][0], vals[0][1]);  // cg=0
                        a_frag[rg + 2] = pack_half2(vals[1][0], vals[1][1]);  // cg=1
                    }
                }

                // --- Execute mma.sync ---
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n"
                    "    {%0, %1, %2, %3},\n"
                    "    {%4, %5, %6, %7},\n"
                    "    {%8, %9},\n"
                    "    {%10, %11, %12, %13};\n"
                    : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[0]), "r"(b_frag[1]),
                      "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3])
                );

                __syncthreads(); // For x_smem reuse
            }
        }
    }

    // --- Write results ---
    // Each thread owns 4 FP32 values from the 16×8 output tile.
    // Thread layout in mma output (16×8):
    //   acc[0] = D[lane/4, (lane%4)*2]
    //   acc[1] = D[lane/4, (lane%4)*2+1]
    //   acc[2] = D[lane/4+8, (lane%4)*2]
    //   acc[3] = D[lane/4+8, (lane%4)*2+1]
    // We only need column 0 (GEMV output), so only threads with (lane%4)*2 == 0
    // i.e., lane%4 == 0 (lanes 0, 4, 8, 12, 16, 20, 24, 28)

    if (lane % 4 == 0) {
        int row0 = warp_row_start + lane / 4;       // rows 0-7
        int row1 = warp_row_start + lane / 4 + 8;   // rows 8-15

        if (row0 < out_features) {
            float val = acc[0];
            if (residual)
                val += __half2float(residual[row0]);
            y[row0] = __float2half(val);
        }
        if (row1 < out_features) {
            float val = acc[2];
            if (residual)
                val += __half2float(residual[row1]);
            y[row1] = __float2half(val);
        }
    }
}

// ============================================================
// Q5_K mma.sync GEMV kernel
// ============================================================
__global__ void __launch_bounds__(128)
kernel_gemv_q5_k_mma(const block_q5_k* __restrict__ W,
                      const half* __restrict__ x,
                      half* __restrict__ y,
                      const half* __restrict__ residual,
                      int out_features, int in_features) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int block_row_start = blockIdx.x * TILE_M;
    const int warp_row_start = block_row_start + warp_id * MMA_M;

    if (warp_row_start >= out_features) return;

    const int blocks_per_row = in_features / QK_K;
    __shared__ half x_smem[MMA_K];

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t a_frag[4], b_frag[2];

    for (int blk_idx = 0; blk_idx < blocks_per_row; blk_idx++) {
        for (int sb = 0; sb < 8; sb++) {
            for (int half_sb = 0; half_sb < 2; half_sb++) {
                int k_offset = blk_idx * QK_K + sb * 32 + half_sb * 16;

                if (warp_id == 0 && lane < 16) {
                    x_smem[lane] = (k_offset + lane < in_features) ? x[k_offset + lane] : __float2half(0.0f);
                }
                __syncthreads();

                // B fragment (same as Q4_K)
                {
                    int b_k0 = (lane % 4) * 2;
                    int b_n  = lane / 4;
                    half bv0, bv1, bv2, bv3;
                    if (b_n == 0) {
                        bv0 = x_smem[b_k0]; bv1 = x_smem[b_k0+1];
                        bv2 = x_smem[b_k0+8]; bv3 = x_smem[b_k0+9];
                    } else {
                        bv0 = bv1 = bv2 = bv3 = __float2half(0.0f);
                    }
                    b_frag[0] = pack_half2(bv0, bv1);
                    b_frag[1] = pack_half2(bv2, bv3);
                }

                // A fragment: dequantize Q5_K
                {
                    int a_row_local = lane / 4;
                    int a_col_base  = (lane % 4) * 2;

                    for (int rg = 0; rg < 2; rg++) {
                        int global_row = warp_row_start + a_row_local + rg * 8;
                        half vals[2][2];

                        if (global_row < out_features) {
                            const block_q5_k& blk = W[global_row * blocks_per_row + blk_idx];
                            float d = __half2float(blk.d);
                            float dmin = __half2float(blk.dmin);

                            int sc_val, m_val;
                            if (sb < 4) {
                                sc_val = blk.scales[sb] & 0x3F;
                                m_val  = blk.scales[sb + 4] & 0x3F;
                            } else {
                                sc_val = (blk.scales[sb + 4] & 0xF) | ((blk.scales[sb - 4] >> 6) << 4);
                                m_val  = (blk.scales[sb + 4] >> 4) | ((blk.scales[sb] >> 6) << 4);
                            }
                            float scale = d * sc_val;
                            float offset = dmin * m_val;

                            for (int cg = 0; cg < 2; cg++) {
                                for (int ci = 0; ci < 2; ci++) {
                                    int k_local = a_col_base + cg * 8 + ci;
                                    int abs_elem = sb * 32 + half_sb * 16 + k_local;

                                    // Q5_K qs layout (same as Q4_K):
                                    // Sub-block pairs share 32 bytes. Even sb → low nibble, odd → high.
                                    int qs_byte_idx = (abs_elem / 64) * 32 + (abs_elem % 32);
                                    bool is_high_qs = (abs_elem % 64) >= 32;
                                    int q_lo = is_high_qs
                                        ? (blk.qs[qs_byte_idx] >> 4)
                                        : (blk.qs[qs_byte_idx] & 0xF);

                                    // High bit from qh (bit sb in byte abs_elem%32)
                                    int qh_byte_idx = abs_elem % 32;
                                    int qh_bit_pos = abs_elem / 32;
                                    int q_hi = (blk.qh[qh_byte_idx] >> qh_bit_pos) & 1;
                                    int q_val = q_lo | (q_hi << 4);

                                    vals[cg][ci] = __float2half(scale * q_val - offset);
                                }
                            }
                        } else {
                            vals[0][0] = vals[0][1] = vals[1][0] = vals[1][1] = __float2half(0.0f);
                        }
                        // Correct register assignment: rg + cg*2
                        a_frag[rg]     = pack_half2(vals[0][0], vals[0][1]);
                        a_frag[rg + 2] = pack_half2(vals[1][0], vals[1][1]);
                    }
                }

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n"
                    "    {%0, %1, %2, %3},\n"
                    "    {%4, %5, %6, %7},\n"
                    "    {%8, %9},\n"
                    "    {%10, %11, %12, %13};\n"
                    : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[0]), "r"(b_frag[1]),
                      "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3])
                );
                __syncthreads();
            }
        }
    }

    // Write results (same logic as Q4_K)
    if (lane % 4 == 0) {
        int row0 = warp_row_start + lane / 4;
        int row1 = warp_row_start + lane / 4 + 8;
        if (row0 < out_features) {
            float val = acc[0];
            if (residual) val += __half2float(residual[row0]);
            y[row0] = __float2half(val);
        }
        if (row1 < out_features) {
            float val = acc[2];
            if (residual) val += __half2float(residual[row1]);
            y[row1] = __float2half(val);
        }
    }
}

// ============================================================
// Q6_K mma.sync GEMV kernel
// ============================================================
__global__ void __launch_bounds__(128)
kernel_gemv_q6_k_mma(const block_q6_k* __restrict__ W,
                      const half* __restrict__ x,
                      half* __restrict__ y,
                      const half* __restrict__ residual,
                      int out_features, int in_features) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int block_row_start = blockIdx.x * TILE_M;
    const int warp_row_start = block_row_start + warp_id * MMA_M;

    if (warp_row_start >= out_features) return;

    const int blocks_per_row = in_features / QK_K;
    __shared__ half x_smem[MMA_K];

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t a_frag[4], b_frag[2];

    for (int blk_idx = 0; blk_idx < blocks_per_row; blk_idx++) {
        // Q6_K has 16 sub-groups of 16 elements each (256 total)
        // Process in chunks of 16 elements (one mma per chunk)
        for (int sg = 0; sg < 16; sg++) {
            int k_offset = blk_idx * QK_K + sg * 16;

            if (warp_id == 0 && lane < 16) {
                x_smem[lane] = (k_offset + lane < in_features) ? x[k_offset + lane] : __float2half(0.0f);
            }
            __syncthreads();

            // B fragment
            {
                int b_k0 = (lane % 4) * 2;
                int b_n  = lane / 4;
                half bv0, bv1, bv2, bv3;
                if (b_n == 0) {
                    bv0 = x_smem[b_k0]; bv1 = x_smem[b_k0+1];
                    bv2 = x_smem[b_k0+8]; bv3 = x_smem[b_k0+9];
                } else {
                    bv0 = bv1 = bv2 = bv3 = __float2half(0.0f);
                }
                b_frag[0] = pack_half2(bv0, bv1);
                b_frag[1] = pack_half2(bv2, bv3);
            }

            // A fragment: dequantize Q6_K
            {
                int a_row_local = lane / 4;
                int a_col_base  = (lane % 4) * 2;

                for (int rg = 0; rg < 2; rg++) {
                    int global_row = warp_row_start + a_row_local + rg * 8;
                    half vals[2][2];

                    if (global_row < out_features) {
                        const block_q6_k& blk = W[global_row * blocks_per_row + blk_idx];
                        float d = __half2float(blk.d);

                        for (int cg = 0; cg < 2; cg++) {
                            for (int ci = 0; ci < 2; ci++) {
                                int k_local = a_col_base + cg * 8 + ci;  // 0..15
                                int abs_elem = sg * 16 + k_local;

                                // Q6_K layout:
                                // ql[128]: lower 4 bits, packed 2 per byte
                                // qh[64]: upper 2 bits, packed 4 per byte
                                // scales[16]: int8 scales per sub-group of 16
                                int ql_idx = abs_elem / 2;

                                // ql layout: two halves (half_idx = abs_elem / 128)
                                // within each half: quarter = (abs_elem % 128) / 32
                                // ql_byte = half_idx * 64 + (quarter & 1) * 32 + (abs_elem % 32)
                                // But if quarter >= 2, shift right by 4

                                int half_idx = abs_elem / 128;
                                int j = abs_elem % 128;
                                int quarter = j / 32;
                                int pos = j % 32;

                                int ql_byte_idx = half_idx * 64 + (quarter & 1) * 32 + pos;
                                int ql_nibble;
                                if (quarter >= 2)
                                    ql_nibble = (blk.ql[ql_byte_idx] >> 4) & 0xF;
                                else
                                    ql_nibble = blk.ql[ql_byte_idx] & 0xF;

                                int qh_byte_idx = half_idx * 32 + pos;
                                int qh_shift = quarter * 2;
                                int qh_bits = (blk.qh[qh_byte_idx] >> qh_shift) & 0x3;

                                int q_val = ql_nibble | (qh_bits << 4);
                                int scale_idx = half_idx * 8 + quarter * 2 + pos / 16;
                                int8_t scale = blk.scales[scale_idx];

                                vals[cg][ci] = __float2half(d * scale * (q_val - 32));
                            }
                        }
                    } else {
                        vals[0][0] = vals[0][1] = vals[1][0] = vals[1][1] = __float2half(0.0f);
                    }
                    // Correct register assignment: rg + cg*2
                    a_frag[rg]     = pack_half2(vals[0][0], vals[0][1]);
                    a_frag[rg + 2] = pack_half2(vals[1][0], vals[1][1]);
                }
            }

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n"
                "    {%0, %1, %2, %3},\n"
                "    {%4, %5, %6, %7},\n"
                "    {%8, %9},\n"
                "    {%10, %11, %12, %13};\n"
                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                  "r"(b_frag[0]), "r"(b_frag[1]),
                  "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3])
            );
            __syncthreads();
        }
    }

    if (lane % 4 == 0) {
        int row0 = warp_row_start + lane / 4;
        int row1 = warp_row_start + lane / 4 + 8;
        if (row0 < out_features) {
            float val = acc[0];
            if (residual) val += __half2float(residual[row0]);
            y[row0] = __float2half(val);
        }
        if (row1 < out_features) {
            float val = acc[2];
            if (residual) val += __half2float(residual[row1]);
            y[row1] = __float2half(val);
        }
    }
}

// ============================================================
// Launch wrappers
// ============================================================

static void gemv_mma_internal(const void* W, const half* x, half* y, const half* residual,
                               int out_features, int in_features, GGMLType type, cudaStream_t stream) {
    int grid = (out_features + TILE_M - 1) / TILE_M;

    switch (type) {
        case GGMLType::Q4_K:
            kernel_gemv_q4_k_mma<<<grid, 128, 0, stream>>>(
                static_cast<const block_q4_k*>(W), x, y, residual, out_features, in_features);
            break;
        case GGMLType::Q5_K:
            kernel_gemv_q5_k_mma<<<grid, 128, 0, stream>>>(
                static_cast<const block_q5_k*>(W), x, y, residual, out_features, in_features);
            break;
        case GGMLType::Q6_K:
            kernel_gemv_q6_k_mma<<<grid, 128, 0, stream>>>(
                static_cast<const block_q6_k*>(W), x, y, residual, out_features, in_features);
            break;
        default:
            GWEN_CHECK(false, "Unsupported mma GEMV type (Q4_K/Q5_K/Q6_K only)");
    }
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_gemv_mma(const void* W, const half* x, half* y,
                    int out_features, int in_features, GGMLType type, cudaStream_t stream) {
    gemv_mma_internal(W, x, y, nullptr, out_features, in_features, type, stream);
}

void gwen_gemv_mma_residual(const void* W, const half* x, half* y, const half* residual,
                              int out_features, int in_features, GGMLType type, cudaStream_t stream) {
    gemv_mma_internal(W, x, y, residual, out_features, in_features, type, stream);
}

// ============================================================
// Marlin-style Q4_K GEMV — proper formulation
// ============================================================
//
// Key insight from Marlin: for GEMV, put batch in M (padded 1→16) and
// output features in N. This way every mma output column produces a
// different useful output neuron.
//
// C[M,N] = A[M,K] * B[K,N]
//   A = activation (M=1 padded to 16, K=in_features)
//   B = weights (K=in_features, N=out_features) — Q4_K, pre-reshuffled
//   C = output (M=1, N=out_features)
//
// Thread block: 256 threads (8 warps)
// Each warp handles 16 output columns (2 mma N-groups of 8)
// Each block produces N_TILE=128 output features
//
// Reshuffled weight layout per (16K × 128N) tile:
//   nibbles[1024]:   128 columns × 16 K-elements / 2 = 1024 bytes
//                    Thread tid loads 4 bytes at offset tid*4
//   scales[128]:     FP16 combined scale per column (d * sc[sb]) = 256 bytes
//   offsets[128]:    FP16 combined offset per column (dmin * mn[sb]) = 256 bytes
// Total: 1536 bytes per tile
//
// For each Q4_K block (256 K-elements), there are 16 such tiles.
// Metadata (scales/offsets) changes every 2 tiles (32 K-elements = 1 sub-block).
// So we store 8 scale/offset sets per Q4_K block: 8 × 512 = 4096 bytes metadata.
//
// Total per Q4_K-block column-tile (256K × 128N):
//   nibbles: 16 × 1024 = 16384 bytes
//   scales:  8 × 256 = 2048 bytes
//   offsets: 8 × 256 = 2048 bytes
// Total: 20480 bytes

static constexpr int N_TILE = 64;         // output columns per block
static constexpr int MARLIN_THREADS = 128; // threads per block (4 warps)

// Offsets within a Q4_K-block column-tile (256K × 128N)
// With N_TILE=64, MARLIN_THREADS=128:
// nibbles: 16 chunks × 128 threads × 4 bytes = 8192
// scales:  8 sub-blocks × 64 × sizeof(half) = 1024
// offsets: 8 sub-blocks × 64 × sizeof(half) = 1024
static constexpr int MARLIN_NIB_BYTES  = 8192;
static constexpr int MARLIN_SC_BYTES   = 1024;
static constexpr int MARLIN_OFF_BYTES  = 1024;
static constexpr int MARLIN_TILE_BYTES = MARLIN_NIB_BYTES + MARLIN_SC_BYTES + MARLIN_OFF_BYTES; // 10240

// lop3 — three-input bitwise logic in one instruction
template <int lut>
__device__ __forceinline__ int lop3(int a, int b, int c) {
    int res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

// Dequant 4 unsigned INT4 nibbles from a packed int32 → 2 × half2
// Returns values in range [0, 15] as FP16.
// Caller applies scale and offset.
__device__ __forceinline__ void dequant_u4(int q, half2& out0, half2& out1) {
    // Extract low nibbles (positions 0,2) and high nibbles (positions 1,3)
    // and embed into FP16 format using the "construct float" trick.
    // 0x6400 = FP16 for 1024.0. ORing a 4-bit value into the mantissa gives 1024+val.
    constexpr int LO = 0x000f000f;
    constexpr int HI = 0x00f000f0;
    constexpr int EX = 0x64006400;
    int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);  // (q & LO) | EX
    int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);

    // lo as half2 = {1024 + nib0, 1024 + nib2}. Subtract 1024 to get {nib0, nib2}.
    constexpr int BIAS = 0x64006400;  // {1024.0, 1024.0}
    out0 = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<const half2*>(&BIAS));

    // hi has nibbles shifted by 4 in the mantissa → value = 1024 + nib*16.
    // Multiply by 1/16 and subtract 64: (1024 + nib*16)/16 - 64 = 64 + nib - 64 = nib
    constexpr int MUL16 = 0x2c002c00;  // {1/16, 1/16} in FP16
    constexpr int SUB64 = 0xd400d400;  // {-64.0, -64.0} in FP16
    out1 = __hfma2(*reinterpret_cast<half2*>(&hi),
                   *reinterpret_cast<const half2*>(&MUL16),
                   *reinterpret_cast<const half2*>(&SUB64));
}

__global__ void __launch_bounds__(256)
kernel_gemv_q4k_marlin(const uint8_t* __restrict__ W_mma,
                        const half* __restrict__ x,
                        half* __restrict__ y,
                        const half* __restrict__ residual,
                        int out_features, int in_features) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = threadIdx.x;

    const int warp_n_start = blockIdx.x * N_TILE + warp_id * 16;

    extern __shared__ half x_smem[];
    for (int i = tid; i < in_features; i += MARLIN_THREADS)
        x_smem[i] = x[i];
    __syncthreads();

    float acc_g0[4] = {0, 0, 0, 0};
    float acc_g1[4] = {0, 0, 0, 0};

    uint32_t a_frag[4];
    a_frag[1] = 0;
    a_frag[3] = 0;

    const int a_k_base = (lane % 4) * 2;
    const bool a_active = (lane / 4 == 0);
    const int b_n_in_group = lane / 4;

    const int blocks_per_row = in_features / QK_K;

    int col_g0 = warp_id * 16 + b_n_in_group;
    int col_g1 = warp_id * 16 + 8 + b_n_in_group;

    for (int blk_idx = 0; blk_idx < blocks_per_row; blk_idx++) {
        const uint8_t* col_tile = W_mma +
            (size_t)(blockIdx.x * blocks_per_row + blk_idx) * MARLIN_TILE_BYTES;

        const uint8_t* nib_base = col_tile;
        const half* sc_base = reinterpret_cast<const half*>(col_tile + MARLIN_NIB_BYTES);
        const half* off_base = reinterpret_cast<const half*>(col_tile + MARLIN_NIB_BYTES + MARLIN_SC_BYTES);

        for (int sb = 0; sb < 8; sb++) {
            const half* sb_sc = sc_base + sb * N_TILE;
            const half* sb_off = off_base + sb * N_TILE;

            half2 scale_g0 = __half2half2(sb_sc[col_g0]);
            half2 scale_g1 = __half2half2(sb_sc[col_g1]);
            half2 neg_off_g0 = __half2half2(__hneg(sb_off[col_g0]));
            half2 neg_off_g1 = __half2half2(__hneg(sb_off[col_g1]));

            // Prefetch nibbles for both halves of this sub-block
            const uint8_t* chunk0 = nib_base + (sb * 2) * (MARLIN_THREADS * 4);
            const uint8_t* chunk1 = nib_base + (sb * 2 + 1) * (MARLIN_THREADS * 4);
            uint32_t packed0, packed1;
            asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(packed0) : "l"(chunk0 + tid * 4));
            asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(packed1) : "l"(chunk1 + tid * 4));

            // Process half 0
            {
                int k_offset = blk_idx * QK_K + sb * 32;
                if (a_active) {
                    a_frag[0] = pack_half2(x_smem[k_offset + a_k_base],
                                           x_smem[k_offset + a_k_base + 1]);
                    a_frag[2] = pack_half2(x_smem[k_offset + a_k_base + 8],
                                           x_smem[k_offset + a_k_base + 9]);
                } else { a_frag[0] = 0; a_frag[2] = 0; }

                int q_g0 = (packed0 & 0xFF) | (packed0 & 0xFF0000);
                int q_g1 = ((packed0 >> 8) & 0xFF) | ((packed0 >> 8) & 0xFF0000);
                half2 raw0_g0, raw1_g0, raw0_g1, raw1_g1;
                dequant_u4(q_g0, raw0_g0, raw1_g0);
                dequant_u4(q_g1, raw0_g1, raw1_g1);
                uint32_t bf_g0[2], bf_g1[2];
                half2 b0g0 = __hfma2(scale_g0, raw0_g0, neg_off_g0);
                half2 b1g0 = __hfma2(scale_g0, raw1_g0, neg_off_g0);
                half2 b0g1 = __hfma2(scale_g1, raw0_g1, neg_off_g1);
                half2 b1g1 = __hfma2(scale_g1, raw1_g1, neg_off_g1);
                bf_g0[0] = *reinterpret_cast<uint32_t*>(&b0g0);
                bf_g0[1] = *reinterpret_cast<uint32_t*>(&b1g0);
                bf_g1[0] = *reinterpret_cast<uint32_t*>(&b0g1);
                bf_g1[1] = *reinterpret_cast<uint32_t*>(&b1g1);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    :"=f"(acc_g0[0]),"=f"(acc_g0[1]),"=f"(acc_g0[2]),"=f"(acc_g0[3])
                    :"r"(a_frag[0]),"r"(a_frag[1]),"r"(a_frag[2]),"r"(a_frag[3]),
                     "r"(bf_g0[0]),"r"(bf_g0[1]),
                     "f"(acc_g0[0]),"f"(acc_g0[1]),"f"(acc_g0[2]),"f"(acc_g0[3]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    :"=f"(acc_g1[0]),"=f"(acc_g1[1]),"=f"(acc_g1[2]),"=f"(acc_g1[3])
                    :"r"(a_frag[0]),"r"(a_frag[1]),"r"(a_frag[2]),"r"(a_frag[3]),
                     "r"(bf_g1[0]),"r"(bf_g1[1]),
                     "f"(acc_g1[0]),"f"(acc_g1[1]),"f"(acc_g1[2]),"f"(acc_g1[3]));
            }

            // Process half 1
            {
                int k_offset = blk_idx * QK_K + sb * 32 + 16;
                if (a_active) {
                    a_frag[0] = pack_half2(x_smem[k_offset + a_k_base],
                                           x_smem[k_offset + a_k_base + 1]);
                    a_frag[2] = pack_half2(x_smem[k_offset + a_k_base + 8],
                                           x_smem[k_offset + a_k_base + 9]);
                } else { a_frag[0] = 0; a_frag[2] = 0; }

                int q_g0 = (packed1 & 0xFF) | (packed1 & 0xFF0000);
                int q_g1 = ((packed1 >> 8) & 0xFF) | ((packed1 >> 8) & 0xFF0000);
                half2 raw0_g0, raw1_g0, raw0_g1, raw1_g1;
                dequant_u4(q_g0, raw0_g0, raw1_g0);
                dequant_u4(q_g1, raw0_g1, raw1_g1);
                uint32_t bf_g0[2], bf_g1[2];
                half2 b0g0 = __hfma2(scale_g0, raw0_g0, neg_off_g0);
                half2 b1g0 = __hfma2(scale_g0, raw1_g0, neg_off_g0);
                half2 b0g1 = __hfma2(scale_g1, raw0_g1, neg_off_g1);
                half2 b1g1 = __hfma2(scale_g1, raw1_g1, neg_off_g1);
                bf_g0[0] = *reinterpret_cast<uint32_t*>(&b0g0);
                bf_g0[1] = *reinterpret_cast<uint32_t*>(&b1g0);
                bf_g1[0] = *reinterpret_cast<uint32_t*>(&b0g1);
                bf_g1[1] = *reinterpret_cast<uint32_t*>(&b1g1);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    :"=f"(acc_g0[0]),"=f"(acc_g0[1]),"=f"(acc_g0[2]),"=f"(acc_g0[3])
                    :"r"(a_frag[0]),"r"(a_frag[1]),"r"(a_frag[2]),"r"(a_frag[3]),
                     "r"(bf_g0[0]),"r"(bf_g0[1]),
                     "f"(acc_g0[0]),"f"(acc_g0[1]),"f"(acc_g0[2]),"f"(acc_g0[3]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    :"=f"(acc_g1[0]),"=f"(acc_g1[1]),"=f"(acc_g1[2]),"=f"(acc_g1[3])
                    :"r"(a_frag[0]),"r"(a_frag[1]),"r"(a_frag[2]),"r"(a_frag[3]),
                     "r"(bf_g1[0]),"r"(bf_g1[1]),
                     "f"(acc_g1[0]),"f"(acc_g1[1]),"f"(acc_g1[2]),"f"(acc_g1[3]));
            }
        }
    }

    if (lane / 4 == 0) {
        int col_base_g0 = warp_n_start + (lane % 4) * 2;
        int col_base_g1 = warp_n_start + 8 + (lane % 4) * 2;

        if (col_base_g0 < out_features) {
            float v0 = acc_g0[0], v1 = acc_g0[1];
            if (residual) { v0 += __half2float(residual[col_base_g0]); v1 += __half2float(residual[col_base_g0+1]); }
            y[col_base_g0] = __float2half(v0);
            if (col_base_g0 + 1 < out_features) y[col_base_g0 + 1] = __float2half(v1);
        }
        if (col_base_g1 < out_features) {
            float v0 = acc_g1[0], v1 = acc_g1[1];
            if (residual) { v0 += __half2float(residual[col_base_g1]); v1 += __half2float(residual[col_base_g1+1]); }
            y[col_base_g1] = __float2half(v0);
            if (col_base_g1 + 1 < out_features) y[col_base_g1 + 1] = __float2half(v1);
        }
    }
}

// Launch wrapper
void gwen_gemv_mma_reshuffled(const void* W_mma, const half* x, half* y,
                               int out_features, int in_features, cudaStream_t stream) {
    int grid = (out_features + N_TILE - 1) / N_TILE;
    // Shared memory: activation vector + 2 pipeline stages of nibbles
    size_t smem = in_features * sizeof(half);
    kernel_gemv_q4k_marlin<<<grid, MARLIN_THREADS, smem, stream>>>(
        static_cast<const uint8_t*>(W_mma), x, y, nullptr, out_features, in_features);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

void gwen_gemv_mma_reshuffled_residual(const void* W_mma, const half* x, half* y, const half* residual,
                                        int out_features, int in_features, cudaStream_t stream) {
    int grid = (out_features + N_TILE - 1) / N_TILE;
    size_t smem = in_features * sizeof(half);
    kernel_gemv_q4k_marlin<<<grid, MARLIN_THREADS, smem, stream>>>(
        static_cast<const uint8_t*>(W_mma), x, y, residual, out_features, in_features);
    GWEN_CHECK_CUDA(cudaGetLastError());
}

} // namespace gwen
