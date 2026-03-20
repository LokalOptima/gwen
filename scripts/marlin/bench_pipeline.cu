// Test cp.async pipeline + K-splitting for Marlin-style GEMV.
// Standalone — includes its own kernel, no dependency on gemv_mma.cu.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "gwen/kernels.h"
#include "gwen/ggml_quants.h"
#include "gwen/common.h"

using namespace gwen;

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Reshuffling (same as model.cu / bench)
// ============================================================
static void unpack_sc(const uint8_t sp[12], uint8_t sc[8], uint8_t mn[8]) {
    for (int sb = 0; sb < 4; sb++) { sc[sb]=sp[sb]&0x3F; mn[sb]=sp[sb+4]&0x3F; }
    for (int sb = 4; sb < 8; sb++) {
        sc[sb]=(sp[sb+4]&0xF)|((sp[sb-4]>>6)<<4);
        mn[sb]=(sp[sb+4]>>4)|((sp[sb]>>6)<<4);
    }
}
static uint8_t q4k_nib(const uint8_t qs[128], int e) {
    return ((e%64)>=32) ? (qs[(e/64)*32+(e%32)]>>4) : (qs[(e/64)*32+(e%32)]&0xF);
}

static constexpr int R_NT = 64, R_NTH = 128;

static std::vector<uint8_t> reshuffle(const void* data, int out, int in) {
    const auto* blks = static_cast<const block_q4_k*>(data);
    int bpr = in/256, nct = (out+R_NT-1)/R_NT;
    const int NIB=16*R_NTH*4, SC=8*R_NT*2, OFF=8*R_NT*2, TILE=NIB+SC+OFF;
    std::vector<uint8_t> r(nct*bpr*TILE, 0);
    for (int ct = 0; ct < nct; ct++) for (int blk = 0; blk < bpr; blk++) {
        uint8_t* tile = r.data() + (ct*bpr+blk)*TILE;
        half* sc_b = reinterpret_cast<half*>(tile+NIB);
        half* of_b = reinterpret_cast<half*>(tile+NIB+SC);
        for (int cl = 0; cl < R_NT; cl++) {
            int cg = ct*R_NT+cl; if (cg >= out) continue;
            const auto& s = blks[cg*bpr+blk]; uint8_t sc[8], mn[8];
            unpack_sc(s.scales, sc, mn);
            float d=__half2float(s.d), dm=__half2float(s.dmin);
            for (int sb = 0; sb < 8; sb++) {
                sc_b[sb*R_NT+cl]=__float2half(d*sc[sb]);
                of_b[sb*R_NT+cl]=__float2half(dm*mn[sb]);
            }
        }
        for (int ch = 0; ch < 16; ch++) {
            int sb=ch/2, hsb=ch%2;
            for (int tid = 0; tid < R_NTH; tid++) {
                int w=tid/32, t=tid%32, bk=(t%4)*2, bn=t/4;
                int c0=ct*R_NT+w*16+bn, c1=ct*R_NT+w*16+8+bn;
                int ak0=sb*32+hsb*16+bk, ak1=ak0+1, ak8=ak0+8, ak9=ak0+9;
                uint8_t* dst = tile + ch*(R_NTH*4) + tid*4;
                auto gn=[&](int cg, int ak)->uint8_t{ return cg>=out?0:q4k_nib(blks[cg*bpr+blk].qs,ak); };
                dst[0]=gn(c0,ak0)|(gn(c0,ak8)<<4);
                dst[1]=gn(c1,ak0)|(gn(c1,ak8)<<4);
                dst[2]=gn(c0,ak1)|(gn(c0,ak9)<<4);
                dst[3]=gn(c1,ak1)|(gn(c1,ak9)<<4);
            }
        }
    }
    return r;
}

// ============================================================
// Kernel helpers
// ============================================================
static constexpr int NT = 64;
static constexpr int NTHREADS = 128;
static constexpr int QK_K_L = 256;
static constexpr int MARLIN_NIB = 16 * NTHREADS * 4;  // 8192
static constexpr int MARLIN_SC  = 8 * NT * 2;          // 1024
static constexpr int MARLIN_OFF = 8 * NT * 2;          // 1024
static constexpr int MARLIN_TILE = MARLIN_NIB + MARLIN_SC + MARLIN_OFF;
static constexpr int CHUNK_BYTES = NTHREADS * 4;       // 512 bytes per chunk

// Chunks per pipeline stage. 8 chunks = 128 K-elements = 1 Q4_K block / 2
static constexpr int CHUNKS_PER_STAGE = 8;
static constexpr int STAGE_BYTES = CHUNKS_PER_STAGE * CHUNK_BYTES; // 4096

template <int lut>
__device__ __forceinline__ int lop3(int a, int b, int c) {
    int res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

__device__ __forceinline__ void dequant_u4(int q, half2& out0, half2& out1) {
    constexpr int LO = 0x000f000f, HI = 0x00f000f0, EX = 0x64006400;
    int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
    int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
    constexpr int BIAS = 0x64006400;
    out0 = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<const half2*>(&BIAS));
    constexpr int MUL16 = 0x2c002c00, SUB64 = 0xd400d400;
    out1 = __hfma2(*reinterpret_cast<half2*>(&hi),
                   *reinterpret_cast<const half2*>(&MUL16),
                   *reinterpret_cast<const half2*>(&SUB64));
}

__device__ __forceinline__ uint32_t pack_h2(half a, half b) {
    uint32_t r;
    asm("mov.b32 %0, {%1, %2};\n" : "=r"(r)
        : "h"(*reinterpret_cast<unsigned short*>(&a)),
          "h"(*reinterpret_cast<unsigned short*>(&b)));
    return r;
}

__device__ __forceinline__ void cp_async_4B(void* smem, const void* glob) {
    uint32_t s = __cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(glob));
}

__device__ __forceinline__ void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ============================================================
// Pipelined Marlin kernel with K-splitting
// ============================================================
// Grid: (n_col_tiles, k_splits)
// blockIdx.x = column tile, blockIdx.y = K-split index
// Each block processes NT output columns × (K / k_splits) K-elements.
// Partial sums atomicAdd'd to output.

__global__ void __launch_bounds__(128)
kernel_marlin_pipeline(const uint8_t* __restrict__ W_mma,
                        const half* __restrict__ x,
                        float* __restrict__ y_f32,  // FP32 output for atomic accumulation
                        int out_features, int in_features,
                        int k_splits) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = threadIdx.x;

    const int col_tile = blockIdx.x;
    const int k_split_id = blockIdx.y;
    const int warp_n_start = col_tile * NT + warp_id * 16;

    // Shared memory: activation vector + 2 pipeline stages of weight nibbles
    extern __shared__ char smem_raw[];
    half* x_smem = reinterpret_cast<half*>(smem_raw);
    int x_bytes_aligned = ((in_features * (int)sizeof(half)) + 15) & ~15;
    uint8_t* w_smem = reinterpret_cast<uint8_t*>(smem_raw + x_bytes_aligned);
    // w_smem[0 .. STAGE_BYTES-1] = stage 0
    // w_smem[STAGE_BYTES .. 2*STAGE_BYTES-1] = stage 1

    // Load full activation vector
    for (int i = tid; i < in_features; i += NTHREADS)
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
    int col_g0 = warp_id * 16 + b_n_in_group;
    int col_g1 = warp_id * 16 + 8 + b_n_in_group;

    const int blocks_per_row = in_features / QK_K_L;
    const int total_chunks = blocks_per_row * 16;

    // K-splitting: this block handles chunks [k_start, k_end)
    int chunks_per_split = (total_chunks + k_splits - 1) / k_splits;
    // Align to CHUNKS_PER_STAGE boundary
    chunks_per_split = ((chunks_per_split + CHUNKS_PER_STAGE - 1) / CHUNKS_PER_STAGE) * CHUNKS_PER_STAGE;
    int k_start = k_split_id * chunks_per_split;
    int k_end = min(k_start + chunks_per_split, total_chunks);
    if (k_start >= total_chunks) return;

    int total_stages = (k_end - k_start + CHUNKS_PER_STAGE - 1) / CHUNKS_PER_STAGE;

    // Helper: load one stage of weight nibbles to shared memory via cp.async
    auto load_stage = [&](int stage_buf, int stage_chunk_start) {
        for (int lc = 0; lc < CHUNKS_PER_STAGE; lc++) {
            int ci = stage_chunk_start + lc;
            if (ci >= k_end) break;
            int blk_idx = ci / 16;
            int chunk_in_blk = ci % 16;
            const uint8_t* col_tile_base = W_mma +
                (size_t)(col_tile * blocks_per_row + blk_idx) * MARLIN_TILE;
            const uint8_t* src = col_tile_base + chunk_in_blk * CHUNK_BYTES + tid * 4;
            uint8_t* dst = w_smem + stage_buf * STAGE_BYTES + lc * CHUNK_BYTES + tid * 4;
            cp_async_4B(dst, src);
        }
    };

    // Prefill stage 0
    load_stage(0, k_start);
    cp_async_fence();

    for (int si = 0; si < total_stages; si++) {
        int stage_chunk_start = k_start + si * CHUNKS_PER_STAGE;
        int cur_buf = si % 2;

        // Prefetch next stage
        if (si + 1 < total_stages) {
            load_stage(1 - cur_buf, stage_chunk_start + CHUNKS_PER_STAGE);
            cp_async_fence();
        }

        // Wait for current stage
        cp_async_wait<1>();
        __syncthreads();

        // Process CHUNKS_PER_STAGE chunks from shared memory
        for (int lc = 0; lc < CHUNKS_PER_STAGE; lc++) {
            int ci = stage_chunk_start + lc;
            if (ci >= k_end) break;

            int blk_idx = ci / 16;
            int chunk_in_blk = ci % 16;
            int sb = chunk_in_blk / 2;
            int half_sb = chunk_in_blk % 2;
            int k_offset = blk_idx * QK_K_L + sb * 32 + half_sb * 16;

            // Load scales from global (small, cached in L1)
            // Only reload when sub-block changes (every 2 chunks)
            half2 scale_g0, scale_g1, neg_off_g0, neg_off_g1;
            {
                const uint8_t* col_tile_base = W_mma +
                    (size_t)(col_tile * blocks_per_row + blk_idx) * MARLIN_TILE;
                const half* sb_sc = reinterpret_cast<const half*>(col_tile_base + MARLIN_NIB) + sb * NT;
                const half* sb_off = reinterpret_cast<const half*>(col_tile_base + MARLIN_NIB + MARLIN_SC) + sb * NT;
                scale_g0 = __half2half2(sb_sc[col_g0]);
                scale_g1 = __half2half2(sb_sc[col_g1]);
                neg_off_g0 = __half2half2(__hneg(sb_off[col_g0]));
                neg_off_g1 = __half2half2(__hneg(sb_off[col_g1]));
            }

            // A fragment
            if (a_active) {
                a_frag[0] = pack_h2(x_smem[k_offset + a_k_base],
                                    x_smem[k_offset + a_k_base + 1]);
                a_frag[2] = pack_h2(x_smem[k_offset + a_k_base + 8],
                                    x_smem[k_offset + a_k_base + 9]);
            } else {
                a_frag[0] = 0;
                a_frag[2] = 0;
            }

            // B fragment from shared memory
            uint32_t packed = *reinterpret_cast<uint32_t*>(
                &w_smem[cur_buf * STAGE_BYTES + lc * CHUNK_BYTES + tid * 4]);

            int q_g0 = (packed & 0xFF) | (packed & 0xFF0000);
            int q_g1 = ((packed >> 8) & 0xFF) | ((packed >> 8) & 0xFF0000);

            half2 r0g0, r1g0, r0g1, r1g1;
            dequant_u4(q_g0, r0g0, r1g0);
            dequant_u4(q_g1, r0g1, r1g1);

            uint32_t bf_g0[2], bf_g1[2];
            half2 b0g0 = __hfma2(scale_g0, r0g0, neg_off_g0);
            half2 b1g0 = __hfma2(scale_g0, r1g0, neg_off_g0);
            half2 b0g1 = __hfma2(scale_g1, r0g1, neg_off_g1);
            half2 b1g1 = __hfma2(scale_g1, r1g1, neg_off_g1);
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

        __syncthreads();  // Ensure smem reads done before next stage writes
    }

    // Write output: atomicAdd partial sums (FP32)
    if (lane / 4 == 0) {
        int c0 = warp_n_start + (lane % 4) * 2;
        int c1 = warp_n_start + 8 + (lane % 4) * 2;
        if (c0 < out_features) {
            atomicAdd(&y_f32[c0], acc_g0[0]);
            if (c0 + 1 < out_features) atomicAdd(&y_f32[c0 + 1], acc_g0[1]);
        }
        if (c1 < out_features) {
            atomicAdd(&y_f32[c1], acc_g1[0]);
            if (c1 + 1 < out_features) atomicAdd(&y_f32[c1 + 1], acc_g1[1]);
        }
    }
}

// ============================================================
// Benchmark harness
// ============================================================

static float bench_min_us(auto fn, int iters) {
    for (int i = 0; i < 10; i++) fn();
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    float best = 1e9f;
    for (int i = 0; i < iters; i++) {
        CHECK_CUDA(cudaEventRecord(t0));
        fn();
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float ms; CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return best * 1000.0f;
}

int main() {
    printf("=== Pipeline + K-split benchmark ===\n\n");

    struct { const char* name; int out, in; } cfgs[] = {
        {"0.8B gate  (2048x1024)",   2048,  1024},
        {"0.8B ffn_d (1024x3584)",   1024,  3584},
        {"4B gate    (5120x2560)",   5120,  2560},
        {"4B ffn_g   (8960x2560)",   8960,  2560},
        {"4B ffn_d   (2560x8960)",   2560,  8960},
        {"4B attn_o  (2560x5120)",   2560,  5120},
        {"4B lm_head (151936x2560)", 151936, 2560},
    };

    const int ITERS = 200;
    int k_split_options[] = {1, 2, 4, 8};

    printf("%-28s | %8s | ", "Matrix", "dp4a");
    for (int ks : k_split_options) printf("ks=%-3d | ", ks);
    printf("\n");
    printf("----------------------------+----------+");
    for (int i = 0; i < 4; i++) printf("--------+");
    printf("\n");

    for (auto& c : cfgs) {
        int bpr = c.in / 256;
        int total_blocks = c.out * bpr;
        size_t w_bytes = (size_t)total_blocks * 144;

        std::vector<uint8_t> h_W(w_bytes);
        srand(42);
        for (auto& b : h_W) b = rand() % 256;

        void* d_W;
        CHECK_CUDA(cudaMalloc(&d_W, w_bytes));
        CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), w_bytes, cudaMemcpyHostToDevice));

        auto resh = reshuffle(h_W.data(), c.out, c.in);
        void* d_Wm;
        CHECK_CUDA(cudaMalloc(&d_Wm, resh.size()));
        CHECK_CUDA(cudaMemcpy(d_Wm, resh.data(), resh.size(), cudaMemcpyHostToDevice));

        half* d_x;
        CHECK_CUDA(cudaMalloc(&d_x, c.in * sizeof(half)));
        CHECK_CUDA(cudaMemset(d_x, 0, c.in * sizeof(half)));
        void* d_q8;
        CHECK_CUDA(cudaMalloc(&d_q8, (c.in / 32) * sizeof(block_q8_1)));
        gwen_quantize_q8_1(d_x, d_q8, c.in);
        half* d_y;
        CHECK_CUDA(cudaMalloc(&d_y, c.out * sizeof(half)));
        float* d_y_f32;
        CHECK_CUDA(cudaMalloc(&d_y_f32, c.out * sizeof(float)));

        float dp4a_us = bench_min_us([&]{
            gwen_gemv_dp4a(d_W, d_q8, d_y, c.out, c.in, GGMLType::Q4_K);
        }, ITERS);

        printf("%-28s | %6.1f us | ", c.name, dp4a_us);

        for (int ks : k_split_options) {
            int n_col_tiles = (c.out + NT - 1) / NT;
            dim3 grid(n_col_tiles, ks);
            int x_bytes_aligned = ((c.in * (int)sizeof(half)) + 15) & ~15;
            size_t smem = x_bytes_aligned + 2 * STAGE_BYTES;

            float us = bench_min_us([&]{
                CHECK_CUDA(cudaMemset(d_y_f32, 0, c.out * sizeof(float)));
                kernel_marlin_pipeline<<<grid, NTHREADS, smem>>>(
                    static_cast<const uint8_t*>(d_Wm), d_x, d_y_f32,
                    c.out, c.in, ks);
            }, ITERS);

            float ratio = us / dp4a_us;
            printf("%5.1f %4.2f | ", us, ratio);
        }
        printf("\n");

        cudaFree(d_W); cudaFree(d_Wm); cudaFree(d_x); cudaFree(d_q8);
        cudaFree(d_y); cudaFree(d_y_f32);
    }

    printf("\nValues are: time_us ratio_vs_dp4a\n");
    return 0;
}
