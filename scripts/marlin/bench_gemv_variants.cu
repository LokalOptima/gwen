// Micro-benchmark: dp4a vs Marlin-style mma GEMV.
// Reports raw kernel times. BW% = Q4_K_weight_bytes / time / 896 GB/s.

#include <cstdio>
#include <cstdlib>
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

static const double PEAK_BW = 896e9;

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
static std::vector<uint8_t> reshuffle(const void* data, int out, int in) {
    const auto* blks = static_cast<const block_q4_k*>(data);
    const int NT = 64, NTH = 128;
    int bpr = in/256, nct = (out+NT-1)/NT;
    const int NIB=16*NTH*4, SC=8*NT*2, OFF=8*NT*2, TILE=NIB+SC+OFF;
    std::vector<uint8_t> r(nct*bpr*TILE, 0);
    for (int ct = 0; ct < nct; ct++) for (int blk = 0; blk < bpr; blk++) {
        uint8_t* tile = r.data() + (ct*bpr+blk)*TILE;
        half* sc_b = reinterpret_cast<half*>(tile+NIB);
        half* of_b = reinterpret_cast<half*>(tile+NIB+SC);
        for (int cl = 0; cl < NT; cl++) {
            int cg = ct*NT+cl; if (cg >= out) continue;
            const auto& s = blks[cg*bpr+blk]; uint8_t sc[8], mn[8];
            unpack_sc(s.scales, sc, mn);
            float d=__half2float(s.d), dm=__half2float(s.dmin);
            for (int sb = 0; sb < 8; sb++) {
                sc_b[sb*NT+cl]=__float2half(d*sc[sb]);
                of_b[sb*NT+cl]=__float2half(dm*mn[sb]);
            }
        }
        for (int ch = 0; ch < 16; ch++) {
            int sb=ch/2, hsb=ch%2;
            for (int tid = 0; tid < NTH; tid++) {
                int w=tid/32, t=tid%32, bk=(t%4)*2, bn=t/4;
                int c0=ct*NT+w*16+bn, c1=ct*NT+w*16+8+bn;
                int ak0=sb*32+hsb*16+bk, ak1=ak0+1, ak8=ak0+8, ak9=ak0+9;
                uint8_t* dst = tile + ch*(NTH*4) + tid*4;
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
    printf("=== GEMV Micro-Benchmark: dp4a vs Marlin ===\n");
    printf("GPU: RTX 5070 Ti, 896 GB/s peak\n\n");

    struct { const char* name; int out, in; } cfgs[] = {
        // 0.8B
        {"0.8B gate  (2048x1024)",   2048,  1024},
        {"0.8B ffn_g (3584x1024)",   3584,  1024},
        {"0.8B ffn_d (1024x3584)",   1024,  3584},
        // 4B
        {"4B gate    (5120x2560)",   5120,  2560},
        {"4B ffn_g   (8960x2560)",   8960,  2560},
        {"4B ffn_d   (2560x8960)",   2560,  8960},
        {"4B attn_o  (2560x5120)",   2560,  5120},
        {"4B lm_head (151936x2560)", 151936, 2560},
    };

    const int ITERS = 200;

    printf("%-28s | %8s %6s | %8s %6s | %7s\n",
           "Matrix", "dp4a_us", "BW%", "marlin", "BW%", "ratio");
    printf("--------------------------------------------------------------------------\n");

    for (auto& c : cfgs) {
        int bpr = c.in / 256;
        int total_blocks = c.out * bpr;
        size_t w_bytes = (size_t)total_blocks * 144;  // Q4_K = 144 bytes/block

        // Allocate and fill Q4_K weights
        std::vector<uint8_t> h_W(w_bytes);
        srand(42);
        for (auto& b : h_W) b = rand() % 256;

        void* d_W;
        CHECK_CUDA(cudaMalloc(&d_W, w_bytes));
        CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), w_bytes, cudaMemcpyHostToDevice));

        // Reshuffle for Marlin
        auto resh = reshuffle(h_W.data(), c.out, c.in);
        void* d_Wm;
        CHECK_CUDA(cudaMalloc(&d_Wm, resh.size()));
        CHECK_CUDA(cudaMemcpy(d_Wm, resh.data(), resh.size(), cudaMemcpyHostToDevice));

        // Input vectors
        half* d_x;
        CHECK_CUDA(cudaMalloc(&d_x, c.in * sizeof(half)));
        CHECK_CUDA(cudaMemset(d_x, 0, c.in * sizeof(half)));
        void* d_q8;
        CHECK_CUDA(cudaMalloc(&d_q8, (c.in / 32) * sizeof(block_q8_1)));
        gwen_quantize_q8_1(d_x, d_q8, c.in);
        half* d_y;
        CHECK_CUDA(cudaMalloc(&d_y, c.out * sizeof(half)));

        float dp4a_us = bench_min_us([&]{
            gwen_gemv_dp4a(d_W, d_q8, d_y, c.out, c.in, GGMLType::Q4_K);
        }, ITERS);

        float marlin_us = bench_min_us([&]{
            gwen_gemv_mma_reshuffled(d_Wm, d_x, d_y, c.out, c.in);
        }, ITERS);

        double dp4a_bw = (double)w_bytes / (dp4a_us * 1e-6) / PEAK_BW * 100.0;
        double marlin_bw = (double)w_bytes / (marlin_us * 1e-6) / PEAK_BW * 100.0;
        float ratio = marlin_us / dp4a_us;

        printf("%-28s | %7.1f %5.1f%% | %7.1f %5.1f%% | %6.2fx\n",
               c.name, dp4a_us, dp4a_bw, marlin_us, marlin_bw, ratio);

        cudaFree(d_W); cudaFree(d_Wm); cudaFree(d_x); cudaFree(d_q8); cudaFree(d_y);
    }

    printf("\nratio = marlin/dp4a (< 1.0 means marlin wins)\n");
    return 0;
}
