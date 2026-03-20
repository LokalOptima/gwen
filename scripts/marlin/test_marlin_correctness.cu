// Verify Marlin-style kernel produces correct results by comparing against dp4a.
// Tests both 0.8B and 4B model matrix sizes.

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

// Marlin reshuffling (must match model.cu exactly)
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
static std::vector<uint8_t> reshuffle(const block_q4_k* blks, int out, int in) {
    const int NT = 64, NTHREADS = 128;
    int bpr = in/256, nct = (out+NT-1)/NT;
    const int NIB=16*NTHREADS*4, SC=8*NT*2, OFF=8*NT*2, TILE=NIB+SC+OFF;
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
            for (int tid = 0; tid < NTHREADS; tid++) {
                int w=tid/32, t=tid%32, bk=(t%4)*2, bn=t/4;
                int c0=ct*NT+w*16+bn, c1=ct*NT+w*16+8+bn;
                int ak0=sb*32+hsb*16+bk, ak1=ak0+1, ak8=ak0+8, ak9=ak0+9;
                uint8_t* dst = tile + ch*(NTHREADS*4) + tid*4;
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

static bool test(const char* name, int out, int in) {
    printf("  %-25s %5d x %-5d ... ", name, out, in);

    int bpr = in / 256;
    int total_blocks = out * bpr;
    int q8_blocks = in / 32;

    // Random Q4_K weights
    std::vector<block_q4_k> h_W(total_blocks);
    srand(42);
    for (auto& b : h_W) {
        b.d = __float2half((float)(rand()%100+1)/100.0f);
        b.dmin = __float2half((float)(rand()%50)/100.0f);
        for (int j = 0; j < 12; j++) b.scales[j] = rand()%256;
        for (int j = 0; j < 128; j++) b.qs[j] = rand()%256;
    }

    // Random FP16 input
    std::vector<half> h_x(in);
    for (int i = 0; i < in; i++)
        h_x[i] = __float2half((float)(rand()%200-100)/100.0f);

    // Reshuffle
    auto resh = reshuffle(h_W.data(), out, in);

    // Device allocations
    void *d_W, *d_W_mma, *d_x_q8;
    half *d_x, *d_y_dp4a, *d_y_marlin;
    CHECK_CUDA(cudaMalloc(&d_W, total_blocks * sizeof(block_q4_k)));
    CHECK_CUDA(cudaMalloc(&d_W_mma, resh.size()));
    CHECK_CUDA(cudaMalloc(&d_x, in * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_x_q8, q8_blocks * sizeof(block_q8_1)));
    CHECK_CUDA(cudaMalloc(&d_y_dp4a, out * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_y_marlin, out * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), total_blocks * sizeof(block_q4_k), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_mma, resh.data(), resh.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), in * sizeof(half), cudaMemcpyHostToDevice));

    gwen_quantize_q8_1(d_x, d_x_q8, in);

    CHECK_CUDA(cudaMemset(d_y_dp4a, 0, out * sizeof(half)));
    CHECK_CUDA(cudaMemset(d_y_marlin, 0, out * sizeof(half)));

    gwen_gemv_dp4a(d_W, d_x_q8, d_y_dp4a, out, in, GGMLType::Q4_K);
    gwen_gemv_mma_reshuffled(d_W_mma, d_x, d_y_marlin, out, in);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<half> h_dp4a(out), h_marlin(out);
    CHECK_CUDA(cudaMemcpy(h_dp4a.data(), d_y_dp4a, out * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_marlin.data(), d_y_marlin, out * sizeof(half), cudaMemcpyDeviceToHost));

    float max_abs = 0, max_mag = 0;
    int first_bad = -1;
    for (int i = 0; i < out; i++) {
        float a = __half2float(h_dp4a[i]);
        float b = __half2float(h_marlin[i]);
        float err = fabsf(a - b);
        if (err > max_abs) max_abs = err;
        if (fabsf(a) > max_mag) max_mag = fabsf(a);
        if (first_bad < 0 && err > 1.0f) first_bad = i;
    }

    float norm = (max_mag > 0) ? max_abs / max_mag : max_abs;
    bool ok = (norm < 0.02f);

    if (ok) {
        printf("PASS (max_abs=%.1f, norm=%.4f)\n", max_abs, norm);
    } else {
        printf("FAIL (max_abs=%.1f, norm=%.4f)\n", max_abs, norm);
        for (int i = first_bad; i < out && i < first_bad + 5; i++) {
            float a = __half2float(h_dp4a[i]);
            float b = __half2float(h_marlin[i]);
            printf("    [%d] dp4a=%.4f marlin=%.4f err=%.4f\n", i, a, b, fabsf(a-b));
        }
    }

    cudaFree(d_W); cudaFree(d_W_mma); cudaFree(d_x); cudaFree(d_x_q8);
    cudaFree(d_y_dp4a); cudaFree(d_y_marlin);
    return ok;
}

int main() {
    printf("=== Marlin-style kernel correctness ===\n\n");
    int pass = 0, fail = 0;

    // 0.8B sizes
    test("0.8B attn_gate", 2048, 1024) ? pass++ : fail++;
    test("0.8B ffn_gate",  3584, 1024) ? pass++ : fail++;
    test("0.8B ffn_down",  1024, 3584) ? pass++ : fail++;

    // 4B sizes
    test("4B gate",    5120, 2560) ? pass++ : fail++;
    test("4B ffn_gate",8960, 2560) ? pass++ : fail++;
    test("4B ffn_down",2560, 8960) ? pass++ : fail++;
    test("4B attn_o",  2560, 5120) ? pass++ : fail++;

    printf("\n%d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
