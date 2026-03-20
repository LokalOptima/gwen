// Correctness test: compare gemv_dp4a_async and gemv_mma against dp4a baseline
// Build: cmake --build build -j && nvcc -arch=sm_120 -O3 -I include -L build -l gwen_core
// Actually just link against the library from build dir.
//
// Compile: nvcc -arch=sm_120 -O3 --expt-relaxed-constexpr --use_fast_math \
//          -I include -I third_party/cutlass/include \
//          -o scratch/test_gemv_variants scratch/test_gemv_variants.cu \
//          -L build -l gwen_core -lcudart
//
// Run: LD_LIBRARY_PATH=build ./scratch/test_gemv_variants

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
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

// Fill Q4_K blocks with deterministic pseudo-random data
static void fill_q4k_random(block_q4_k* blocks, int n_blocks, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n_blocks; i++) {
        float d = (float)(rand() % 100 + 1) / 100.0f;
        float dmin = (float)(rand() % 50) / 100.0f;
        blocks[i].d = __float2half(d);
        blocks[i].dmin = __float2half(dmin);
        for (int j = 0; j < 12; j++)
            blocks[i].scales[j] = rand() % 256;
        for (int j = 0; j < 128; j++)
            blocks[i].qs[j] = rand() % 256;
    }
}

// Fill Q5_K blocks
static void fill_q5k_random(block_q5_k* blocks, int n_blocks, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n_blocks; i++) {
        float d = (float)(rand() % 100 + 1) / 100.0f;
        float dmin = (float)(rand() % 50) / 100.0f;
        blocks[i].d = __float2half(d);
        blocks[i].dmin = __float2half(dmin);
        for (int j = 0; j < 12; j++)
            blocks[i].scales[j] = rand() % 256;
        for (int j = 0; j < 32; j++)
            blocks[i].qh[j] = rand() % 256;
        for (int j = 0; j < 128; j++)
            blocks[i].qs[j] = rand() % 256;
    }
}

// Fill Q6_K blocks
static void fill_q6k_random(block_q6_k* blocks, int n_blocks, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n_blocks; i++) {
        float d = (float)(rand() % 100 + 1) / 200.0f;
        blocks[i].d = __float2half(d);
        for (int j = 0; j < 128; j++)
            blocks[i].ql[j] = rand() % 256;
        for (int j = 0; j < 64; j++)
            blocks[i].qh[j] = rand() % 256;
        for (int j = 0; j < 16; j++)
            blocks[i].scales[j] = (int8_t)(rand() % 256 - 128);
    }
}

// Fill FP16 vector with random values
static void fill_fp16_random(half* data, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; i++)
        data[i] = __float2half((float)(rand() % 200 - 100) / 100.0f);
}

struct TestConfig {
    const char* name;
    int out_features;
    int in_features;
    GGMLType type;
};

static float max_abs_diff(const half* a, const half* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float fa = __half2float(a[i]);
        float fb = __half2float(b[i]);
        float err = fabsf(fa - fb);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static float max_rel_diff(const half* a, const half* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float fa = __half2float(a[i]);
        float fb = __half2float(b[i]);
        float denom = fmaxf(fabsf(fa), fabsf(fb));
        if (denom > 1e-6f) {
            float err = fabsf(fa - fb) / denom;
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

// CPU-side Q4_K reshuffling for test
static void unpack_scales_test(const uint8_t sp[12], uint8_t sc[8], uint8_t mn[8]) {
    for (int sb = 0; sb < 4; sb++) { sc[sb] = sp[sb] & 0x3F; mn[sb] = sp[sb+4] & 0x3F; }
    for (int sb = 4; sb < 8; sb++) {
        sc[sb] = (sp[sb+4] & 0xF) | ((sp[sb-4] >> 6) << 4);
        mn[sb] = (sp[sb+4] >> 4) | ((sp[sb] >> 6) << 4);
    }
}
static uint8_t q4k_nib(const uint8_t qs[128], int e) {
    int b = (e/64)*32 + (e%32);
    return ((e%64)>=32) ? (qs[b]>>4) : (qs[b]&0xF);
}
static std::vector<uint8_t> reshuffle_q4k_test(const block_q4_k* blocks, int out, int in) {
    int bpr = in/256, nrt = (out+15)/16;
    std::vector<uint8_t> r(nrt*bpr*2368, 0);
    for (int rt = 0; rt < nrt; rt++) for (int blk = 0; blk < bpr; blk++) {
        uint8_t* t = r.data() + (rt*bpr+blk)*2368;
        half* td = (half*)(t+2048); half* tdm = (half*)(t+2080);
        uint8_t* tsc = t+2112; uint8_t* tmn = t+2240;
        for (int row = 0; row < 16; row++) {
            int gr = rt*16+row;
            if (gr >= out) { td[row]=__float2half(0.f); tdm[row]=__float2half(0.f); continue; }
            const auto& s = blocks[gr*bpr+blk];
            td[row]=s.d; tdm[row]=s.dmin;
            unpack_scales_test(s.scales, &tsc[row*8], &tmn[row*8]);
            for (int ch = 0; ch < 16; ch++) {
                int sb=ch/2, hsb=ch%2;
                for (int th = 0; th < 32; th++) {
                    int r0=th/4, r1=th/4+8, k0=(th%4)*2, k1=k0+1, k8=k0+8, k9=k1+8;
                    uint8_t* d = t+ch*128+th*4;
                    if (row==r0) { d[0]=q4k_nib(s.qs,sb*32+hsb*16+k0)|(q4k_nib(s.qs,sb*32+hsb*16+k1)<<4);
                                   d[2]=q4k_nib(s.qs,sb*32+hsb*16+k8)|(q4k_nib(s.qs,sb*32+hsb*16+k9)<<4); }
                    if (row==r1) { d[1]=q4k_nib(s.qs,sb*32+hsb*16+k0)|(q4k_nib(s.qs,sb*32+hsb*16+k1)<<4);
                                   d[3]=q4k_nib(s.qs,sb*32+hsb*16+k8)|(q4k_nib(s.qs,sb*32+hsb*16+k9)<<4); }
                }
            }
        }
    }
    return r;
}

static bool test_q4k(const TestConfig& cfg) {
    printf("  %-30s %5d x %-5d Q4_K ... ", cfg.name, cfg.out_features, cfg.in_features);

    int blocks_per_row = cfg.in_features / 256;
    int total_blocks = cfg.out_features * blocks_per_row;
    int q8_blocks = cfg.in_features / 32;

    // Host allocations
    std::vector<block_q4_k> h_W(total_blocks);
    std::vector<half> h_x(cfg.in_features);
    std::vector<half> h_y_dp4a(cfg.out_features);
    std::vector<half> h_y_async(cfg.out_features);
    std::vector<half> h_y_mma(cfg.out_features);

    fill_q4k_random(h_W.data(), total_blocks, 42);
    fill_fp16_random(h_x.data(), cfg.in_features, 123);

    // Device allocations
    void *d_W, *d_x_q8;
    half *d_x, *d_y_dp4a, *d_y_async, *d_y_mma;
    size_t w_bytes = total_blocks * sizeof(block_q4_k);
    size_t q8_bytes = q8_blocks * sizeof(block_q8_1);

    CHECK_CUDA(cudaMalloc(&d_W, w_bytes));
    CHECK_CUDA(cudaMalloc(&d_x, cfg.in_features * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_x_q8, q8_bytes));
    CHECK_CUDA(cudaMalloc(&d_y_dp4a, cfg.out_features * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_y_async, cfg.out_features * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_y_mma, cfg.out_features * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), w_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), cfg.in_features * sizeof(half), cudaMemcpyHostToDevice));

    // Quantize input for dp4a
    gwen_quantize_q8_1(d_x, d_x_q8, cfg.in_features);

    // Run baseline dp4a
    CHECK_CUDA(cudaMemset(d_y_dp4a, 0, cfg.out_features * sizeof(half)));
    gwen_gemv_dp4a(d_W, d_x_q8, d_y_dp4a, cfg.out_features, cfg.in_features, GGMLType::Q4_K);

    // Run async dp4a
    CHECK_CUDA(cudaMemset(d_y_async, 0, cfg.out_features * sizeof(half)));
    gwen_gemv_dp4a_async(d_W, d_x_q8, d_y_async, cfg.out_features, cfg.in_features, GGMLType::Q4_K);

    // Run mma (takes FP16 input directly)
    CHECK_CUDA(cudaMemset(d_y_mma, 0, cfg.out_features * sizeof(half)));
    gwen_gemv_mma(d_W, d_x, d_y_mma, cfg.out_features, cfg.in_features, GGMLType::Q4_K);

    // Run mma reshuffled
    auto reshuffled = reshuffle_q4k_test(h_W.data(), cfg.out_features, cfg.in_features);
    void* d_W_mma;
    CHECK_CUDA(cudaMalloc(&d_W_mma, reshuffled.size()));
    CHECK_CUDA(cudaMemcpy(d_W_mma, reshuffled.data(), reshuffled.size(), cudaMemcpyHostToDevice));
    std::vector<half> h_y_reshuf(cfg.out_features);
    half* d_y_reshuf;
    CHECK_CUDA(cudaMalloc(&d_y_reshuf, cfg.out_features * sizeof(half)));
    CHECK_CUDA(cudaMemset(d_y_reshuf, 0, cfg.out_features * sizeof(half)));
    gwen_gemv_mma_reshuffled(d_W_mma, d_x, d_y_reshuf, cfg.out_features, cfg.in_features);

    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy results back
    CHECK_CUDA(cudaMemcpy(h_y_dp4a.data(), d_y_dp4a, cfg.out_features * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y_async.data(), d_y_async, cfg.out_features * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y_mma.data(), d_y_mma, cfg.out_features * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y_reshuf.data(), d_y_reshuf, cfg.out_features * sizeof(half), cudaMemcpyDeviceToHost));

    // Compare async vs dp4a (should be identical — same math)
    float async_err = max_abs_diff(h_y_dp4a.data(), h_y_async.data(), cfg.out_features);

    // Compare mma vs dp4a
    float mma_abs = max_abs_diff(h_y_dp4a.data(), h_y_mma.data(), cfg.out_features);

    // Compare reshuffled vs mma (should be identical — same math, different data layout)
    float reshuf_vs_mma = max_abs_diff(h_y_mma.data(), h_y_reshuf.data(), cfg.out_features);

    // Compare reshuffled vs dp4a
    float reshuf_abs = max_abs_diff(h_y_dp4a.data(), h_y_reshuf.data(), cfg.out_features);

    bool async_ok = (async_err < 0.01f);

    float max_mag = 0.0f;
    for (int i = 0; i < cfg.out_features; i++)
        max_mag = fmaxf(max_mag, fabsf(__half2float(h_y_dp4a[i])));
    float mma_norm = (max_mag > 0) ? mma_abs / max_mag : mma_abs;
    float reshuf_norm = (max_mag > 0) ? reshuf_abs / max_mag : reshuf_abs;
    bool mma_ok = (mma_norm < 0.02f);
    bool reshuf_ok = (reshuf_vs_mma < 0.01f);  // reshuffled should match mma exactly

    if (async_ok && mma_ok && reshuf_ok) {
        printf("PASS (async=%.4f, mma_n=%.4f, reshuf_vs_mma=%.4f)\n",
               async_err, mma_norm, reshuf_vs_mma);
    } else {
        printf("FAIL\n");
        if (!async_ok) printf("    async vs dp4a: %.4f\n", async_err);
        if (!mma_ok) printf("    mma norm: %.4f\n", mma_norm);
        if (!reshuf_ok) printf("    reshuf vs mma: %.4f (should be ~0)\n", reshuf_vs_mma);
    }

    cudaFree(d_W); cudaFree(d_x); cudaFree(d_x_q8);
    cudaFree(d_y_dp4a); cudaFree(d_y_async); cudaFree(d_y_mma);
    cudaFree(d_W_mma); cudaFree(d_y_reshuf);

    return async_ok && mma_ok && reshuf_ok;
}

static bool test_q5k(const TestConfig& cfg) {
    printf("  %-30s %5d x %-5d Q5_K ... ", cfg.name, cfg.out_features, cfg.in_features);

    int blocks_per_row = cfg.in_features / 256;
    int total_blocks = cfg.out_features * blocks_per_row;
    int q8_blocks = cfg.in_features / 32;

    std::vector<block_q5_k> h_W(total_blocks);
    std::vector<half> h_x(cfg.in_features);
    std::vector<half> h_y_dp4a(cfg.out_features);
    std::vector<half> h_y_async(cfg.out_features);
    std::vector<half> h_y_mma(cfg.out_features);

    fill_q5k_random(h_W.data(), total_blocks, 42);
    fill_fp16_random(h_x.data(), cfg.in_features, 123);

    void *d_W, *d_x_q8;
    half *d_x, *d_y_dp4a, *d_y_async, *d_y_mma;
    CHECK_CUDA(cudaMalloc(&d_W, total_blocks * sizeof(block_q5_k)));
    CHECK_CUDA(cudaMalloc(&d_x, cfg.in_features * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_x_q8, q8_blocks * sizeof(block_q8_1)));
    CHECK_CUDA(cudaMalloc(&d_y_dp4a, cfg.out_features * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_y_async, cfg.out_features * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_y_mma, cfg.out_features * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), total_blocks * sizeof(block_q5_k), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), cfg.in_features * sizeof(half), cudaMemcpyHostToDevice));
    gwen_quantize_q8_1(d_x, d_x_q8, cfg.in_features);

    gwen_gemv_dp4a(d_W, d_x_q8, d_y_dp4a, cfg.out_features, cfg.in_features, GGMLType::Q5_K);
    gwen_gemv_dp4a_async(d_W, d_x_q8, d_y_async, cfg.out_features, cfg.in_features, GGMLType::Q5_K);
    gwen_gemv_mma(d_W, d_x, d_y_mma, cfg.out_features, cfg.in_features, GGMLType::Q5_K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_y_dp4a.data(), d_y_dp4a, cfg.out_features * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y_async.data(), d_y_async, cfg.out_features * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y_mma.data(), d_y_mma, cfg.out_features * sizeof(half), cudaMemcpyDeviceToHost));

    float async_err = max_abs_diff(h_y_dp4a.data(), h_y_async.data(), cfg.out_features);
    float mma_abs = max_abs_diff(h_y_dp4a.data(), h_y_mma.data(), cfg.out_features);

    bool async_ok = (async_err < 0.01f);
    float max_mag = 0.0f;
    for (int i = 0; i < cfg.out_features; i++)
        max_mag = fmaxf(max_mag, fabsf(__half2float(h_y_dp4a[i])));
    float normalized_err = (max_mag > 0) ? mma_abs / max_mag : mma_abs;
    bool mma_ok = (normalized_err < 0.02f);

    if (async_ok && mma_ok)
        printf("PASS (async_err=%.4f, mma_abs=%.1f, norm=%.4f)\n", async_err, mma_abs, normalized_err);
    else {
        printf("FAIL\n");
        if (!async_ok) printf("    async vs dp4a: max_abs_err=%.4f\n", async_err);
        if (!mma_ok) printf("    mma vs dp4a: max_abs=%.1f, normalized=%.4f\n", mma_abs, normalized_err);
    }

    cudaFree(d_W); cudaFree(d_x); cudaFree(d_x_q8);
    cudaFree(d_y_dp4a); cudaFree(d_y_async); cudaFree(d_y_mma);
    return async_ok && mma_ok;
}

static bool test_q6k(const TestConfig& cfg) {
    printf("  %-30s %5d x %-5d Q6_K ... ", cfg.name, cfg.out_features, cfg.in_features);

    int blocks_per_row = cfg.in_features / 256;
    int total_blocks = cfg.out_features * blocks_per_row;
    int q8_blocks = cfg.in_features / 32;

    std::vector<block_q6_k> h_W(total_blocks);
    std::vector<half> h_x(cfg.in_features);
    std::vector<half> h_y_dp4a(cfg.out_features);
    std::vector<half> h_y_async(cfg.out_features);
    std::vector<half> h_y_mma(cfg.out_features);

    fill_q6k_random(h_W.data(), total_blocks, 42);
    fill_fp16_random(h_x.data(), cfg.in_features, 123);

    void *d_W, *d_x_q8;
    half *d_x, *d_y_dp4a, *d_y_async, *d_y_mma;
    CHECK_CUDA(cudaMalloc(&d_W, total_blocks * sizeof(block_q6_k)));
    CHECK_CUDA(cudaMalloc(&d_x, cfg.in_features * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_x_q8, q8_blocks * sizeof(block_q8_1)));
    CHECK_CUDA(cudaMalloc(&d_y_dp4a, cfg.out_features * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_y_async, cfg.out_features * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_y_mma, cfg.out_features * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), total_blocks * sizeof(block_q6_k), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), cfg.in_features * sizeof(half), cudaMemcpyHostToDevice));
    gwen_quantize_q8_1(d_x, d_x_q8, cfg.in_features);

    gwen_gemv_dp4a(d_W, d_x_q8, d_y_dp4a, cfg.out_features, cfg.in_features, GGMLType::Q6_K);
    gwen_gemv_dp4a_async(d_W, d_x_q8, d_y_async, cfg.out_features, cfg.in_features, GGMLType::Q6_K);
    gwen_gemv_mma(d_W, d_x, d_y_mma, cfg.out_features, cfg.in_features, GGMLType::Q6_K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_y_dp4a.data(), d_y_dp4a, cfg.out_features * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y_async.data(), d_y_async, cfg.out_features * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y_mma.data(), d_y_mma, cfg.out_features * sizeof(half), cudaMemcpyDeviceToHost));

    float async_err = max_abs_diff(h_y_dp4a.data(), h_y_async.data(), cfg.out_features);
    float mma_abs = max_abs_diff(h_y_dp4a.data(), h_y_mma.data(), cfg.out_features);

    bool async_ok = (async_err < 0.01f);
    float max_mag = 0.0f;
    for (int i = 0; i < cfg.out_features; i++)
        max_mag = fmaxf(max_mag, fabsf(__half2float(h_y_dp4a[i])));
    float normalized_err = (max_mag > 0) ? mma_abs / max_mag : mma_abs;
    bool mma_ok = (normalized_err < 0.02f);

    if (async_ok && mma_ok)
        printf("PASS (async_err=%.4f, mma_abs=%.1f, norm=%.4f)\n", async_err, mma_abs, normalized_err);
    else {
        printf("FAIL\n");
        if (!async_ok) printf("    async vs dp4a: max_abs_err=%.4f\n", async_err);
        if (!mma_ok) printf("    mma vs dp4a: max_abs=%.1f, normalized=%.4f\n", mma_abs, normalized_err);
    }

    cudaFree(d_W); cudaFree(d_x); cudaFree(d_x_q8);
    cudaFree(d_y_dp4a); cudaFree(d_y_async); cudaFree(d_y_mma);
    return async_ok && mma_ok;
}

int main() {
    printf("=== GEMV Variant Correctness Tests ===\n\n");

    // Test configurations matching real model dimensions
    TestConfig q4k_tests[] = {
        {"attn_gate (1024→2048)",     2048,   1024, GGMLType::Q4_K},
        {"ffn_gate (1024→3584)",      3584,   1024, GGMLType::Q4_K},
        {"ffn_down (3584→1024)",      1024,   3584, GGMLType::Q4_K},
        {"attn_output (2048→1024)",   1024,   2048, GGMLType::Q4_K},
        {"small (64→1024)",             64,   1024, GGMLType::Q4_K},
    };

    TestConfig q5k_tests[] = {
        {"attn_qkv (1024→6144)",      6144,   1024, GGMLType::Q5_K},
        {"ssm_out (2048→1024)",       1024,   2048, GGMLType::Q5_K},
    };

    TestConfig q6k_tests[] = {
        {"ffn_down_q6k (3584→1024)",  1024,   3584, GGMLType::Q6_K},
        {"token_embd (248320→1024)",  248320, 1024, GGMLType::Q6_K},
    };

    int pass = 0, fail = 0;

    printf("Q4_K tests:\n");
    for (auto& t : q4k_tests) {
        if (test_q4k(t)) pass++; else fail++;
    }

    printf("\nQ5_K tests:\n");
    for (auto& t : q5k_tests) {
        if (test_q5k(t)) pass++; else fail++;
    }

    printf("\nQ6_K tests:\n");
    for (auto& t : q6k_tests) {
        if (test_q6k(t)) pass++; else fail++;
    }

    printf("\n=== Results: %d passed, %d failed ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
