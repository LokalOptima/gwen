#pragma once
// MMA tile types and primitives for SM_75+ (Turing / mma.sync).
// Extracted from llama.cpp ggml/src/ggml-cuda/mma.cuh — Turing (I_MAJOR) path only.
// Only the tile types and MMA overloads needed for D=256 flash-attention are retained:
//   tile<16,  8, half2>   — Q/K/V fragment (row-major A or col-major B)
//   tile<16, 16, float>   — KQ accumulator (wide path, ncols>8)
//   tile<16,  8, float>   — KQ accumulator (narrow path, ncols==8)
//   tile<16,  4, half2>   — VKQ accumulator (narrow path, ncols==8)
//   tile< 8,  8, half2>   — B fragment for VKQ MMA
//   tile< 8,  4, half2>   — helper for get_transposed
//   tile<16,  4, half2>   — helper input for get_transposed
//
// Architecture guards: this file is intentionally compiled only for __CUDA_ARCH__ >= 750.
// On SM_120 (Blackwell consumer) __CUDA_ARCH__ == 1200 so TURING_MMA_AVAILABLE is defined
// and Ampere ptx variants (m16n8k16) are used throughout.

#include <cuda_fp16.h>
#include <cassert>

// ---- Architecture availability macros (mirror llama.cpp logic) ---------------
#define WARP_SIZE 32
#define GGML_CUDA_CC_TURING  750
#define GGML_CUDA_CC_AMPERE  800

// Turing MMA (mma.sync) available on SM_75+
#if !defined(GGML_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#  define TURING_MMA_AVAILABLE
#endif

// Ampere path (m16n8k16 wider PTX) available on SM_80+
#if !defined(GGML_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#  define AMPERE_MMA_AVAILABLE
#endif

// ldmatrix.trans available on Turing+
#if defined(TURING_MMA_AVAILABLE)
#  define LDMATRIX_TRANS_AVAILABLE
#endif

// ---- Utilities ---------------------------------------------------------------

static __device__ __forceinline__ constexpr int gwen_cuda_get_physical_warp_size() {
    return WARP_SIZE;
}

// Compile-time unroll helper (identical to ggml_cuda_unroll in llama.cpp)
template<int n>
struct gwen_cuda_unroll {
    template<typename F, typename... Args>
    __device__ __forceinline__ void operator()(F f, Args&&... args) const {
        f(n - 1, args...);
        gwen_cuda_unroll<n - 1>{}(f, args...);
    }
};
template<>
struct gwen_cuda_unroll<1> {
    template<typename F, typename... Args>
    __device__ __forceinline__ void operator()(F f, Args&&... args) const {
        f(0, args...);
    }
};

// ---- movmatrix (8x8 b16 transpose) -------------------------------------------
// Available natively via PTX on sm_80+; emulated with shuffles on older arches.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

static __device__ __forceinline__ int gwen_cuda_movmatrix(const int x) {
    int ret = 0;
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
        : "=r"(ret) : "r"(x));
    return ret;
}

#else  // Turing: emulate with warp shuffles

static __device__ __forceinline__ int gwen_cuda_movmatrix(const int x) {
    const int src_i_low  = 2 * (threadIdx.x % 4);
    const int src_i_high = src_i_low + 1;
    const int src_j      = threadIdx.x / 4;

    const int src_laneid_low  = src_i_low  * 4 + src_j / 2;
    const int src_laneid_high = src_i_high * 4 + src_j / 2;

    const int shift_low  = ((src_j + 0) % 2) * 16;
    const int shift_high = ((src_j + 1) % 2) * 16;

    const int ret_low  = (__shfl_sync(0xFFFFFFFF, x, src_laneid_low,  WARP_SIZE) >> shift_low)  & 0x0000FFFF;
    const int ret_high = (__shfl_sync(0xFFFFFFFF, x, src_laneid_high, WARP_SIZE) << shift_high) & 0xFFFF0000;

    return ret_low | ret_high;
}

#endif  // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

static __device__ __forceinline__ half2 gwen_cuda_movmatrix(const half2 x) {
    half2 ret;
    *((int *) &ret) = gwen_cuda_movmatrix(*((const int *) &x));
    return ret;
}

// ---- Tile types --------------------------------------------------------------
// Only the DATA_LAYOUT_I_MAJOR specialisations needed for D=256 are included.
// Layout conventions (same as llama.cpp mma.cuh):
//   A: row-major M×K
//   B: column-major K×N
//   C: column-major M×N
// All represented as I-major tiles (I == row dimension in the logical matrix).
// ne = number of physical 32-bit elements per warp.

namespace gwen_mma {

// ------ tile<16, 8, half2> — A/B fragment for KQ and VKQ MMA ----------------
// ne = 16*8/32 = 4  half2 elements per thread
struct tile_16_8_h2 {
    static constexpr int I  = 16;
    static constexpr int J  = 8;
    static constexpr int ne = I * J / WARP_SIZE;   // = 4

    half2 x[ne] = {{0.0f, 0.0f}};

    // Physical row index for element l  (l in 0..ne-1)
    static __device__ __forceinline__ int get_i(const int l) {
        // Each row-group of 8 threads covers 8 rows (8 consecutive lanes / 4 = 2 rows each)
        return ((l % 2) * 8) + (threadIdx.x / 4);
    }
    // Physical column index for element l (measured in half2 units)
    static __device__ __forceinline__ int get_j(const int l) {
        return ((l / 2) * 4) + (threadIdx.x % 4);
    }
};

// ------ tile<8, 8, half2> — B fragment for VKQ MMA (narrow path) ------------
// ne = 8*8/32 = 2  half2 elements per thread
struct tile_8_8_h2 {
    static constexpr int I  = 8;
    static constexpr int J  = 8;
    static constexpr int ne = I * J / WARP_SIZE;   // = 2

    half2 x[ne] = {{0.0f, 0.0f}};

    static __device__ __forceinline__ int get_i(const int /*l*/) {
        return threadIdx.x / 4;
    }
    static __device__ __forceinline__ int get_j(const int l) {
        return (l * 4) + (threadIdx.x % 4);
    }
};

// ------ tile<16, 4, half2> — VKQ accumulator (narrow path, ncols==8) --------
// ne = 16*4/32 = 2  half2 elements per thread
struct tile_16_4_h2 {
    static constexpr int I  = 16;
    static constexpr int J  = 4;
    static constexpr int ne = I * J / WARP_SIZE;   // = 2

    half2 x[ne] = {{0.0f, 0.0f}};

    static __device__ __forceinline__ int get_i(const int l) {
        return (l * 8) + (threadIdx.x / 4);
    }
    static __device__ __forceinline__ int get_j(const int /*l*/) {
        return threadIdx.x % 4;
    }
};

// ------ tile<16, 8, float> — KQ accumulator (narrow path, ncols==8) ---------
// ne = 16*8/32 = 4  float elements per thread
struct tile_16_8_f32 {
    static constexpr int I  = 16;
    static constexpr int J  = 8;
    static constexpr int ne = I * J / WARP_SIZE;   // = 4

    float x[ne] = {0.0f};

    // get_i / get_j match the half2 tile (same physical layout, different element type)
    static __device__ __forceinline__ int get_i(const int l) {
        return (((l / 2) % 2) * 8) + (threadIdx.x / 4);
    }
    static __device__ __forceinline__ int get_j(const int l) {
        return ((l / 4) * 8) + ((threadIdx.x % 4) * 2) + (l % 2);
    }
};

// ------ tile<16, 16, float> — KQ accumulator (wide path, ncols>8) -----------
// ne = 16*16/32 = 8  float elements per thread
struct tile_16_16_f32 {
    static constexpr int I  = 16;
    static constexpr int J  = 16;
    static constexpr int ne = I * J / WARP_SIZE;   // = 8

    float x[ne] = {0.0f};

    // For the float C tile the layout follows the CUDA PTX convention for
    // m16n8k16 with two n=8 halves packed side-by-side.
    static __device__ __forceinline__ int get_i(const int l) {
        return (((l / 2) % 2) * 8) + (threadIdx.x / 4);
    }
    static __device__ __forceinline__ int get_j(const int l) {
        return ((l / 4) * 8) + ((threadIdx.x % 4) * 2) + (l % 2);
    }
};

// ---- load_ldmatrix -----------------------------------------------------------
// Loads a tile from shared memory using the ldmatrix PTX instruction.
// xs0 must be a __shared__ pointer aligned to 16 bytes.
// stride is in units of T (half2 or float).

// load_ldmatrix for tile_16_8_h2  (ldmatrix.x4)
static __device__ __forceinline__ void load_ldmatrix(
        tile_16_8_h2 & t, const half2 * __restrict__ xs0, const int stride) {
#ifdef TURING_MMA_AVAILABLE
    int * xi = (int *) t.x;
    const int * xs = (const int *) xs0
        + (threadIdx.x % t.I) * stride
        + (threadIdx.x / t.I) * (t.J / 2);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(xs));
#else
    // Generic fallback (should not be reached for target SM)
    #pragma unroll
    for (int l = 0; l < t.ne; ++l) {
        t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
    }
#endif
}

// load_ldmatrix for tile_8_8_h2  (ldmatrix.x2)
static __device__ __forceinline__ void load_ldmatrix(
        tile_8_8_h2 & t, const half2 * __restrict__ xs0, const int stride) {
#ifdef TURING_MMA_AVAILABLE
    int * xi = (int *) t.x;
    const int * xs = (const int *) xs0
        + (threadIdx.x % t.I) * stride
        + ((threadIdx.x / t.I) * (t.J / 2)) % t.J;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(xi[0]), "=r"(xi[1])
        : "l"(xs));
#else
    #pragma unroll
    for (int l = 0; l < t.ne; ++l) {
        t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
    }
#endif
}

// load_ldmatrix for tile_16_4_h2  (ldmatrix.x2)
static __device__ __forceinline__ void load_ldmatrix(
        tile_16_4_h2 & t, const half2 * __restrict__ xs0, const int stride) {
#ifdef TURING_MMA_AVAILABLE
    int * xi = (int *) t.x;
    const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(xi[0]), "=r"(xi[1])
        : "l"(xs));
#else
    #pragma unroll
    for (int l = 0; l < t.ne; ++l) {
        t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
    }
#endif
}

// load_ldmatrix_trans for tile_16_8_h2  (ldmatrix.x4.trans)
// Loads V transposed from shared memory (columns become rows in registers).
static __device__ __forceinline__ void load_ldmatrix_trans(
        tile_16_8_h2 & t, const half2 * __restrict__ xs0, const int stride) {
#ifdef TURING_MMA_AVAILABLE
    int * xi = (int *) t.x;
    const int * xs = (const int *) xs0
        + (threadIdx.x % t.I) * stride
        + (threadIdx.x / t.I) * (t.J / 2);
    // Note: output registers are reordered vs non-transposed load to match
    // the PTX .trans semantics (xi[0]/xi[2] swap vs xi[1]/xi[3] swap).
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
        : "l"(xs));
#else
    // Should not be reached: LDMATRIX_TRANS_AVAILABLE == TURING_MMA_AVAILABLE
    (void)t; (void)xs0; (void)stride;
    assert(false);
#endif
}

// ---- get_half2 ---------------------------------------------------------------
// Converts a float tile to a half2 tile by packing consecutive float pairs.

static __device__ __forceinline__ tile_16_8_h2 get_half2(const tile_16_16_f32 & tf) {
    // tile<16,16,float> -> tile<16,8,half2>: pack floats[l0], floats[l0+1]
    tile_16_8_h2 ret;
    #pragma unroll
    for (int l0 = 0; l0 < tile_16_16_f32::ne; l0 += 2) {
        ret.x[l0/2] = make_half2(tf.x[l0 + 0], tf.x[l0 + 1]);
    }
    return ret;
}

static __device__ __forceinline__ tile_8_8_h2 get_half2_8x8(const tile_16_8_f32 & tf) {
    // tile<16,8,float> -> tile<8,8,half2>: pack pairs (same layout, just narrower)
    // Used for narrow path (ncols==8): KQ_C is tile<16,8,float>, B is tile<8,8,half2>
    tile_8_8_h2 ret;
    #pragma unroll
    for (int l0 = 0; l0 < tile_16_8_f32::ne; l0 += 2) {
        ret.x[l0/2] = make_half2(tf.x[l0 + 0], tf.x[l0 + 1]);
    }
    return ret;
}

// ---- get_transposed ----------------------------------------------------------
// Converts tile<16,4,half2> to tile<8,8,half2> using movmatrix (B matrix for VKQ).
// Used in the narrow path (ncols==8) to convert KQ accumulator -> VKQ B fragment.

static __device__ __forceinline__ tile_8_8_h2 get_transposed(const tile_16_4_h2 & t) {
    tile_8_8_h2 ret;
    ret.x[0] = gwen_cuda_movmatrix(t.x[0]);
    ret.x[1] = gwen_cuda_movmatrix(t.x[1]);
    return ret;
}

// ---- mma() overloads ---------------------------------------------------------
// All use mma.sync.aligned PTX.
// On Ampere (SM_80+, including SM_120): m16n8k16 is available.
// On Turing (SM_75): m16n8k8 used twice instead.

// KQ MMA (narrow, ncols==8):
//   D: tile<16,8,float>   C accumulator
//   A: tile<16,8,half2>   K fragment (row-major)
//   B: tile<8,8,half2>    Q fragment (col-major)
static __device__ __forceinline__ void mma(
        tile_16_8_f32 & D, const tile_16_8_h2 & A, const tile_8_8_h2 & B) {
#ifdef TURING_MMA_AVAILABLE
    const int * Axi = (const int *) A.x;
    const int * Bxi = (const int *) B.x;
    int       * Dxi = (int       *) D.x;
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
    // Turing: 2x m16n8k8
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif
#endif
}

// KQ MMA (wide, ncols>8):
//   D: tile<16,16,float>  C accumulator
//   A: tile<16,8,half2>   K fragment (row-major)
//   B: tile<16,8,half2>   Q fragment (col-major, treated as two n=8 halves)
// Note: CUDA wide path swaps A and B vs the AMD path; here we follow the CUDA convention
// (caller passes Q as B and K as A).
static __device__ __forceinline__ void mma(
        tile_16_16_f32 & D, const tile_16_8_h2 & A, const tile_16_8_h2 & B) {
#ifdef TURING_MMA_AVAILABLE
    const int * Axi = (const int *) A.x;
    const int * Bxi = (const int *) B.x;
    int       * Dxi = (int       *) D.x;
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[2]));
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(Dxi[4]), "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]), "r"(Bxi[3]));
#else
    // Turing: 4x m16n8k8
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[2]));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(Dxi[4]), "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[1]));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(Dxi[4]), "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
        : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[3]));
#endif
#endif
}

// VKQ MMA (narrow, ncols==8):
//   D: tile<16,4,half2>   VKQ accumulator (row-major)
//   A: tile<16,8,half2>   V^T fragment (loaded via ldmatrix_trans)
//   B: tile<8,8,half2>    softmax(KQ) fragment (from get_transposed)
static __device__ __forceinline__ void mma(
        tile_16_4_h2 & D, const tile_16_8_h2 & A, const tile_8_8_h2 & B) {
#ifdef TURING_MMA_AVAILABLE
    const int * Axi = (const int *) A.x;
    const int * Bxi = (const int *) B.x;
    int       * Dxi = (int       *) D.x;
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(Dxi[0]), "+r"(Dxi[1])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
    // Turing: 2x m16n8k8
    asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
        : "+r"(Dxi[0]), "+r"(Dxi[1])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
    asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
        : "+r"(Dxi[0]), "+r"(Dxi[1])
        : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif
#endif
}

// VKQ MMA (wide, ncols>8):
//   D: tile<16,8,half2>   VKQ accumulator (col-major)
//   A: tile<16,8,half2>   softmax(KQ) fragment (B in logical sense, swapped for CUDA)
//   B: tile<16,8,half2>   V^T fragment
// The CUDA wide path swaps A and B: mma(D, softmax_KQ, V_trans)
static __device__ __forceinline__ void mma(
        tile_16_8_h2 & D, const tile_16_8_h2 & A, const tile_16_8_h2 & B) {
#ifdef TURING_MMA_AVAILABLE
    const int * Axi = (const int *) A.x;
    const int * Bxi = (const int *) B.x;
    int       * Dxi = (int       *) D.x;
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(Dxi[0]), "+r"(Dxi[1])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[2]));
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]), "r"(Bxi[3]));
#else
    // Turing: 4x m16n8k8
    asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
        : "+r"(Dxi[0]), "+r"(Dxi[1])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
    asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
        : "+r"(Dxi[0]), "+r"(Dxi[1])
        : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[2]));
    asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
        : "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[1]));
    asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
        : "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[3]));
#endif
#endif
}

}  // namespace gwen_mma
