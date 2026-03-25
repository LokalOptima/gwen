// ggml-defs.h — Minimal type definitions extracted from ggml.h, ggml-impl.h,
// ggml-common.h, ggml-backend.h, ggml-alloc.h, gguf.h, and ggml-cuda.h.
//
// This single header replaces all 6+ llama.cpp headers for GWEN's vendored
// CUDA kernel code.  It contains ONLY the enums, structs, constants, macros,
// and lookup tables that the kernel headers (mmq.cuh, mma.cuh, fattn-*.cuh,
// vecdotq.cuh, convert.cuh, quantize.cuh) actually reference.
//
// NO backend, tensor graph, pool, context, or runtime code is included.

#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// ============================================================================
// CUDA-flavour typedefs (normally set by GGML_COMMON_DECL_CUDA in ggml-common.h)
// ============================================================================

typedef half  ggml_half;
typedef half2 ggml_half2;

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S data

// ============================================================================
// ggml_type enum (from ggml.h)
// ============================================================================

enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    GGML_TYPE_MXFP4   = 39,
    GGML_TYPE_NVFP4   = 40,
    GGML_TYPE_COUNT   = 41,
};

// ============================================================================
// Utility macros (from ggml.h)
// ============================================================================

#define GGML_MAX_DIMS       4
#define GGML_MAX_SRC        10
#define GGML_MAX_OP_PARAMS  64

#define GGML_UNUSED(x) (void)(x)
#ifdef __CUDACC__
template <typename... Args>
__host__ __device__ constexpr inline void ggml_unused_vars_impl(Args&&...) noexcept {}
#define GGML_UNUSED_VARS(...) ggml_unused_vars_impl(__VA_ARGS__)
#else
#define GGML_UNUSED_VARS(...) do { (void)sizeof((__VA_ARGS__, 0)); } while(0)
#endif

#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

// ggml_abort — declared extern, defined in each .cu that uses these headers
extern "C" [[noreturn]] void ggml_abort(const char * file, int line, const char * fmt, ...);

#define GGML_ABORT(...) ggml_abort(__FILE__, __LINE__, __VA_ARGS__)
#define GGML_ASSERT(x) if (!(x)) GGML_ABORT("GGML_ASSERT(%s) failed", #x)

// ============================================================================
// GGML_TABLE_BEGIN/END for CUDA (device-side lookup tables)
// ============================================================================

#define GGML_TABLE_BEGIN(type, name, size) static const __device__ type name[size] = {
#define GGML_TABLE_END() };

// ============================================================================
// Quant block sizes (from ggml-common.h DECL section)
// ============================================================================

#define QK_K 256
#define K_SCALE_SIZE 12

#define QK4_0 32
#define QK4_1 32
#define QK5_0 32
#define QK5_1 32
#define QK8_0 32
#define QK8_1 32
#define QK4_NL 32
#define QK_MXFP4 32
#define QK_NVFP4 64
#define QK_NVFP4_SUB 16

// QR (ratio) and QI (int count) constants for CUDA kernels
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))

#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))

#define QR_MXFP4 2
#define QI_MXFP4 (QK_MXFP4 / (4 * QR_MXFP4))

#define QR_NVFP4 2
#define QI_NVFP4 (QK_NVFP4 / (4 * QR_NVFP4))

#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))

#define QR5_1 2
#define QI5_1 (QK5_1 / (4 * QR5_1))

#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))

#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))

#define QR2_K 4
#define QI2_K (QK_K / (4*QR2_K))

#define QR3_K 4
#define QI3_K (QK_K / (4*QR3_K))

#define QR4_K 2
#define QI4_K (QK_K / (4*QR4_K))

#define QR5_K 2
#define QI5_K (QK_K / (4*QR5_K))

#define QR6_K 2
#define QI6_K (QK_K / (4*QR6_K))

#define QR2_XXS 4
#define QI2_XXS (QK_K / (4*QR2_XXS))

#define QR2_XS 4
#define QI2_XS (QK_K / (4*QR2_XS))

#define QR2_S 4
#define QI2_S (QK_K / (4*QR2_S))

#define QR3_XXS 4
#define QI3_XXS (QK_K / (4*QR3_XXS))

#define QR3_XS 4
#define QI3_XS (QK_K / (4*QR3_XS))

#define QR1_S 8
#define QI1_S (QK_K / (4*QR1_S))

#define QR1_M 8
#define QI1_M (QK_K / (4*QR1_M))

#define QR4_NL 2
#define QI4_NL (QK4_NL / (4*QR4_NL))

#define QR4_XS 2
#define QI4_XS (QK_K / (4*QR4_XS))

#define QR3_S 4
#define QI3_S (QK_K / (4*QR3_S))

// ============================================================================
// GGML_EXTENSION for anonymous unions in CUDA C++
// ============================================================================

#ifdef _MSC_VER
#define GGML_EXTENSION
#else
#define GGML_EXTENSION __extension__
#endif

// ============================================================================
// Quant block structs (from ggml-common.h DECL section, CUDA flavour)
// ============================================================================

typedef struct {
    ggml_half d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");

typedef struct {
    GGML_EXTENSION union {
        struct { ggml_half d; ggml_half m; } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t qs[QK4_1 / 2];
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(ggml_half) + QK4_1 / 2, "wrong q4_1 block size/padding");

typedef struct {
    uint8_t e;
    uint8_t qs[QK_MXFP4/2];
} block_mxfp4;
static_assert(sizeof(block_mxfp4) == sizeof(uint8_t) + QK_MXFP4/2, "wrong mxfp4 block size/padding");

typedef struct {
    uint8_t d[QK_NVFP4/QK_NVFP4_SUB];
    uint8_t qs[QK_NVFP4/2];
} block_nvfp4;
static_assert(sizeof(block_nvfp4) == sizeof(uint8_t)*(QK_NVFP4/QK_NVFP4_SUB) + QK_NVFP4/2, "wrong nvfp4 block size/padding");

typedef struct {
    ggml_half d;
    uint8_t qh[4];
    uint8_t qs[QK5_0 / 2];
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_half) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

typedef struct {
    GGML_EXTENSION union {
        struct { ggml_half d; ggml_half m; } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t qh[4];
    uint8_t qs[QK5_1 / 2];
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_half) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

typedef struct {
    ggml_half d;
    int8_t  qs[QK8_0];
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");

typedef struct {
    GGML_EXTENSION union {
        struct { ggml_half d; ggml_half s; } GGML_COMMON_AGGR_S;
        ggml_half2 ds;
    } GGML_COMMON_AGGR_U;
    int8_t qs[QK8_1];
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_half) + QK8_1, "wrong q8_1 block size/padding");

// Ternary quantization
typedef struct {
    uint8_t qs[(QK_K - 4 * QK_K / 64) / 5];
    uint8_t qh[QK_K/64];
    ggml_half d;
} block_tq1_0;

typedef struct {
    uint8_t qs[QK_K/4];
    ggml_half d;
} block_tq2_0;

// Super-block K-quant structs
typedef struct {
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/4];
    GGML_EXTENSION union {
        struct { ggml_half d; ggml_half dmin; } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_half) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

typedef struct {
    uint8_t hmask[QK_K/8];
    uint8_t qs[QK_K/4];
    uint8_t scales[12];
    ggml_half d;
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(ggml_half) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");

typedef struct {
    GGML_EXTENSION union {
        struct { ggml_half d; ggml_half dmin; } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");

typedef struct {
    GGML_EXTENSION union {
        struct { ggml_half d; ggml_half dmin; } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    ggml_half d;
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_half) + QK_K / 16 + 3*QK_K/4, "wrong q6_K block size/padding");

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "wrong q8_K block size/padding");

// IQ quant structs
typedef struct {
    ggml_half d;
    uint16_t qs[QK_K/8];
} block_iq2_xxs;

typedef struct {
    ggml_half d;
    uint16_t qs[QK_K/8];
    uint8_t  scales[QK_K/32];
} block_iq2_xs;

typedef struct {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t scales[QK_K/32];
} block_iq2_s;

typedef struct {
    ggml_half d;
    uint8_t qs[3*QK_K/8];
} block_iq3_xxs;

#define IQ3S_N_SCALE QK_K/64
typedef struct {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t signs[QK_K/8];
    uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;

typedef struct {
    ggml_half d;
    uint8_t  qs[QK_K/8];
    uint16_t qh[QK_K/32];
} block_iq1_s;

typedef struct {
    uint8_t  qs[QK_K/8];
    uint8_t  qh[QK_K/16];
    uint8_t  scales[QK_K/32];
} block_iq1_m;

typedef union {
    ggml_half f16;
    uint16_t  u16;
} iq1m_scale_t;

typedef struct {
    ggml_half d;
    uint8_t qs[QK4_NL/2];
} block_iq4_nl;

typedef struct {
    ggml_half d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];
    uint8_t  qs[QK_K/2];
} block_iq4_xs;

// ============================================================================
// IQ constants
// ============================================================================

#define NGRID_IQ1S 2048
#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f

// ============================================================================
// Lookup tables (from ggml-common.h IMPL_CUDA section)
// All tables are __device__ qualified for CUDA.
// ============================================================================

// We include ggml-common.h's IMPL section for the large lookup tables.
// The DECL section is skipped because we already defined everything above.
// We pre-define GGML_COMMON_DECL to prevent re-entry of the DECL block.
#define GGML_COMMON_DECL

// Now trigger only the IMPL block:
#ifndef GGML_COMMON_IMPL
#define GGML_COMMON_IMPL_CUDA
#include "llama_ggml_common.h"
#endif

// ============================================================================
// ggml-cuda.h constants
// ============================================================================

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES 16

// ============================================================================
// ggml_fp16_t / ggml_bf16_t (from ggml.h, minimal)
// ============================================================================

typedef uint16_t ggml_fp16_t;
typedef struct { uint16_t bits; } ggml_bf16_t;

// ============================================================================
// Logging stub (referenced by ggml-impl.h paths that we don't use,
// but needed to avoid undefined-symbol errors in some template paths)
// ============================================================================

enum ggml_log_level {
    GGML_LOG_LEVEL_NONE  = 0,
    GGML_LOG_LEVEL_INFO  = 1,
    GGML_LOG_LEVEL_WARN  = 2,
    GGML_LOG_LEVEL_ERROR = 3,
    GGML_LOG_LEVEL_DEBUG = 4,
    GGML_LOG_LEVEL_CONT  = 5,
};

// Stub: no-op logging for GWEN (kernel code never actually logs)
#define GGML_LOG_DEBUG(...) ((void)0)

// end of ggml-defs.h
