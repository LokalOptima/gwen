#pragma once

#include <cstdint>
#include <cuda_fp16.h>

// GGML quantization block structures — packed to match binary layout exactly
#pragma pack(push, 1)

namespace gwen {

// Q8_0: 32 values per block, 34 bytes
struct block_q8_0 {
    half d;           // delta (scale)
    int8_t qs[32];    // quantized values
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size mismatch");

// Q4_K: 256 values per block, 144 bytes
// d (fp16) + dmin (fp16) + scales (12 bytes, packed 6-bit) + qs (128 bytes, 4-bit)
struct block_q4_k {
    half d;              // super-block scale
    half dmin;           // super-block minimum
    uint8_t scales[12];  // 6-bit scales for 8 sub-blocks (packed)
    uint8_t qs[128];     // 4-bit quantized values (2 per byte)
};
static_assert(sizeof(block_q4_k) == 144, "block_q4_k size mismatch");

// Q5_K: 256 values per block, 176 bytes
struct block_q5_k {
    half d;              // super-block scale
    half dmin;           // super-block minimum
    uint8_t scales[12];  // 6-bit scales for 8 sub-blocks
    uint8_t qh[32];      // 5th bits (high bits)
    uint8_t qs[128];     // 4 low bits (2 per byte)
};
static_assert(sizeof(block_q5_k) == 176, "block_q5_k size mismatch");

// Q6_K: 256 values per block, 210 bytes
struct block_q6_k {
    uint8_t ql[128];     // lower 4 bits of 6-bit quants
    uint8_t qh[64];      // upper 2 bits of 6-bit quants
    int8_t scales[16];   // 8-bit scales for 16 sub-groups of 16
    half d;              // super-block scale
};
static_assert(sizeof(block_q6_k) == 210, "block_q6_k size mismatch");

} // namespace gwen

#pragma pack(pop)
