// CUTLASS 3.x SM120 FP8 GEMM with groupwise scaling (per-row weight scales)
// Reference: examples/87b_blackwell_geforce_fp8_bf16_gemm_groupwise.cu
//
// A (weights):     FP8 E4M3  [M, K] RowMajor,    per-row F32 scales (ScaleGranularityM=1)
// B (activations): FP8 E4M3  [K, N] ColumnMajor,  per-block F32 scales (128×128)
// D (output):      FP16      [M, N] ColumnMajor
// Accumulator:     F32

#include "gwen/kernels.h"

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"

#include <cuda_fp8.h>

using namespace cute;

// ============================================================
// CUTLASS 3.x SM120 FP8 GEMM Configuration
// ============================================================

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)

namespace {

// Element types
using ElementA            = cutlass::float_e4m3_t;              // weights
using LayoutA             = cutlass::layout::RowMajor;          // [M, K]
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16

using ElementB            = cutlass::float_e4m3_t;              // activations (quantized on-the-fly)
using LayoutB             = cutlass::layout::ColumnMajor;       // [K, N]
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16

using ElementC            = cutlass::bfloat16_t;                 // output BF16 (SM120 blockwise requirement)
using LayoutC             = cutlass::layout::RowMajor;          // [M, N] — SM120 blockwise requires RowMajor
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8

using ElementD            = ElementC;
using LayoutD             = LayoutC;
constexpr int AlignmentD  = AlignmentC;

using ElementAccumulator  = float;
using ElementCompute      = float;

// Scale config: per-row for A, block(128) for B along N and K
constexpr int ScaleGranularityM = 1;
constexpr int ScaleGranularityN = 128;
constexpr int ScaleGranularityK = 128;
using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
    ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;

using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

// Tile shapes
using CoopTileShape    = Shape<_128, _128, _128>;
using PingTileShape    = Shape<_64,  _128, _128>;
using ClusterShape_MNK = Shape<_1, _1, _1>;  // GeForce: no multicast

// Epilogue builder (templated on tile shape)
template <class TileShape>
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Mainloop builder (templated on tile shape + schedule)
template <class TileShape, class Schedule>
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue<TileShape>::SharedStorage))>,
    Schedule
>::CollectiveOp;

// GEMM kernel + adapter
template <class TileShape, class Schedule>
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop<TileShape, Schedule>,
    CollectiveEpilogue<TileShape>,
    void>;  // CLC-based tile scheduler

using CooperativeGemm = cutlass::gemm::device::GemmUniversalAdapter<
    GemmKernel<CoopTileShape, cutlass::gemm::KernelScheduleSm120Blockwise>>;

using PingpongGemm = cutlass::gemm::device::GemmUniversalAdapter<
    GemmKernel<PingTileShape, cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120>>;

// Stride types (shared between both GEMM variants)
using StrideA = typename CooperativeGemm::GemmKernel::StrideA;
using StrideB = typename CooperativeGemm::GemmKernel::StrideB;
using StrideC = typename CooperativeGemm::GemmKernel::StrideC;
using StrideD = typename CooperativeGemm::GemmKernel::StrideD;

// Run GEMM with a specific kernel type
template <class Gemm>
static void run_gemm_fp8(
    const void* A, const float* SFA,
    const void* B, const float* SFB,
    half* D,
    int M, int K, int N,
    void* workspace, size_t workspace_size,
    cudaStream_t stream)
{
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));

    auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    auto* D_ptr = reinterpret_cast<ElementD*>(reinterpret_cast<void*>(D));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            static_cast<const ElementA*>(A), stride_A,
            static_cast<const ElementB*>(B), stride_B,
            SFA, layout_SFA,
            SFB, layout_SFB
        },
        {
            {},  // epilogue.thread (alpha/beta)
            D_ptr, stride_D,    // C — must be valid pointer for TMA descriptor (beta=0, not read)
            D_ptr, stride_D
        }
    };

    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;

    Gemm gemm;
    GWEN_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess,
               "CUTLASS FP8 GEMM: problem size not supported");
    GWEN_CHECK(gemm.get_workspace_size(arguments) <= workspace_size,
               "CUTLASS FP8 GEMM: workspace too small");
    GWEN_CHECK(gemm.initialize(arguments, workspace, stream) == cutlass::Status::kSuccess,
               "CUTLASS FP8 GEMM: initialization failed");
    GWEN_CHECK(gemm.run(stream) == cutlass::Status::kSuccess,
               "CUTLASS FP8 GEMM: execution failed");
}

} // anonymous namespace

#endif // CUTLASS_ARCH_MMA_SM120_SUPPORTED


// ============================================================
// Activation Quantization: FP16 → FP8 E4M3 with per-block scaling
// ============================================================

// Quantize a (128 columns × 128 K elements) block of activations from FP16 to FP8.
// One thread block per (N_block, K_block). 256 threads.
// Each thread handles multiple elements within the 128×128 block.
// Computes a single absmax scale for the entire block.
__global__ void __launch_bounds__(256)
kernel_quantize_fp16_to_fp8(
    const half* __restrict__ input,     // [N_real, K] row-major
    uint8_t* __restrict__ output,       // [N_padded, K] row-major
    float* __restrict__ scales,         // [n_n_blocks * n_k_blocks] flat
    int K, int N_padded, int N_real, int n_k_blocks)
{
    const int n_block = blockIdx.x;     // which 128-column block
    const int k_block = blockIdx.y;     // which 128-element K chunk
    const int tid = threadIdx.x;        // 0..255

    const int n_start = n_block * 128;
    const int k_start = k_block * 128;

    // Each thread processes 128*128/256 = 64 elements
    // Thread tid handles elements in a strided pattern within the 128×128 block
    // Layout: iterate over (col, k) pairs within the block

    // Phase 1: Find absmax across entire 128×128 block
    float local_max = 0.0f;

    // 128×128 = 16384 elements, 256 threads → 64 elements per thread
    for (int i = tid; i < 128 * 128; i += 256) {
        int local_col = i / 128;   // which column within block (0..127)
        int local_k = i % 128;     // which K element within block (0..127)
        int col = n_start + local_col;
        int k_idx = k_start + local_k;

        if (col < N_real && k_idx < K) {
            float val = __half2float(input[col * K + k_idx]);
            local_max = fmaxf(local_max, fabsf(val));
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }

    // Cross-warp reduction (8 warps)
    __shared__ float warp_max[8];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_max[warp_id] = local_max;
    __syncthreads();

    float block_max;
    if (tid < 8) {
        block_max = warp_max[tid];
        for (int offset = 4; offset > 0; offset >>= 1) {
            block_max = fmaxf(block_max, __shfl_xor_sync(0xff, block_max, offset));
        }
        if (tid == 0) warp_max[0] = block_max;
    }
    __syncthreads();
    block_max = warp_max[0];

    float scale = (block_max > 0.0f) ? (block_max / 448.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    // Phase 2: Quantize and write FP8 output
    for (int i = tid; i < 128 * 128; i += 256) {
        int local_col = i / 128;
        int local_k = i % 128;
        int col = n_start + local_col;
        int k_idx = k_start + local_k;

        uint8_t fp8_byte = 0;
        if (col < N_real && k_idx < K) {
            float val = __half2float(input[col * K + k_idx]);
            float scaled = val * inv_scale;
            scaled = fmaxf(-448.0f, fminf(448.0f, scaled));
            __nv_fp8_e4m3 fp8_val = __nv_fp8_e4m3(scaled);
            fp8_byte = *reinterpret_cast<uint8_t*>(&fp8_val);
        }
        if (col < N_padded && k_idx < K) {
            output[col * K + k_idx] = fp8_byte;
        }
    }

    // Write scale factor (one per block)
    // CUTLASS expects column-major [N_blocks, K_blocks]: index = k_block * n_n_blocks + n_block
    if (tid == 0) {
        int n_n_blocks = gridDim.x;
        scales[k_block * n_n_blocks + n_block] = scale;
    }
}

// Replicate per-row scales to SFA format (column-major [M, K_blocks]): sfa[kb * M + row] = scale[row]
__global__ void __launch_bounds__(256)
kernel_replicate_row_scales(
    const float* __restrict__ row_scales,   // [M]
    float* __restrict__ sfa,                // [M * n_k_blocks]
    int M, int n_k_blocks)
{
    int row = blockIdx.x;
    if (row >= M) return;

    float scale = row_scales[row];
    // CUTLASS expects column-major [M, K_blocks]: index = kb * M + row
    for (int kb = threadIdx.x; kb < n_k_blocks; kb += blockDim.x) {
        sfa[kb * M + row] = scale;
    }
}


// Transpose BF16 [M, N_stride] RowMajor → FP16 [N, M] RowMajor (= ColumnMajor [M, N])
// Only first N_out columns are read/written (N_stride may be padded).
// Grid: (ceil(N_out, 32), ceil(M, 32))  Block: (32, 8)  — tiled transpose
__global__ void __launch_bounds__(256)
kernel_transpose_bf16_to_fp16(
    const __nv_bfloat16* __restrict__ src,  // [M, N_stride] RowMajor
    half* __restrict__ dst,                 // [N_out, M] RowMajor
    int M, int N_out, int N_stride)
{
    __shared__ float tile[32][33];  // 33 to avoid bank conflicts

    int bx = blockIdx.x * 32;
    int by = blockIdx.y * 32;

    // Load 32×32 tile from src[by..by+31, bx..bx+31]
    for (int i = threadIdx.y; i < 32; i += 8) {
        int row = by + i;
        int col = bx + threadIdx.x;
        if (row < M && col < N_out) {
            tile[i][threadIdx.x] = __bfloat162float(src[row * N_stride + col]);
        } else {
            tile[i][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    // Write transposed: dst[bx..bx+31, by..by+31]
    for (int i = threadIdx.y; i < 32; i += 8) {
        int row = bx + i;         // was column, now row
        int col = by + threadIdx.x;  // was row, now column
        if (row < N_out && col < M) {
            dst[row * M + col] = __float2half(tile[threadIdx.x][i]);
        }
    }
}

// ============================================================
// Public API
// ============================================================

namespace gwen {

void gwen_gemm_fp8(
    const void* W_fp8, const float* W_sfa,
    const void* X_fp8, const float* X_sfb,
    half* Y,
    int out_features, int in_features, int seq_len,
    void* workspace, size_t workspace_size,
    cudaStream_t stream)
{
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    int M = out_features, K = in_features, N = seq_len;
    int N_padded = (N + 127) / 128 * 128;

    // Workspace layout: [BF16 temp output | CUTLASS workspace]
    // BF16 temp: M * N_padded * 2 bytes
    size_t bf16_temp_bytes = (size_t)M * N_padded * sizeof(__nv_bfloat16);
    size_t cutlass_ws_offset = bf16_temp_bytes;

    GWEN_CHECK(workspace_size >= bf16_temp_bytes,
               "FP8 GEMM workspace too small for BF16 temp");

    __nv_bfloat16* bf16_temp = static_cast<__nv_bfloat16*>(workspace);
    void* cutlass_ws = static_cast<uint8_t*>(workspace) + cutlass_ws_offset;
    size_t cutlass_ws_size = workspace_size - cutlass_ws_offset;

    // Run CUTLASS GEMM → BF16 RowMajor [M, N_padded]
    int n_blocks_coop = ((M + 127) / 128) * ((N_padded + 127) / 128);
    if (n_blocks_coop < 140) {
        run_gemm_fp8<PingpongGemm>(W_fp8, W_sfa, X_fp8, X_sfb,
                                    reinterpret_cast<half*>(bf16_temp),
                                    M, K, N_padded, cutlass_ws, cutlass_ws_size, stream);
    } else {
        run_gemm_fp8<CooperativeGemm>(W_fp8, W_sfa, X_fp8, X_sfb,
                                       reinterpret_cast<half*>(bf16_temp),
                                       M, K, N_padded, cutlass_ws, cutlass_ws_size, stream);
    }

    // Transpose BF16 [M, N_padded] RowMajor → FP16 [N, M] RowMajor
    // Only transpose the first N columns (not padding)
    dim3 grid_t((N + 31) / 32, (M + 31) / 32);
    dim3 block_t(32, 8);
    kernel_transpose_bf16_to_fp16<<<grid_t, block_t, 0, stream>>>(
        bf16_temp, Y, M, N, N_padded);
#else
    GWEN_CHECK(false, "FP8 GEMM requires SM120 support (CUDA 12.8+)");
#endif
}

void gwen_quantize_fp16_to_fp8(
    const half* input, void* output_fp8, float* output_sfb,
    int K, int N, cudaStream_t stream)
{
    int n_k_blocks = (K + 127) / 128;
    // Pad N to next multiple of 128 for CUTLASS blockwise scaling requirement
    int N_padded = (N + 127) / 128 * 128;
    int n_n_blocks = N_padded / 128;

    // Grid: one block per (N_block, K_block)
    dim3 grid(n_n_blocks, n_k_blocks);
    kernel_quantize_fp16_to_fp8<<<grid, 256, 0, stream>>>(
        input, static_cast<uint8_t*>(output_fp8), output_sfb,
        K, N_padded, N, n_k_blocks);
}

void gwen_replicate_fp8_scales(
    const float* row_scales, float* sfa, int M, int n_k_blocks,
    cudaStream_t stream)
{
    kernel_replicate_row_scales<<<M, 256, 0, stream>>>(
        row_scales, sfa, M, n_k_blocks);
}

size_t gwen_gemm_fp8_workspace_size(int max_M, int max_K, int max_N) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    int max_N_padded = (max_N + 127) / 128 * 128;

    // Query workspace for the larger (cooperative) GEMM at max problem size
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(max_M, max_K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(max_N_padded, max_K, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(max_M, max_N_padded, 1));
    auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(max_M, max_N_padded, max_K, 1));
    auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(max_M, max_N_padded, max_K, 1));

    typename CooperativeGemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {max_M, max_N_padded, max_K, 1},
        {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
        {{}, nullptr, stride_D, nullptr, stride_D}
    };
    size_t ws_coop = CooperativeGemm::get_workspace_size(args);

    typename PingpongGemm::Arguments args_pp{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {max_M, max_N_padded, max_K, 1},
        {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
        {{}, nullptr, stride_D, nullptr, stride_D}
    };
    size_t ws_pp = PingpongGemm::get_workspace_size(args_pp);

    // Total: BF16 temp output + CUTLASS workspace
    size_t bf16_temp = (size_t)max_M * max_N_padded * sizeof(__nv_bfloat16);
    return bf16_temp + std::max(ws_coop, ws_pp);
#else
    return 0;
#endif
}

} // namespace gwen
