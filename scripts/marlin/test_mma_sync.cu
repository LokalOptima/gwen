// Minimal mma.sync.m16n8k16 validation on SM_120 (RTX 5070 Ti)
// Compile: nvcc -arch=sm_120 -O3 -o test_mma_sync scratch/test_mma_sync.cu
// Run: ./test_mma_sync

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// A: 16x16 (row-major), each thread holds 8 FP16 values in 4 registers
// B: 16x8  (col-major), each thread holds 4 FP16 values in 2 registers
// C/D: 16x8, each thread holds 4 FP32 values in 4 registers
__global__ void kernel_test_mma(const half* __restrict__ A,
                                 const half* __restrict__ B,
                                 float* __restrict__ D) {
    // Thread layout within a warp (32 threads)
    int lane = threadIdx.x % 32;

    // Load A fragments: each thread loads 8 half values (4 registers of half2)
    // Thread mapping for A (16x16, row-major):
    //   Threads 0-3:   row 0, threads 4-7:  row 1, ...
    //   Within each group of 4, thread t loads columns [t*2, t*2+1] and [t*2+8, t*2+9]
    //   Repeated for rows 8-15

    uint32_t a_frag[4]; // 4 registers, each holding 2 FP16 values
    int a_row_base = (lane / 4);           // 0..7
    int a_col_base = (lane % 4) * 2;       // 0,2,4,6

    // Row group 0 (rows 0-7)
    int idx0 = a_row_base * 16 + a_col_base;
    int idx1 = a_row_base * 16 + a_col_base + 8;
    a_frag[0] = *reinterpret_cast<const uint32_t*>(&A[idx0]);
    a_frag[1] = *reinterpret_cast<const uint32_t*>(&A[idx1]);

    // Row group 1 (rows 8-15)
    int idx2 = (a_row_base + 8) * 16 + a_col_base;
    int idx3 = (a_row_base + 8) * 16 + a_col_base + 8;
    a_frag[2] = *reinterpret_cast<const uint32_t*>(&A[idx2]);
    a_frag[3] = *reinterpret_cast<const uint32_t*>(&A[idx3]);

    // Load B fragments: each thread loads 4 half values (2 registers of half2)
    // B is 16x8, col-major. Thread mapping:
    //   row = (lane % 4) * 2 + {0,1} for first pair
    //   col = lane / 4
    int b_row_base = (lane % 4) * 2;
    int b_col = (lane / 4);
    // B is col-major: B[row, col] = B[col * 16 + row]
    // But for mma.sync we need specific layout:
    // Each thread t loads B[t%4*2 : t%4*2+1, t/4] for K-dim positions
    uint32_t b_frag[2];
    int bidx0 = b_row_base * 8 + b_col;       // K-rows 0..7
    int bidx1 = (b_row_base + 8) * 8 + b_col; // K-rows 8..15
    b_frag[0] = *reinterpret_cast<const uint32_t*>(&B[bidx0]);
    b_frag[1] = *reinterpret_cast<const uint32_t*>(&B[bidx1]);

    // Wait, the B layout for mma is more nuanced. Let me use the correct fragment layout.
    // Actually, let's just use the PTX directly and load from shared memory with ldmatrix.

    // For simplicity, load A and B into shared memory and use ldmatrix
    __shared__ half A_smem[16 * 16]; // A: 16x16
    __shared__ half B_smem[16 * 8];  // B: 16x8

    // Cooperative load into shared memory
    for (int i = lane; i < 256; i += 32)
        A_smem[i] = A[i];
    for (int i = lane; i < 128; i += 32)
        B_smem[i] = B[i];
    __syncwarp();

    // Use ldmatrix to load fragments
    // A: load 4 x m8n8 matrices (8x8 each) to fill the 16x16 A operand
    uint32_t a_reg[4];
    {
        // Each thread loads from a specific row of shared memory
        // ldmatrix loads one 8x8 matrix fragment per .x count
        // For .x4, we load 4 matrices worth = 16 bytes per thread
        int row = lane % 16;
        int off = (lane / 16) * 8;
        uint32_t smem_addr = __cvta_generic_to_shared(&A_smem[row * 16 + off]);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_reg[0]), "=r"(a_reg[1]), "=r"(a_reg[2]), "=r"(a_reg[3])
            : "r"(smem_addr)
        );
    }

    // B: load 2 x m8n8 matrices for the 16x8 B operand
    uint32_t b_reg[2];
    {
        int row = lane % 16;
        uint32_t smem_addr = __cvta_generic_to_shared(&B_smem[row * 8]);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b_reg[0]), "=r"(b_reg[1])
            : "r"(smem_addr)
        );
    }

    // Initialize accumulator to zero
    float d_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Execute mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n"
        "    {%0, %1, %2, %3},\n"
        "    {%4, %5, %6, %7},\n"
        "    {%8, %9},\n"
        "    {%10, %11, %12, %13};\n"
        : "=f"(d_reg[0]), "=f"(d_reg[1]), "=f"(d_reg[2]), "=f"(d_reg[3])
        : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
          "r"(b_reg[0]), "r"(b_reg[1]),
          "f"(d_reg[0]), "f"(d_reg[1]), "f"(d_reg[2]), "f"(d_reg[3])
    );

    // Each thread writes its portion of the 16x8 output matrix
    // Thread t owns: D[t/4, t%4*2] and D[t/4, t%4*2+1] for rows 0-7
    //                D[t/4+8, t%4*2] and D[t/4+8, t%4*2+1] for rows 8-15
    int d_row0 = lane / 4;
    int d_col0 = (lane % 4) * 2;
    D[d_row0 * 8 + d_col0]       = d_reg[0];
    D[d_row0 * 8 + d_col0 + 1]   = d_reg[1];
    D[(d_row0 + 8) * 8 + d_col0]     = d_reg[2];
    D[(d_row0 + 8) * 8 + d_col0 + 1] = d_reg[3];
}

int main() {
    printf("=== mma.sync.m16n8k16 validation on SM_120 ===\n\n");

    // Create test matrices
    // A: 16x16, B: 16x8, D = A * B should be 16x8
    half h_A[16 * 16];
    half h_B[16 * 8];
    float h_D[16 * 8];
    float h_D_ref[16 * 8];

    // Fill with known values
    // A[i][j] = (i + 1) * 0.1, B[i][j] = (j + 1) * 0.1
    // This makes D[i][j] = sum_k A[i][k] * B[k][j] = 0.01 * (i+1) * (j+1) * sum_k(1) = 0.01 * (i+1) * (j+1) * 16
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
            h_A[i * 16 + j] = __float2half(1.0f);  // All ones for simplicity

    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 8; j++)
            h_B[i * 8 + j] = __float2half((float)(j + 1));  // Column j = j+1

    // Reference: D = A * B
    // A is all 1s, B has column j = j+1
    // So D[i][j] = sum over k=0..15 of 1.0 * (j+1) = 16 * (j+1)
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 8; j++)
            h_D_ref[i * 8 + j] = 16.0f * (j + 1);

    // Allocate and copy to device
    half *d_A, *d_B;
    float *d_D;
    CHECK_CUDA(cudaMalloc(&d_A, sizeof(h_A)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(h_B)));
    CHECK_CUDA(cudaMalloc(&d_D, sizeof(h_D)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_D, 0, sizeof(h_D)));

    // Launch with 1 warp
    kernel_test_mma<<<1, 32>>>(d_A, d_B, d_D);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_D, d_D, sizeof(h_D), cudaMemcpyDeviceToHost));

    // Verify
    int errors = 0;
    float max_err = 0.0f;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            float got = h_D[i * 8 + j];
            float ref = h_D_ref[i * 8 + j];
            float err = fabsf(got - ref);
            if (err > max_err) max_err = err;
            if (err > 0.1f) {
                if (errors < 10)
                    printf("  MISMATCH D[%d][%d]: got %.4f, expected %.4f (err=%.4f)\n",
                           i, j, got, ref, err);
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("PASS: All 128 elements match (max_err=%.6f)\n", max_err);
    } else {
        printf("FAIL: %d mismatches (max_err=%.4f)\n", errors, max_err);
    }

    // Print first few rows for inspection
    printf("\nD (first 4 rows):\n");
    for (int i = 0; i < 4; i++) {
        printf("  row %2d: ", i);
        for (int j = 0; j < 8; j++)
            printf("%8.2f", h_D[i * 8 + j]);
        printf("\n");
    }

    printf("\nExpected (first 4 rows):\n");
    for (int i = 0; i < 4; i++) {
        printf("  row %2d: ", i);
        for (int j = 0; j < 8; j++)
            printf("%8.2f", h_D_ref[i * 8 + j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);

    return errors > 0 ? 1 : 0;
}
