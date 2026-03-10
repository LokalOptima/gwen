# GWEN — Implementation Plan

## Pure CUDA/CUTLASS Qwen3.5-0.8B Inference, Hyper-Optimized for RTX 5070 Ti

---

## Target Hardware Profile: RTX 5070 Ti (SM_120)

| Spec | Value | Optimization Implication |
|------|-------|--------------------------|
| Compute Capability | SM_120 (Blackwell consumer) | Uses `mma.sync`, NOT `wgmma`/`tcgen05` |
| SMs | 70 | Target 280-560+ thread blocks for full occupancy |
| CUDA Cores | 8,960 | |
| Tensor Cores | 280 (5th gen) | FP16, BF16, FP8, FP4, INT8 |
| VRAM | 16 GB GDDR7 | Model is ~500MB Q4_K_M, plenty of room for KV cache |
| Memory Bandwidth | 896 GB/s | Decode is bandwidth-bound; every byte saved = speed |
| L2 Cache | 48 MB | Can cache ~10% of model weights! Exploit this. |
| L1/Shared per SM | 128 KB (configurable) | Up to 99 KB shared memory per block |
| Max Warps/SM | 48 | Aim for ~42 regs/thread at full occupancy |
| TMA | Available (no multicast) | Use for async global→shared copies |
| Clusters | 1x1x1 only | No distributed shared memory tricks |
| FP4 Tensor Core | Yes (NVFP4/MXFP4) | Consider FP4 for compute-bound ops |

### Key Insight: Decode is Memory-Bandwidth-Bound

For single-token decode (the dominant use case), the bottleneck is loading weights from VRAM, not computing. With Q4_K_M weights (~500MB) and 896 GB/s bandwidth, theoretical minimum latency per full forward pass is ~0.56ms. Every optimization should focus on:
1. **Reducing bytes loaded** (quantization, weight layout)
2. **Maximizing bandwidth utilization** (coalesced access, async copy)
3. **Fusing operations** (fewer kernel launches, less synchronization)
4. **Exploiting L2 cache** (48MB can hold ~10% of weights hot)

---

## Architecture: Qwen3.5-0.8B Hybrid DeltaNet+Transformer

```
Input tokens
    │
    ▼
┌──────────────┐
│ Token Embed   │  vocab=248320, dim=1024, Q6_K
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  × 6 repetitions of:                                      │
│                                                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────┐ │
│  │ DeltaNet Layer   │→│ DeltaNet Layer   │→│ DeltaNet   │ │
│  │ (linear attn)    │  │ (linear attn)    │  │ Layer      │ │
│  └─────────────────┘  └─────────────────┘  └─────┬─────┘ │
│                                                    │       │
│                                              ┌─────▼─────┐ │
│                                              │ Full Attn  │ │
│                                              │ Layer (GQA)│ │
│                                              └───────────┘ │
│                                                            │
│  Each layer also has: RMSNorm → [Attn] → Residual →       │
│                       RMSNorm → SwiGLU FFN → Residual     │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│ RMSNorm       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ LM Head       │  (tied to token_embd, dim=1024 → vocab=248320)
└──────────────┘
```

### DeltaNet Layer (18 layers: 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22)

```
Input (1024)
  │
  ├─ RMSNorm
  │
  ├─ Joint QKV+Z projection: [1024] → [6144] (Q5_K)
  │    Split: QKV=[4096], Z/gate=[2048]
  │
  ├─ 1D Conv (kernel=4) over QKV with state caching
  │
  ├─ SiLU activation
  │
  ├─ Split QKV → Q[2048], K[2048], V[2048]
  │    (16 heads × 128 dim each for Q,K,V)
  │
  ├─ L2-normalize Q and K
  │
  ├─ Gated DeltaNet recurrence:
  │    - Compute α (alpha) and β (beta) from input
  │    - A_log → decay parameter
  │    - dt_bias → timestep
  │    - Recurrent state update: S_t = decay * S_{t-1} + β_t ⊗ v_t * k_t^T
  │    - Output: o_t = S_t * q_t
  │
  ├─ Gated RMSNorm: norm(output) * SiLU(z)
  │
  ├─ Output projection: [2048] → [1024] (Q5_K)
  │
  ├─ Residual add
  │
  ├─ Post-attention RMSNorm
  │
  ├─ SwiGLU FFN:
  │    gate = W_gate @ x    [1024→3584] (Q4_K)
  │    up   = W_up @ x      [1024→3584] (Q4_K)
  │    out  = SiLU(gate) * up
  │    down = W_down @ out   [3584→1024] (Q4_K/Q6_K)
  │
  └─ Residual add
```

### Full Attention Layer (6 layers: 3,7,11,15,19,23)

```
Input (1024)
  │
  ├─ RMSNorm
  │
  ├─ Q projection: [1024] → [2048] (8 heads × 256 dim) (Q4_K)
  ├─ K projection: [1024] → [512]  (2 heads × 256 dim) (Q4_K)
  ├─ V projection: [1024] → [512]  (2 heads × 256 dim) (Q4_K/Q6_K)
  │
  ├─ Per-head Q RMSNorm, K RMSNorm
  │
  ├─ imRoPE (interleaved multi-axis RoPE):
  │    - sections = [11, 11, 10, 0] over 64 dims (partial_rotary_factor=0.25)
  │    - Only first 64 of 256 dims get rotary encoding
  │    - Interleaved across 3 axes (temporal, height, width)
  │
  ├─ GQA attention (8Q:2KV, ratio 4:1):
  │    - KV cache store
  │    - Q @ K^T / sqrt(256)
  │    - Causal mask + softmax
  │    - Attn_weights @ V
  │
  ├─ Gated attention: output * sigmoid(gate_from_Q)
  │
  ├─ Output projection: [2048] → [1024] (Q4_K)
  │
  ├─ Residual add
  │
  ├─ Post-attention RMSNorm
  │
  ├─ SwiGLU FFN (same as DeltaNet layer)
  │
  └─ Residual add
```

---

## Implementation Phases

### Phase 0: Infrastructure & Tooling (Week 1)
**Goal**: Build system, GGUF loader, test harness, benchmark framework

#### 0.1 Build System
- CMake with CUDA support, targeting SM_120
- CUTLASS 4.x as git submodule
- Compiler flags: `-arch=sm_120 -O3 --use_fast_math` (verify correctness with/without fast_math)
- Set up `__launch_bounds__` macros for occupancy control

#### 0.2 GGUF Loader
- Parse GGUF file format (header, metadata KV pairs, tensor info, tensor data)
- Memory-map the file for zero-copy weight access
- Build tensor registry: name → {data_ptr, shape, quantization_type}
- Implement Q4_K, Q5_K, Q6_K, Q8_0 dequantization (CPU, for verification)
- **Test**: dump tensor names/shapes, compare against `gguf-dump` output from llama.cpp

#### 0.3 Test Harness
- Python script (`tests/compare_outputs.py`) that:
  1. Runs llama.cpp inference with a set of test prompts, captures logits (via server API or `--logits-all`)
  2. Runs gwen inference with the same prompts
  3. Compares token-by-token: exact match for greedy, KL-divergence for distribution
  4. Reports per-layer intermediate values if debug mode is enabled
- Test prompts: short (1 token), medium (32 tokens), long (512 tokens)
- Golden reference: save llama.cpp outputs to disk for reproducible comparison

#### 0.4 Benchmark Framework
- **Micro-bench** (`bench/micro_bench.cu`): individual kernel timing
  - RMSNorm, SwiGLU FFN, attention, DeltaNet scan, RoPE, dequant GEMM
  - Report: TFLOPS, GB/s achieved, % of theoretical peak
  - Nsight Compute integration for roofline analysis
- **Macro-bench** (`bench/macro_bench.py`):
  - Time-to-first-token (TTFT) for various prompt lengths
  - Tokens-per-second for decode
  - Memory usage (peak VRAM)
  - Compare against llama.cpp with identical settings
  - Latency breakdown by layer type (DeltaNet vs Full Attention vs FFN)

#### 0.5 Blog Post #0: "Starting GWEN"
- Motivation, architecture overview, why Qwen3.5 is interesting (hybrid architecture)
- GPU target and what makes RTX 5070 Ti unique for this workload
- What we expect: where the bottlenecks will be

---

### Phase 1: Core Kernels (Week 2-3)
**Goal**: Implement and verify all individual compute kernels in isolation

#### 1.1 Memory Management
- GPU memory allocator (arena-based for inference, no fragmentation)
- Weight upload: mmap GGUF → pinned host → device (async)
- KV cache pre-allocation (full attention layers only, 6 layers × 2 heads × 256 dim)
- DeltaNet recurrent state allocation (18 layers × 16 heads × 128 × 128)

#### 1.2 Dequantization Kernels
- Q4_K dequant kernel: packed block → FP16
  - Super-block: d(fp16) + dmin(fp16) + scales(12 bytes) + data(128 bytes)
  - 8 sub-blocks of 32 values each, hierarchical scale reconstruction
  - **Optimization**: warp-cooperative dequant, one warp per super-block
- Q5_K, Q6_K, Q8_0 dequant kernels (similar structure)
- **Test**: dequant on GPU, download, compare against CPU reference (bit-exact for FP16)
- **Optimization target**: achieve >80% of memory bandwidth (>717 GB/s)

#### 1.3 Quantized GEMV (the critical kernel for decode)
- For batch=1 decode, GEMM becomes GEMV (matrix-vector multiply)
- Fuse dequantization INTO the GEMV kernel:
  1. Each warp handles one row (or a few rows) of the weight matrix
  2. Load Q4_K block, reconstruct scale+min, dequant to FP16 in registers
  3. FMA with input activation vector (loaded to shared memory)
  4. Warp reduction for dot product
- **RTX 5070 Ti specific**: exploit 48MB L2 for activation vector caching
- CUTLASS may not help here — GEMV is too bandwidth-bound; hand-written kernel likely wins
- **Test**: compare output against CPU dequant + cuBLAS FP16 GEMV
- **Target**: >85% bandwidth utilization (~760 GB/s)

#### 1.4 Quantized GEMM (for prefill / batched decode)
- For M>1 (prefill or batched), use CUTLASS-based GEMM
- Strategy: dequantize Q4_K to FP16 on-the-fly during tile loading, then FP16 tensor core GEMM
- Use CUTLASS 4.x SM_120 code path (`mma.sync` based)
- Tile sizes to try: 128×128×32, 64×128×64, 128×64×64
- **Alternative**: convert weights at load time to a simpler INT4+scale format that maps directly to CUTLASS mixed-input GEMM
- **Test**: compare against cuBLAS FP16 (with pre-dequantized weights)

#### 1.5 RMSNorm Kernel
- `output = x * rsqrt(mean(x²) + eps) * weight`
- Small kernel (dim=1024), bandwidth-bound
- One warp per vector, warp shuffle for reduction
- Fuse with subsequent operation where possible (e.g., RMSNorm + GEMV input broadcast)
- **Test**: compare against PyTorch/numpy reference

#### 1.6 SiLU / SwiGLU Kernel
- SiLU: `x * sigmoid(x)`
- SwiGLU: `SiLU(gate) * up` — element-wise, fuse into FFN down projection GEMV
- Fused kernel: load gate+up results, compute SwiGLU, store for down projection
- Or better: fuse as epilogue of gate/up GEMV
- **Test**: numerical comparison

#### 1.7 Softmax Kernel
- Online softmax (numerically stable, single pass)
- For attention: causal masked softmax over sequence dimension
- Head dim 256 means scores are over seq_len, which varies
- Warp-cooperative for short sequences, multi-warp for longer
- **Test**: compare against cuDNN softmax

#### 1.8 RoPE Kernel (imRoPE variant)
- Interleaved multi-axis RoPE with sections [11, 11, 10, 0]
- Only applies to first 64 of 256 dims in head (partial_rotary_factor=0.25)
- Interleaved means: dim 0 goes to axis 0, dim 1 to axis 1, dim 2 to axis 2, dim 3 to axis 0, ...
- The position IDs come from 3 separate axes (for text, all 3 axes use the same position)
- **Test**: compare against llama.cpp rope.cu output

#### 1.9 Blog Post #1: "Building the Kernel Arsenal"
- Each kernel's design decisions
- Bandwidth analysis and roofline model for RTX 5070 Ti
- Micro-benchmark results table
- Where we're leaving performance on the table and why

---

### Phase 2: DeltaNet Linear Attention (Week 3-4)
**Goal**: Implement the Gated DeltaNet recurrence, the novel component

#### 2.1 Understanding DeltaNet
- DeltaNet maintains a recurrent state S ∈ R^{d_k × d_v} per head
- Update rule: S_t = diag(α_t) · S_{t-1} + β_t · v_t · k_t^T
  - α_t: decay factor (from A_log + dt, sigmoid-activated)
  - β_t: input gate
  - k_t, v_t: key and value vectors
  - q_t: query vector, output = S_t · q_t
- This is a linear recurrence — no softmax, no KV cache growing with sequence length
- For prefill: can be computed via "chunkwise" parallel scan
- For decode: simple sequential state update (very fast, O(d_k × d_v) per step)

#### 2.2 DeltaNet Decode Kernel (single token)
- Per head: update state matrix S (128×128) and compute output
- S_new = diag(α) · S + β · outer(v, k)
- o = S_new · q
- **Optimization**: S is 128×128 FP16 = 32KB per head, 16 heads = 512KB per layer
  - Fits in shared memory! Load S, update in-place, store back
  - Or keep S in registers across warps (128×128 / 32 threads per warp = complex tiling)
- α is a scalar per head (from A_log), so diag(α)·S = α·S (scalar multiply)
- Fuse: state_update + query into single kernel
- **Test**: compare state and output against llama.cpp's delta_net autoregressive

#### 2.3 DeltaNet Prefill Kernel (chunked parallel scan)
- For processing multiple tokens at once (prefill phase)
- Chunk the sequence, compute intra-chunk attention, then scan across chunks
- llama.cpp uses `build_delta_net_chunking` — study this path
- This is more complex but needed for fast prefill
- **Can defer to Phase 4 optimization** — start with sequential scan for correctness

#### 2.4 1D Convolution with State
- Conv1d with kernel_size=4 over the QKV features
- State: last 3 values cached for autoregressive generation
- Simple sliding window convolution
- Fuse with subsequent SiLU activation
- **Test**: compare against PyTorch nn.Conv1d

#### 2.5 L2 Normalization
- Normalize Q and K vectors to unit length before DeltaNet attention
- `x / ||x||_2` — simple but must be numerically stable
- Fuse into DeltaNet input preparation kernel

#### 2.6 Gated RMSNorm
- `output = RMSNorm(attn_output) * SiLU(gate_z)`
- Fuse: RMSNorm + SiLU + elementwise multiply
- gate_z comes from the Z portion of the joint QKV+Z projection

#### 2.7 Full DeltaNet Layer Integration
- Wire all components: norm → proj → conv → silu → split → l2norm → deltanet → gated_norm → out_proj → residual → norm → ffn → residual
- Single-token forward pass through one DeltaNet layer
- **Test**: compare intermediate values at each stage against llama.cpp
- **Bench**: full DeltaNet layer latency

#### 2.8 Blog Post #2: "Taming DeltaNet on the GPU"
- What DeltaNet is and why it's interesting (O(1) memory per step vs O(n) for attention)
- The recurrent state management challenge
- Kernel fusion decisions
- Comparison: DeltaNet layer vs full attention layer performance
- Bugs encountered with numerical precision in the recurrence

---

### Phase 3: Full Attention + Complete Model (Week 4-5)
**Goal**: Implement GQA attention, wire everything together

#### 3.1 GQA Attention (decode)
- 8 query heads, 2 KV heads → each KV head serves 4 query heads
- Head dim = 256 (large! affects register pressure)
- Steps: Q/K/V projections → QK RMSNorm → imRoPE → KV cache → attention → gate → output
- KV cache: pre-allocated [max_seq_len × 2_heads × 256_dim] per layer (6 layers)
- At max 4K context: 6 layers × 4K × 2 × 256 × 2 bytes = 24MB (fits easily)
- **Decode attention kernel**: for each query head, load KV cache, compute attention
  - Split-K over sequence length for parallelism
  - Warp-cooperative: each warp handles part of the sequence
  - Online softmax to avoid materializing full attention matrix

#### 3.2 GQA Attention (prefill / Flash Attention)
- For prefill, use a FlashAttention-style kernel
- Tiling over sequence length, online softmax
- Head dim 256 is supported by FlashAttention but less common — may need custom tuning
- Consider using CUTLASS's built-in attention support if available for SM_120
- **Alternative**: use cuDNN's fused attention if available in CUDA 13.1
- **Can defer optimization** — start with naive attention for correctness

#### 3.3 Attention Output Gating
- `output = attn_result * sigmoid(gate)`
- gate is computed from Q projection (extra output)
- Fuse with attention output

#### 3.4 Full Attention Layer Integration
- Wire: norm → Q/K/V proj → QK norm → RoPE → GQA attn → gate → out_proj → residual → norm → ffn → residual
- **Test**: compare against llama.cpp full attention layer output

#### 3.5 Complete Model Assembly
- Embedding lookup (with Q6_K dequantization)
- 24 layers in sequence (checking layer type: DeltaNet or FullAttn)
- Final RMSNorm
- LM head (tied weights = transpose of embedding, Q6_K)
- **Test**: full forward pass comparison, greedy decode of 100 tokens

#### 3.6 Sampling & Generation Loop
- Temperature, top-k, top-p sampling
- Greedy (argmax) for deterministic testing
- Token-by-token generation loop with proper state management

#### 3.7 Blog Post #3: "Full Model Assembly and First Words"
- Putting it all together
- The GQA attention implementation
- First successful generation — what it felt like
- Output comparison against llama.cpp
- Initial performance numbers

---

### Phase 4: RTX 5070 Ti Hyper-Optimization (Week 5-7)
**Goal**: Squeeze every last drop of performance from SM_120

#### 4.1 Profiling & Bottleneck Analysis
- Nsight Compute profiling of full forward pass
- Identify: compute-bound vs memory-bound per kernel
- Roofline analysis per kernel on RTX 5070 Ti
- L2 cache hit rate analysis — is the 48MB helping?

#### 4.2 Kernel Fusion Pass
Based on profiling, fuse kernels to reduce launch overhead and memory traffic:
- **RMSNorm + GEMV input**: norm the input, then immediately use as GEMV operand
- **GEMV + bias + activation**: fuse dequant-GEMV with SiLU or sigmoid epilogue
- **SwiGLU FFN**: fuse gate_GEMV and up_GEMV into single kernel (or at least launch concurrently)
- **DeltaNet full pipeline**: fuse conv → silu → split → l2norm → state_update → gated_norm
- **Graph-level**: fuse entire layer into minimal kernel launches (target: 3-5 kernels per layer)

#### 4.3 Memory Access Pattern Optimization
- **Weight layout**: reorder Q4_K blocks to match warp access patterns
  - Group sub-blocks that are accessed by the same warp contiguously
  - Consider converting to a custom "GWEN Q4" format at load time
- **Activation buffer**: double-buffer between layers (only need 2 × 1024 × sizeof(half))
- **KV cache**: optimize layout for sequential access during attention decode
- **DeltaNet state**: optimize layout for register-level access during updates

#### 4.4 L2 Cache Exploitation
- The 48MB L2 can cache significant portions of the model
- Strategy: process layers in an order that maximizes L2 reuse
- Keep frequently-accessed small tensors (norms, biases, A_log, dt_bias) permanently in L2
- Consider: persistent kernel that processes multiple layers without returning to global

#### 4.5 Tensor Core Utilization
- For prefill GEMM: ensure we're using FP16 `mma.sync` instructions
- Experiment with FP8 accumulation for non-critical paths
- Consider FP4 for weights if accuracy permits (needs testing!)
- Profile tensor core utilization % and optimize tile sizes

#### 4.6 Occupancy Tuning
- Per-kernel `__launch_bounds__` with tuned max threads and min blocks
- Register spilling analysis — are we spilling to local memory?
- Shared memory vs registers tradeoff per kernel
- Consider persistent kernels for decode (all 70 SMs doing one layer at a time)

#### 4.7 Async and Pipelining
- Use `cp.async` for global→shared memory loads (available on SM_120)
- Double-buffer shared memory for next tile while computing current
- Overlap weight loading with computation
- Stream-based overlap where possible (e.g., DeltaNet state store + FFN start)

#### 4.8 Custom Weight Format
- Design a "GWEN format" optimized for RTX 5070 Ti access patterns:
  - Interleaved scales and data for single-pass dequant+GEMV
  - Aligned to 128-byte cache lines
  - Sub-block ordering matches warp lane assignments
- One-time conversion from GGUF Q4_K at model load (takes milliseconds)

#### 4.9 Blog Post #4: "Squeezing the 5070 Ti"
- Nsight Compute roofline plots
- Before/after numbers for each optimization
- L2 cache analysis
- Custom weight format design
- How close we got to theoretical limits

---

### Phase 5: Prefill Optimization (Week 7-8)
**Goal**: Fast prompt processing

#### 5.1 Chunked DeltaNet Parallel Scan
- Implement the parallel chunkwise algorithm for DeltaNet prefill
- Allows processing all prompt tokens in parallel within chunks
- Inter-chunk: parallel prefix scan over chunk states
- This is the key to fast prefill for the 18 DeltaNet layers

#### 5.2 Flash Attention for Prefill
- Implement or integrate FlashAttention for the 6 full attention layers
- Head dim 256 needs careful tiling (larger tiles, more shared memory)
- causal mask handling

#### 5.3 Batched GEMM for Prefill
- Switch from GEMV to GEMM path when processing multiple tokens
- Use CUTLASS SM_120 GEMM kernels with dequantization
- Tile size tuning for various prompt lengths

#### 5.4 Blog Post #5: "Prefill: Going Parallel"
- Chunked DeltaNet algorithm explanation with diagrams
- Flash Attention implementation for large head dims
- TTFT benchmarks at various prompt lengths

---

### Phase 6: Polish & Final Benchmarks (Week 8-9)

#### 6.1 Comprehensive Correctness Suite
- Test across 50+ diverse prompts (multilingual, code, math, long context)
- KL-divergence histogram across positions
- Perplexity comparison on a standard dataset (wikitext)
- Edge cases: empty prompt, single token, max context length

#### 6.2 Final Benchmark Suite
**Micro-level** (per-kernel):
| Kernel | Input Size | Metric | vs llama.cpp |
|--------|-----------|--------|--------------|
| RMSNorm | 1×1024 | GB/s | ? |
| Q4_K GEMV | 1×1024 → 1×3584 | GB/s | ? |
| SwiGLU | 1×3584 | GB/s | ? |
| DeltaNet decode | 16 heads, 128×128 state | μs/step | ? |
| GQA attention decode | 8Q:2KV, seqlen=512 | μs/step | ? |
| imRoPE | 8 heads × 256 dim | GB/s | ? |
| Softmax | seqlen=512, 8 heads | GB/s | ? |

**Macro-level**:
| Metric | Prompt Length | gwen | llama.cpp | Speedup |
|--------|-------------|------|-----------|---------|
| TTFT | 1 token | ? | ? | ? |
| TTFT | 128 tokens | ? | ? | ? |
| TTFT | 512 tokens | ? | ? | ? |
| TTFT | 2048 tokens | ? | ? | ? |
| Decode tok/s | steady state | ? | ? | ? |
| Peak VRAM | max context | ? | ? | ? |

**Latency breakdown**:
- Per-layer-type time (DeltaNet vs FullAttn vs FFN)
- Kernel launch overhead
- Memory allocation overhead
- Weight loading time

#### 6.3 Blog Post #6: "Final Results and Lessons Learned"
- Complete benchmark tables
- Architecture retrospective
- What worked, what didn't
- Comparison with other inference engines
- Future directions (multi-batch, longer context, other models)

---

## File Structure

```
gwen/
├── CLAUDE.md                    # Project instructions for AI assistant
├── PLAN.md                      # This file
├── CMakeLists.txt               # Build system
├── Qwen3.5-0.8B-Q4_K_M.gguf    # Model weights
│
├── include/
│   ├── gwen/
│   │   ├── common.h             # Error checking, types, macros
│   │   ├── gguf.h               # GGUF file parser
│   │   ├── model.h              # Model definition (layers, config)
│   │   ├── kernels.h            # Kernel launch wrappers
│   │   ├── memory.h             # GPU memory management
│   │   ├── kv_cache.h           # KV cache for full attention
│   │   ├── deltanet_state.h     # Recurrent state for DeltaNet
│   │   └── sampling.h           # Token sampling
│   └── ...
│
├── src/
│   ├── main.cu                  # Entry point, generation loop
│   ├── gguf.cu                  # GGUF parser implementation
│   ├── model.cu                 # Model loading and forward pass
│   ├── kernels/
│   │   ├── dequant.cu           # Q4_K, Q5_K, Q6_K, Q8_0 dequantization
│   │   ├── gemv.cu              # Fused dequant+GEMV for decode
│   │   ├── gemm.cu              # CUTLASS GEMM for prefill
│   │   ├── rmsnorm.cu           # RMSNorm
│   │   ├── rope.cu              # imRoPE
│   │   ├── attention.cu         # GQA attention (decode + prefill)
│   │   ├── deltanet.cu          # DeltaNet recurrence (decode + prefill)
│   │   ├── conv1d.cu            # 1D convolution with state
│   │   ├── activation.cu        # SiLU, SwiGLU, sigmoid
│   │   └── softmax.cu           # Online softmax
│   └── memory.cu                # GPU allocator
│
├── tests/
│   ├── compare_outputs.py       # End-to-end comparison vs llama.cpp
│   ├── test_gguf_loader.py      # GGUF parsing tests
│   ├── test_kernels.cu          # Per-kernel unit tests
│   └── test_layers.cu           # Per-layer integration tests
│
├── bench/
│   ├── micro_bench.cu           # Per-kernel benchmarks
│   ├── macro_bench.py           # End-to-end benchmarks vs llama.cpp
│   └── results/                 # Saved benchmark results (JSON)
│
├── blog/
│   ├── README.md                # Blog index
│   ├── 00-starting-gwen.md
│   ├── 01-kernel-arsenal.md
│   ├── 02-taming-deltanet.md
│   ├── 03-first-words.md
│   ├── 04-squeezing-5070ti.md
│   ├── 05-prefill-going-parallel.md
│   └── 06-final-results.md
│
├── scripts/
│   ├── setup_deps.sh            # Install CUTLASS, build llama.cpp
│   ├── convert_weights.py       # GGUF Q4_K → GWEN format conversion
│   └── dump_gguf.py             # Debug: dump GGUF metadata
│
└── third_party/
    ├── cutlass/                  # CUTLASS git submodule
    └── llama.cpp/                # llama.cpp for reference (git submodule)
```

---

## Dependencies & Setup

```bash
# 1. Clone CUTLASS
cd third_party
git clone --depth 1 --branch v4.4.1 https://github.com/NVIDIA/cutlass.git

# 2. Build llama.cpp for reference comparison
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)

# 3. Python deps (for tests/bench)
uv pip install gguf numpy torch matplotlib

# 4. Build gwen
cd /home/lapo/git/gwen
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j$(nproc)
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| DeltaNet numerical instability | High — wrong outputs | Use FP32 accumulation for state, L2 norm, compare per-layer |
| Q4_K_M dequant precision loss | Medium — slightly wrong logits | Compare against FP16 reference, accept small KL-div |
| CUTLASS SM_120 support gaps | Medium — can't use some features | Fall back to hand-written mma.sync or cuBLAS |
| Head dim 256 too large for shared mem | Medium — poor attention perf | Split computation, use registers, or Flash Attention |
| imRoPE implementation bugs | High — completely wrong outputs | Test RoPE in isolation against llama.cpp rope.cu |
| DeltaNet prefill chunking complexity | Medium — slow prefill | Start with sequential, optimize later |
| Large vocab (248K) LM head bottleneck | Low — slow sampling | Only compute top-k logits, or split across SMs |

---

## Success Criteria

1. **Correctness**: Greedy decode matches llama.cpp token-for-token on 20+ test prompts
2. **Performance**: ≥1.5× decode tok/s vs llama.cpp on RTX 5070 Ti (ambitious but achievable with fusion)
3. **Documentation**: 6 blog posts covering the full journey
4. **Code quality**: Clean, readable CUDA code that others can learn from
