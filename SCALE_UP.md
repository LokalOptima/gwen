# GWEN Scale-Up Plan: Qwen3.5-4B / 9B + SM_120 Optimization Experiments

## Goal

Port GWEN to Qwen3.5-4B (primary) and 9B (stretch), maximizing kernel reuse from the 0.8B codebase. Then run a series of SM_120-specific optimization experiments (FP8 GEMV, L2 persistence, vectorized loads) that were never tried on the 0.8B.

---

## Target Models: Confirmed Architecture Specs

| Parameter | 0.8B (current) | 4B | 9B |
|---|---|---|---|
| `hidden_size` | 1024 | **2560** | **4096** |
| `num_layers` | 24 | **32** | **32** |
| `intermediate_size` | 3584 | **9216** | **12288** |
| `vocab_size` | 248320 | 248320 | 248320 |
| `n_head` (Q, full attn) | 8 | **16** | **16** |
| `n_head_kv` (KV, full attn) | 2 | **4** | **4** |
| `head_dim` | 256 | 256 | 256 |
| `linear_key_head_dim` (dk) | 128 | 128 | 128 |
| `linear_value_head_dim` (dv) | 128 | 128 | 128 |
| `linear_num_key_heads` | 16 | 16 | 16 |
| `linear_num_value_heads` | 16 | **32** | **32** |
| `ssm_inner_size` (v_heads*dv) | 2048 | **4096** | **4096** |
| `ssm_conv_kernel` | 4 | 4 | 4 |
| `full_attn_interval` | 4 | 4 | 4 |
| DeltaNet layers | 18 | **24** | **24** |
| FullAttn layers | 6 | **8** | **8** |
| `mtp_num_hidden_layers` | 1 | 1 | 1 |
| `tie_word_embeddings` | true | true | **false** |
| `rope_theta` | 1e7 | 1e7 | 1e7 |
| `rope_sections` | [11,11,10,0] | [11,11,10,0] | [11,11,10,0] |
| `rms_norm_eps` | 1e-6 | 1e-6 | 1e-6 |

### GGUF sizes

| Model | Q4_K_M | Q8_0 | BF16 |
|---|---|---|---|
| 4B | 2.74 GB | 4.48 GB | 8.42 GB |
| 9B | 5.68 GB | 9.53 GB | 17.9 GB (no fit) |

### VRAM budget (RTX 5070 Ti, 16 GB)

| Component | 4B Q4_K_M | 4B Q8_0 | 9B Q4_K_M | 9B Q8_0 |
|---|---|---|---|---|
| Model weights | 2.74 GB | 4.48 GB | 5.68 GB | 9.53 GB |
| DeltaNet S state (all layers) | 48 MB | 48 MB | 48 MB | 48 MB |
| Conv1d state (all layers) | 2.3 MB | 2.3 MB | 2.3 MB | 2.3 MB |
| KV cache (4K ctx, FP16) | 128 MB | 128 MB | 128 MB | 128 MB |
| Scratch buffers (~) | ~200 MB | ~200 MB | ~300 MB | ~300 MB |
| **Total** | **~3.1 GB** | **~4.8 GB** | **~6.1 GB** | **~10.0 GB** |
| **Headroom** | **~12.9 GB** | **~11.2 GB** | **~9.9 GB** | **~6.0 GB** |

The 4B at Q8_0 or even BF16 (8.42 GB) fits comfortably. The 9B fits at Q4_K_M and Q8_0 but NOT at BF16.

### Critical architectural difference: V_heads != K_heads

The 0.8B has `n_k_heads == n_v_heads == 16`. Both 4B and 9B have `n_k_heads=16, n_v_heads=32` (2:1 ratio). This means:
- QKV projection output: Q=[16*128], K=[16*128], V=[**32**\*128] = 2048+2048+4096 = 8192
- DeltaNet S matrix: [**32** heads, 128, 128] = 2 MB/layer (vs 1 MB for 0.8B)
- `ssm_alpha/beta` projections: [hidden, **32**] (vs [hidden, 16])
- `attn_gate` (Z): [hidden, **4096**] (vs [hidden, 2048])
- Conv1d: [4, **8192**] (vs [4, 6144])
- The llama.cpp GGUF converter reorders V heads from grouped-by-K to tiled layout

### DeltaNet S matrix total: 48 MB

This is significant. The 0.8B's total S was 18 MB, easily fitting in the 48 MB L2. At 48 MB total S for 4B/9B, it **exactly equals** the full L2 cache. L2 persistence for S is no longer viable — there's no room left for anything else. This changes the optimization calculus: pinning S in L2 (which we never tried) wouldn't have helped anyway at this scale.

---

## Phase 0: Model Download + llama.cpp Baseline

**Goal**: Get GGUF files, establish reference outputs and performance baseline.

### Steps
1. Download Qwen3.5-4B-Q4_K_M.gguf from `unsloth/Qwen3.5-4B-GGUF` → `~/models/`
2. Download Qwen3.5-9B-Q4_K_M.gguf from `unsloth/Qwen3.5-9B-GGUF` → `~/models/`
3. Run llama.cpp benchmarks on both:
   - `llama-bench -m ~/models/Qwen3.5-4B-Q4_K_M.gguf -t 1 -ngl 999 -p 512 -n 128`
   - Same for 9B
   - Record: tok/s decode, tok/s prefill (prompt eval), TTFT
4. Generate reference outputs (greedy, temp=0) for correctness comparison:
   - Short prompt: "The capital of France is"
   - Medium prompt: 512 tokens of English text
   - Save token-by-token output + final logits for each
5. Record all baselines in `blog/25-scaling-up.md`

### Acceptance criteria
- Have GGUF files and llama.cpp baselines for both models
- Reference outputs saved for correctness testing

---

## Phase 1: Generalize ModelConfig + GGUF Metadata Reading

**Goal**: Make GWEN load any Qwen3.5 variant from GGUF metadata instead of hardcoded defaults.

### What changes

**`include/gwen/common.h` — ModelConfig**:
```
Current:
  ssm_n_heads = 16      (ambiguous: K or V heads?)
  ssm_inner_size = 2048

Needed:
  ssm_n_k_heads = 16    (linear_num_key_heads, GGUF: ssm_group_count)
  ssm_n_v_heads = 16/32 (linear_num_value_heads, GGUF: ssm_time_step_rank)
  ssm_inner_size = v_heads * dv (derived, but validate against GGUF)
  ssm_qkv_dim = k_heads*dk + k_heads*dk + v_heads*dv (new: total QKV projection width)
  tie_word_embeddings = true/false
```

**`src/model.cu` — Model::load()**:
- Read GGUF metadata keys: `qwen35.block_count`, `qwen35.embedding_length`, `qwen35.feed_forward_length`, `qwen35.attention.head_count`, `qwen35.attention.head_count_kv`, `ssm.conv_kernel`, `ssm.state_size`, `ssm.group_count`, `ssm.time_step_rank`, `ssm.inner_size`
- Fall back to current defaults if keys missing (backward compat with 0.8B GGUF)
- For 9B: load `output.weight` as a separate `lm_head` WeightRef (currently uses `token_embd` for both)

**`include/gwen/model.h` — Model struct**:
- Add `WeightRef lm_head` (points to `token_embd` when tied, separate tensor when not)

**`include/gwen/inference.h` — DeltaNetState**:
- Rename `n_heads` → clarify as `n_v_heads` (it's the count of S matrices)
- `qkv_dim` goes from `3 * ssm_inner` = 6144 to `2*k_heads*dk + v_heads*dv` = 8192

**`include/gwen/inference.h` — InferenceState scratch buffers**:
- All currently hardcoded comments like `[6144]`, `[2048]`, `[3584]`, `[4096]`, `[512]` become cfg-dependent
- No code changes needed if allocations already use `cfg.ssm_inner_size` etc. — **verify each one**

### Steps
1. Add new fields to ModelConfig, keep old `ssm_n_heads` as alias for `ssm_n_k_heads`
2. Add GGUF metadata reading to `Model::load()` with fallback
3. Add `lm_head` WeightRef to Model, set from `token_embd` or `output.weight`
4. Audit every allocation in `InferenceState::allocate()` — grep for literal numbers
5. Audit every kernel launch in `inference.cu` — grep for `cfg.ssm_n_heads`, any literal 16/2048/6144
6. Build and load the 0.8B GGUF — verify nothing regresses
7. Build and load the 4B GGUF — verify it loads, prints correct config, all tensor shapes match

### Acceptance criteria
- `gwen --model ~/models/Qwen3.5-4B-Q4_K_M.gguf --info` prints correct hyperparameters
- 0.8B still loads and runs identically (no regressions)

---

## Phase 2: DeltaNet Kernel Adaptation for Asymmetric V/K Heads

**Goal**: Handle `n_v_heads=32, n_k_heads=16` in the DeltaNet forward path.

### The problem

The current DeltaNet decode kernel launches one block per head (16 blocks). The S matrix is `[n_heads, dk, dv]` = `[16, 128, 128]`. Each block owns one 128x128 S slice.

For 4B/9B, the S matrix becomes `[32, 128, 128]` — 32 blocks, each still 128x128. The kernel itself (the inner recurrence logic) doesn't change because dk=dv=128 is the same. What changes:
- **Number of blocks**: 16 → 32 per kernel launch
- **QKV split**: Q is `[k_heads*dk]`, K is `[k_heads*dk]`, V is `[v_heads*dv]` — these are NOT equal widths anymore
- **Alpha/beta projections**: `[hidden, v_heads]` — one scalar per V head
- **Output projection**: operates on `[v_heads*dv]` = `[4096]`

### What changes

**`kernel_deltanet_fused` / `kernel_deltanet_fused_2tok`**:
- Currently: `int head = blockIdx.x` indexes into Q, K, V, S all with stride `dk`. All three have `n_heads` entries.
- New: `int v_head = blockIdx.x` (0..31). For Q and K, the corresponding K-head is `v_head / (n_v_heads / n_k_heads)` = `v_head / 2`. Two V heads share one K head.
- S indexing: `S + v_head * dk * dv` — unchanged (just more heads)
- Alpha/beta indexing: `alpha[v_head]` — now 32 values per token
- Gate/beta GEMV: currently one `[n_embed] × [n_heads]` dot product per head, now `[n_embed] × [n_v_heads]`

**`kernel_conv1d_silu`**:
- Operates on qkv_dim = Q+K+V concatenated. Width changes from 6144 to 8192.
- Already parameterized by `dim` — should just work.

**`kernel_gated_rmsnorm`**:
- Operates per-head on `dv` elements. `n_heads` → `n_v_heads`. Should work if parameterized.

**QKV projection split**:
- After the GEMV for `attn_qkv`, we split into Q, K, V by offset:
  - Q: `[0, k_heads*dk)` = `[0, 2048)`
  - K: `[k_heads*dk, 2*k_heads*dk)` = `[2048, 4096)`
  - V: `[2*k_heads*dk, 2*k_heads*dk + v_heads*dv)` = `[4096, 8192)`
- Currently this split is done with pointer arithmetic. The offsets must use the asymmetric dimensions.

### Steps
1. Update the QKV split logic in `forward_body` to use `cfg.ssm_n_k_heads * cfg.ssm_state_size` for Q/K width and `cfg.ssm_n_v_heads * cfg.ssm_state_size` for V width
2. Update `kernel_deltanet_fused` to accept `n_k_heads`, `n_v_heads` and compute K-head index from V-head index
3. Update `kernel_deltanet_fused_2tok` similarly
4. Update chunkwise DeltaNet kernels (prefill path) for asymmetric heads
5. Update `DeltaNetState` allocation: `n_v_heads * dk * dv` for S, `(conv_kernel-1) * qkv_dim` for conv
6. Verify on 0.8B (n_k_heads == n_v_heads, should be identity change)
7. First forward pass on 4B — compare output logits against llama.cpp

### Acceptance criteria
- 4B greedy decode matches llama.cpp: exact token match for first 20 tokens
- KL-divergence < 0.01 on logits vs llama.cpp
- 0.8B still matches (no regressions)

---

## Phase 3: Full Attention + FFN Scaling

**Goal**: Verify full attention layers and FFN work at 4B dimensions.

### Expected to "just work" (parameterized by cfg)
- RMSNorm: parameterized by `dim`, works at any width
- RoPE: parameterized by `n_heads`, `n_kv_heads`, `head_dim`, `rope_dim` — all correct for 4B
- Softmax: parameterized by `rows`, `cols`
- SwiGLU / activations: parameterized by `n`
- Embedding lookup: parameterized by `dim` (but verify Q6_K block math at dim=2560)
- dp4a GEMV: parameterized by `out_features`, `in_features` — should scale

### May need tuning
- **dp4a GEMV warp count dispatch**: Currently `NW=2` for small matrices, `NW=4` for larger. The thresholds may need adjustment for wider 4B matrices (2560→9216 vs 1024→3584).
- **CUTLASS GEMM tile size**: 128×128×32 may not be optimal for the wider 4B dimensions. Profile before changing.
- **Embedding lookup at dim=2560**: 2560/256 = 10 Q6_K blocks per row (vs 4 for 1024). Verify the kernel handles non-power-of-2 row counts.
- **Full attention GQA ratio**: 0.8B has 8:2 (4:1) Q:KV ratio. 4B has 16:4 (4:1). Same ratio, just more heads — should scale linearly.

### Steps
1. Run full forward pass on 4B, check output
2. If token mismatch: dump per-layer activations and bisect which layer diverges
3. Profile decode with nsys: identify if any kernel is disproportionately slow at new dimensions
4. Record decode tok/s for 4B

### Acceptance criteria
- 4B generates coherent text
- Greedy output matches llama.cpp for reference prompts
- Performance recorded (expected: 250-400 tok/s decode for Q4_K_M)

---

## Phase 4: Performance Baseline + Profile

**Goal**: Establish clean performance baselines before optimization experiments.

### Steps
1. Run decode benchmark: 128 tokens, greedy, measure tok/s (warm, averaged over 5 runs)
2. Run prefill benchmark: 512-token prompt, measure TTFT
3. nsys profile of one full decode step:
   - Break down by kernel: what % is GEMV, DeltaNet, RMSNorm, other?
   - Compare proportions vs 0.8B
4. ncu profile of the top 3 kernels:
   - Bandwidth utilization (achieved vs theoretical)
   - Compute utilization
   - Register pressure, occupancy
5. Record all results in blog post

### Key questions to answer
- Is the LM head (248K × 2560 GEMV) still the single biggest kernel? (Expected: yes, even bigger share)
- Does the 32-head DeltaNet kernel (vs 16-head) achieve reasonable SM utilization? (32 blocks on 70 SMs — 45% base occupancy)
- What's the bandwidth utilization gap vs 0.8B?

### Acceptance criteria
- Complete kernel-level breakdown with bandwidth utilization numbers
- Clear identification of top bottlenecks at 4B scale

---

## Phase 5: Optimization Experiments

These are the SM_120-specific optimizations that were **never tried** on the 0.8B. Each is an independent experiment — measure before/after, revert if no win.

### Experiment 5A: FP8 (E4M3) GEMV Kernels

**Hypothesis**: FP8 weights are half the size of FP16, cutting memory traffic in half for bandwidth-bound GEMV. Unlike Q4_K which requires complex dequantization (super-blocks, sub-blocks, scales, mins), FP8 dequant is a single hardware `cvt` instruction per element. This could yield simpler, faster kernels despite higher per-element byte cost vs Q4_K.

**Why not tried on 0.8B**: The 0.8B was distributed as Q4_K_M GGUF and all development used GGML quantization formats. FP8 is not a GGML format — it requires a separate quantization path.

**What to build**:
1. FP8 quantization script: convert Q4_K_M GGUF → FP8 weight file (E4M3 with per-tensor or per-channel FP32 scale)
2. `kernel_gemv_fp8_dp4a`: FP8 GEMV using native SM_120 `cvt` instructions. Each thread loads E4M3 bytes, converts to FP16 via hardware intrinsic, accumulates.
3. Alternative: `kernel_gemv_fp8_native`: Use FP8 MMA (`mma.sync.aligned.kind::f8f6f4.m16n8k32`) for batch-1 GEMV if we can formulate it as a thin GEMM.

**Measure**: Bandwidth utilization, tok/s vs Q4_K dp4a path. Also measure quality (perplexity on a reference text).

**Risk**: FP8 at 4B parameters may have non-trivial quality loss vs Q4_K_M (which has higher effective precision due to super-block structure). Perplexity check is essential.

### Experiment 5B: SM_120 GEMV Tuning (Vectorized Loads + Cache Hints)

**Hypothesis**: The Blackwell FP4 GEMV hackathon results showed that 256-bit vectorized loads and L1 cache hints are key to reaching speed-of-light on SM_120. Our current GEMV kernels use `half2` (32-bit) loads. Switching to `ld.global.v4.u64` (256-bit = 32 bytes per load) could improve bandwidth utilization.

**Why not tried on 0.8B**: Developed before the hackathon results were published. The dp4a GEMV was already faster than llama.cpp, so there was no urgency to explore further.

**What to build**:
1. Modified `kernel_gemv_q4_k_dp4a` with `ld.global.v4.u64` for weight loading (inline PTX)
2. Add `L1::no_allocate` streaming hint on weight loads (weights are read once per forward pass, no reuse — don't pollute L1)
3. Add `L1::evict_last` on activation loads (activations are reused across rows — keep in L1)
4. Experiment with `-maxrregcount=32` to increase occupancy (currently unconstrained)
5. Profile with ncu: compare achieved bandwidth, L1 hit rates, occupancy

**Measure**: Bandwidth utilization before/after, tok/s impact. Focus on the LM head GEMV (248K×2560 — the dominant kernel).

**Risk**: Q4_K block structure (super-blocks of 256 elements with sub-blocks) may not align cleanly with 256-bit vector loads. May need to restructure the inner loop.

### Experiment 5C: Mixed-Precision MMA (FP4 Weights × FP8 Activations)

**Hypothesis**: SM_120 supports `mma.sync.aligned.kind::f8f6f4.m16n8k64` which processes FP4 weights × FP8 activations in a single instruction with FP32 accumulation. This could halve memory traffic vs FP8 while achieving 8× the tensor core throughput of FP16.

**Why not tried on 0.8B**: FP4 at 0.8B parameters causes substantial quality degradation. At 4B, FP4 with per-channel scaling should recover most quality (95-98% benchmark recovery per NVIDIA's data).

**What to build**:
1. NVFP4 quantization: block-scaled E2M1 (16 values share an E4M3 scale factor + per-tensor FP32 scale)
2. `kernel_gemv_fp4`: Load FP4 weights + block scales, convert activations to FP8, accumulate via MMA or CUDA cores
3. For decode GEMV (N=1): FP4 helps by halving weight memory traffic. Tensor cores are underutilized at N=1 — the win is purely bandwidth.
4. For prefill GEMM (N>1): FP4 MMA at m16n8k64 could give real compute speedup

**Measure**: Bandwidth utilization, tok/s, perplexity at FP4 vs Q4_K_M vs FP8.

**Risk**: High implementation complexity. NVFP4 block scaling adds overhead per 16-element group. The hackathon results showed FP4 GEMV achieving only ~50% speed-of-light — it may not beat a well-tuned Q4_K dp4a kernel. Try FP8 (5A) first.

**Dependency**: Run 5A first. Only pursue 5C if FP8 shows clear wins.

### Experiment 5D: L2 Cache Persistence for Hot Tensors

**Hypothesis**: While the full DeltaNet S state (48 MB) no longer fits in L2, smaller hot tensors could benefit from persistence: output_norm, all layer norms, ssm_a/dt_bias parameters, and potentially the MTP projection weights.

**Why not tried on 0.8B**: Blog post 05 correctly identified that CUDA Graph execution has no temporal locality — weights are loaded once per forward pass in a fixed order, so L2 caching of weights doesn't help. But we never tried pinning the *small* tensors (norms, biases) that ARE accessed every token.

**What to pin** (4B model):
| Tensor class | Per-layer size | Total | Benefit |
|---|---|---|---|
| Layer norms (all 32) | ~20 KB | ~640 KB | Avoid 32 DRAM reads/token |
| ssm_a + dt_bias | 384 B | ~9.2 KB | Tiny, hot every DeltaNet layer |
| output_norm | 10 KB | 10 KB | Read every token |
| RoPE tables | ~2 KB | ~2 KB | Read every full attn layer |
| **Total** | | **~660 KB** | |

**Implementation**: `cudaAccessPolicyWindow` with `cudaAccessPropertyPersisting` on the norm/bias device pointers after upload.

**Measure**: Use ncu to verify L2 hit rate on these tensors before/after. Measure tok/s delta.

**Risk**: The total pinned data is tiny (<1 MB). The benefit may be unmeasurable — these tensors might already hit L2 naturally due to their small size. Worth 30 minutes to test.

### Experiment 5E: TMA for Weight Streaming

**Hypothesis**: TMA (Tensor Memory Accelerator) can issue async global→shared copies from a single thread, freeing the other 127 threads to do useful work while weights stream in. Currently, all threads participate in cooperative weight loading via `cp.async`.

**Why not tried on 0.8B**: TMA setup overhead (descriptor creation) was assumed to not justify the benefit for the small 0.8B weight matrices.

**What to build**:
1. Create TMA descriptors for each weight tensor during model upload
2. Modified GEMV kernel: thread 0 issues `cp.async.bulk.tensor` for the next weight tile while remaining threads process the current tile
3. Double-buffer in shared memory: while computing on tile A, tile B streams in via TMA

**Measure**: Compare kernel time vs current `cp.async` approach for the LM head GEMV.

**Risk**: The FlashAttention-for-5090 author achieved 94.4% of peak using standard `cp.async` without TMA, suggesting TMA's advantage is marginal for well-pipelined kernels on SM_120. The overhead of TMA descriptor management may eat the savings. Low priority — try after 5A/5B.

---

## Phase 6: Speculative Decode at 4B Scale

**Goal**: Port the MTP speculative decode pipeline to 4B.

### What changes
- MTP head FC projection: `[n_embed, 2*n_embed]` = `[2560, 5120]` (vs `[1024, 2048]`)
- MTP full attention layer: wider Q/K/V (16/4 heads vs 8/2)
- MTP hidden state: 2560 FP16 = 5 KB (vs 2 KB)
- S snapshot buffers for rollback: 24 layers × 2 MB = 48 MB (vs 18 layers × 1 MB = 18 MB)

The rollback cost goes from ~18 MB memcpy to ~48 MB memcpy. At 896 GB/s this is ~0.05 ms — still negligible.

### 4B MTP quality question
The 0.8B's single MTP layer achieved ~55% acceptance rate with exact-match greedy. The 4B's MTP layer operates on a wider hidden state (2560 vs 1024), which should give it more capacity. **But**: we fine-tuned the 0.8B MTP head on spoken English. For 4B, we'd need to either:
- Use the pretrained MTP weights as-is (likely lower acceptance on spoken English)
- Fine-tune the 4B MTP head (requires training infrastructure adaptation)
- Skip MTP initially and focus on raw decode performance

**Recommendation**: Skip MTP initially. Get raw decode working and optimized first. MTP fine-tuning is Phase 7+.

### Steps
1. Adapt MTP weight loading for 4B dimensions
2. Adapt MTP inference path (wider FC, wider attention)
3. Test with pretrained MTP weights
4. Measure acceptance rate and effective tok/s with speculation

### Acceptance criteria
- Speculative decode runs on 4B without crashes
- Effective tok/s measured with pretrained MTP head

---

## Phase 7: 9B Port (Stretch)

**Goal**: After 4B is working and optimized, scale to 9B.

The 4B→9B delta is smaller than 0.8B→4B:
- Same `n_v_heads=32, n_k_heads=16` (no structural change)
- Same 32 layers, same layer pattern
- Just wider: hidden 4096, intermediate 12288
- **One new thing**: `tie_word_embeddings=false` — separate `output.weight` tensor for lm_head

### Steps
1. Verify GGUF loading reads `output.weight` when present
2. Update `Model::lm_head` to point to `output.weight` instead of `token_embd`
3. Run forward pass, verify correctness
4. Profile and optimize (the wider dimensions may shift bottleneck proportions)

### VRAM concern
9B Q4_K_M at 5.68 GB + state (48 MB) + KV (128 MB) + scratch (~300 MB) ≈ 6.1 GB. Comfortable.
9B Q8_0 at 9.53 GB + state + KV + scratch ≈ 10.0 GB. Tight but fits.
The separate lm_head adds ~0.5 GB at Q6_K for the output projection (248320 × 4096). Already counted in the GGUF size.

---

## Expected Performance Targets

Based on bandwidth-bound analysis (896 GB/s theoretical, ~60% utilization typical):

| Model | Q4_K_M size | Theoretical max | Expected | With MTP spec decode |
|---|---|---|---|---|
| 0.8B (actual) | 495 MB | 1810 tok/s | 605 tok/s | 928 tok/s |
| 4B | 2.74 GB | 327 tok/s | 200-250 tok/s | 300-400 tok/s |
| 9B | 5.68 GB | 158 tok/s | 100-130 tok/s | 150-200 tok/s |

The 0.8B achieves 33% of theoretical maximum (605/1810). If we maintain that ratio:
- 4B: ~108 tok/s (conservative). If optimization experiments improve BW utilization to 40%: ~130 tok/s.
- 9B: ~52 tok/s (conservative). With improved BW: ~63 tok/s.

Speculative decode at 1.5× boost (same as 0.8B):
- 4B: ~160-200 tok/s
- 9B: ~80-100 tok/s

These numbers assume Q4_K_M. With FP8 (experiment 5A), model size approximately doubles, halving throughput — but if FP8 GEMV kernels are simpler and achieve higher BW utilization, the net effect could be neutral or slightly positive.

---

## Priority Order

1. **Phase 0**: Download + baseline (1 hour)
2. **Phase 1**: Generalize ModelConfig (half day)
3. **Phase 2**: DeltaNet V/K asymmetry (1 day — the hard part)
4. **Phase 3**: Full attention + FFN verification (half day)
5. **Phase 4**: Profile (half day)
6. **Phase 5B**: Vectorized loads + cache hints (1 day — most likely to help, least risk)
7. **Phase 5A**: FP8 GEMV (2-3 days — new quantization path)
8. **Phase 5D**: L2 persistence for norms (30 min experiment)
9. **Phase 6**: Speculative decode port (1 day)
10. **Phase 5C**: FP4 mixed-precision MMA (3+ days — high effort, uncertain payoff)
11. **Phase 5E**: TMA weight streaming (2 days — likely marginal)
12. **Phase 7**: 9B port (half day after 4B works)

---

## Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-03-18 | Target 4B primary, 9B stretch | 4B fits at Q8/BF16, ~90% kernel reuse, same architecture |
| 2026-03-18 | Skip FP4 initially, try FP8 first | FP8 is simpler (no block scaling), lower quality risk, still halves traffic vs FP16 |
| 2026-03-18 | Skip MTP fine-tuning in first pass | Get raw decode optimized first, MTP is a training project |
| 2026-03-18 | L2 persistence for S not viable at 4B+ | 48 MB S = 100% of L2 cache, no room for other data |
| 2026-03-18 | Vectorized loads (5B) before FP8 (5A) | 5B is a targeted change to existing Q4_K kernels, 5A requires new quant format |
