# Methodology: Porting Neural Net Inference to Custom CUDA

A reusable guide for agents implementing custom GPU inference engines from scratch, distilled from the GWEN project (pure CUDA reimplementation of a hybrid transformer, achieving +34% over llama.cpp).

This is not about any specific model — it's the approach.

---

## Core Principle: Correctness, Then Performance, Then Elegance

Every decision follows this priority. A fast wrong answer is worthless. A correct slow answer is a starting point. Only optimize what you've proven correct.

---

## Phase 1: Understand the Model

Before writing any CUDA code:

### 1.1 Find a reference implementation
You need a ground-truth oracle that produces correct outputs. Options:
- llama.cpp (broad model support, C++, easy to extract intermediate values)
- HuggingFace Transformers (Python, easy to inspect, slow but correct)
- vLLM / TensorRT-LLM (if you need production baselines)

**Build the reference first.** Run it. Verify it generates sane text. Save its outputs — token IDs, not text — for comparison.

### 1.2 Map the computation graph
Extract from the reference:
- **Layer pattern**: Which layer types appear, in what order, how many times
- **Tensor shapes**: Every weight matrix's dimensions and quantization type
- **Data flow**: What feeds into what. Draw it if needed.
- **Quirks**: Non-standard normalization, unusual activation functions, hybrid architectures (attention + recurrence), gating mechanisms

### 1.3 Understand the weight format
If using GGUF:
- Parse metadata with `gguf` Python package
- Enumerate all tensors: name, shape, quantization type, byte offset
- Understand the quantization block layout (e.g., Q4_K = 256 elements in 144 bytes: 12 scales + 4 mins + 128 quantized bytes)
- Note which tensors use which quant type — it's not uniform

### 1.4 Compute theoretical limits
Before writing any kernel, know the ceiling:
```
total_weight_bytes = sum(all weight tensors in bytes)
total_traffic = weight_bytes + state_read_write + activation_traffic
t_min = total_traffic / GPU_peak_bandwidth
max_tok_per_s = 1 / t_min
```

This number is your upper bound. Everything you build is approaching it.

---

## Phase 2: Build Kernels Bottom-Up

Implement in dependency order. After each kernel, immediately test against the reference.

### Recommended order
1. **Weight loading + dequantization** — GGUF parsing, quant block decode
2. **Embedding lookup** — trivial, but lets you test the pipeline
3. **Normalization** (RMSNorm/LayerNorm) — simple, tests basic CUDA setup
4. **Matrix-vector multiply** (GEMV) — the performance-critical kernel for single-token decode
5. **Activation functions** (SiLU, GELU, SwiGLU) — element-wise operations
6. **Positional encoding** (RoPE, ALiBi, etc.) — if applicable
7. **Attention mechanism** — KV cache, score computation, softmax, value accumulation
8. **Model-specific layers** — linear attention (DeltaNet, Mamba), mixture of experts, etc.
9. **Full forward pass** — wire all components together
10. **Output layer** — logit computation, argmax/sampling

### Testing each kernel
After implementing kernel N:
1. Run it on the same input the reference implementation uses
2. Compare output element-by-element
3. Acceptance criteria: max element-wise difference < threshold (e.g., 0.01 for FP16 accumulation)
4. For the full forward pass: exact greedy token match over 30+ tokens

**Never skip testing. Never "fix it later."** A bug in layer 3 will compound through layers 4-24 and produce garbage that's very hard to debug.

### Debugging divergence
When tokens diverge from the reference:
1. Find the first token that differs
2. Compare logits at that position — are they close (accumulated numerical error) or wildly different (algorithmic bug)?
3. If wildly different: bisect layers. Dump intermediate activations from both engines after each layer until you find where they diverge.
4. Common causes: wrong scale extraction from quant blocks, nibble interleave errors, off-by-one in block indexing, transposed dimensions, incorrect GQA head mapping

---

## Phase 3: Performance Optimization

### 3.1 The optimization loop

```
while (not fast enough) {
    1. Measure end-to-end performance
    2. Profile to find the biggest bottleneck
    3. Understand WHY it's slow (not just THAT it's slow)
    4. Implement targeted fix
    5. Verify correctness still holds
    6. Measure again
}
```

**Never optimize without profiling first.** Your intuition about what's slow is wrong at least half the time.

### 3.2 Profiling toolkit

#### nsys (Nsight Systems) — Timeline analysis
What it tells you: where time is spent across the entire GPU timeline.

```bash
# CRITICAL: use --cuda-graph-trace=node if using CUDA graphs
# Without it, graph-replayed kernels are invisible!
nsys profile --trace=cuda --cuda-graph-trace=node \
    -o profiles/trace ./build/engine ...

# Per-kernel timing summary
nsys stats --report cuda_gpu_kern_sum profiles/trace.nsys-rep
```

Look for:
- Which kernels dominate total GPU time
- Gaps between kernels (launch overhead, CPU bottlenecks)
- Unexpected hotspots (utility kernels that shouldn't be slow)

#### ncu (Nsight Compute) — Per-kernel deep dive
What it tells you: why a specific kernel is slow.

```bash
# Requires elevated permissions:
#   sudo ncu ... (use full path: sudo /usr/local/cuda-X.Y/bin/ncu)
#   OR: echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-perf.conf
# Use regex for template kernel names:
#   --kernel-name regex:"pattern"

sudo /path/to/ncu --kernel-name regex:"gemv" --set full \
    --launch-skip 15 --launch-count 10 \
    ./build/engine ...
```

**Key metrics for bandwidth-bound kernels (decode GEMV):**

| Metric | Meaning | Good value |
|--------|---------|-----------|
| `dram__throughput.avg_pct_of_peak` | % of peak DRAM bandwidth | >70% |
| `launch__registers_per_thread` | Register pressure | <42 for full occupancy |
| `sm__warps_active.avg_pct_of_peak` | Achieved occupancy | >50% |
| `smsp__warps_issue_stalled_long_scoreboard` | Memory latency stalls | Low |

**Sector utilization check (memory coalescing):**
```
useful_bytes_per_sector = bytes_loaded / sectors_requested
# Ideal: 32 bytes/sector
# If << 32: uncoalesced access → wasted bandwidth
```

### 3.3 Common performance problems and fixes

#### Problem: Single-block kernels on many-SM GPUs
**Symptom**: A kernel takes 100+ μs despite doing little work.
**Cause**: Launched with 1 block → uses 1 SM while 69 sit idle.
**Fix**: Multi-block parallel reduction. Phase 1: N blocks each find local max. Phase 2: 1 block reduces N partial results.
**Real example**: Argmax over 248K vocab: 139 μs → 3.5 μs (40x) with 256 blocks.

#### Problem: Under-threaded blocks
**Symptom**: Block does serial work that could be parallel.
**Cause**: Block size chosen for convenience (32 threads) rather than the data dimension (256).
**Fix**: Match thread count to data dimension. If head_dim=256, use 256 threads.
**Real example**: GQA attention with 32 → 256 threads: 276 μs → 47 μs (5.9x).

#### Problem: Poor memory coalescing
**Symptom**: ncu shows DRAM throughput at 30-50% despite high occupancy.
**Cause**: Data layout forces threads to access non-contiguous memory. E.g., Array-of-Structures where each struct is 144 bytes, and 32 threads each access byte 0 of different structs → 4608-byte stride.
**Fix**: Structure-of-Arrays weight relayout at load time, or shared memory staging.
**Diagnosis**: Check `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` — if bytes/sector << 32, you have a coalescing problem.

#### Problem: Warp divergence / shuffle mask deadlock
**Symptom**: Kernel hangs or produces wrong results.
**Cause**: `__shfl_xor_sync(0xFFFFFFFF, val, offset)` called inside a conditional where not all 32 warp threads participate.
**Fix**: Use shared memory for cross-warp communication instead of intra-warp shuffles in conditionals. Or ensure all warp threads reach the shuffle.

#### Problem: Kernel launch overhead dominates
**Symptom**: Many small kernels (norms, activations, quantize) each take 1-3 μs of useful work but add 1-2 μs launch overhead each.
**Fix options**:
1. **Kernel fusion**: Combine adjacent kernels (e.g., RMSNorm + quantize, SwiGLU + quantize, GEMV + residual add)
2. **CUDA graphs**: Capture the kernel sequence once, replay with minimal dispatch cost
3. **Persistent kernels**: One kernel that runs the entire forward pass (highest effort, highest payoff)

#### Problem: CUDA graph hiding bottlenecks
**Symptom**: nsys shows a single `cudaGraphLaunch` bar per decode step — no per-kernel visibility.
**Fix**: Use `nsys profile --cuda-graph-trace=node` to decompose graph replays into individual kernel timing.

### 3.4 Compute vs memory bound analysis

Single-token decode is almost always **memory-bandwidth-bound**. The arithmetic intensity is:
```
AI = 2 * output_features / bytes_per_weight_element ≈ 0.5-4 FLOP/byte
```

The GPU's compute-memory crossover (ridge point) is typically at 60-250 FLOP/byte, so decode GEMVs are 10-100x below the ridge.

Prefill (batch GEMM) transitions from memory-bound to compute-bound as sequence length grows:
```
AI_gemm ≈ min(M, N) / (2 * bytes_per_elem)
# At seq_len=128, FP16: AI ≈ 64 → near the ridge
# At seq_len=512, FP16: AI ≈ 256 → compute-bound
```

### 3.5 Quantization-aware GEMV design

For quantized weight GEMV (Q4_K, Q5_K, Q6_K):
1. **Quantize the input vector to Q8_1** (per-32-element blocks with float scale + sum)
2. **Use dp4a (INT8 dot product)** for the inner loop: 4 multiplies + accumulate per instruction
3. **Dequantize accumulator to FP32** for the final reduction

The dp4a approach gives ~2x throughput vs FP16 dequant-then-multiply because:
- 4 INT8 ops per dp4a instruction vs 2 FP16 ops per HFMA2
- Avoids the dequant → FP16 → multiply → FP32 pipeline

### 3.6 CUDA graph optimization

CUDA graphs eliminate per-kernel launch overhead by capturing a sequence of kernel calls and replaying them with a single API call.

**When to use**: Decode path where the same kernel sequence repeats every token.
**Graph node count matters**: Each kernel in the graph adds ~1 μs of dispatch cost even inside a graph. Reducing from 370 to 250 nodes saved ~0.12ms per decode step.
**Limitation**: Graph parameters (like position counter) must be updated via device-side memory, not host-side arguments.

---

## Phase 4: Documentation and Handoff

### What to document

1. **Performance log** — After every optimization, record:
   - What changed
   - Before/after numbers (tok/s, ms per component)
   - Why it worked (or didn't)

2. **Theoretical analysis** — Keep updated:
   - Total traffic per decode step (exact, from weight sizes)
   - Theoretical minimum latency
   - Current efficiency (measured / theoretical)
   - Per-component breakdown with % of forward pass

3. **Profiling results** — When you profile, record:
   - Which tool (nsys vs ncu)
   - Key metrics and their values
   - What the metrics told you about the bottleneck
   - What you did about it

4. **Blog posts** — One per major milestone:
   - What was implemented
   - Key architectural decisions and why
   - Bugs encountered and how they were resolved
   - Benchmark numbers with full context

### Benchmark reporting template

```
GPU:            [model] (SM_XX)
Driver:         [version]
CUDA:           [version]
Clocks:         SM [freq] MHz, Mem [freq] MHz (locked/unlocked)
Model:          [name] ([parameter count], [quant type])
Total weights:  [bytes] ([MB])

Decode (N tokens):
  Throughput:   [tok/s]
  Latency:      [ms/token]
  Forward pass:  [ms] mean ([ms] min, [ms] max, N=[runs])
  BW efficiency: [%] (vs [peak] GB/s)
  vs reference:  +/-[%] vs [reference] ([ref tok/s])
  Token match:   [N]/[N] exact greedy

Component breakdown:
  [kernel category]    [ms]  [% forward]  [BW efficiency if applicable]
  ...
  Graph overhead       [ms]  (extra, not in forward)
```

### Handoff checklist

When handing off to another agent:
- [ ] Current performance numbers (tok/s, forward pass ms)
- [ ] What was just completed (with commit hashes)
- [ ] What's the next bottleneck and how you know (profiling data)
- [ ] Open questions with hypotheses
- [ ] File map (which files matter, what they do)
- [ ] How to build, test, and benchmark
- [ ] Known bugs or limitations

---

## Key Invariants to Maintain

1. **Correctness first**: Never commit an optimization without running the full correctness suite
2. **Numbers are measured, not estimated**: Every performance claim should be reproducible with a script
3. **Profile before optimizing**: Three failed optimization attempts means you're guessing — profile instead
4. **Lock clocks for benchmarks**: GPU boost behavior causes 5-15% variance
5. **Document as you go**: A future agent (or your future self) needs to know why decisions were made

---

## Typical Performance Journey

Based on the GWEN project trajectory:

| Milestone | Tok/s | vs Reference | Key Technique |
|-----------|------:|-------------|---------------|
| First correct output | ~50 | -80% | Naive FP16 GEMV, no fusion |
| Basic optimization | ~180 | -60% | CUDA graphs, basic fusion |
| Quantized GEMV | ~250 | -40% | dp4a INT8 dot product |
| Aggressive fusion | ~450 | +4% | Fused norms+quant, GEMV+residual |
| Kernel launch reduction | ~490 | +10% | More fusion, fewer graph nodes |
| Profiling-guided fixes | ~600 | +34% | Multi-block argmax, multi-warp attention |
| Weight relayout (todo) | ~700+ | +50%+ | SoA coalescing fix |

The pattern: each phase roughly doubles remaining headroom. Initial gains come from basic CUDA competency (graphs, occupancy, fusion). Later gains require detailed profiling and understanding specific hardware bottlenecks.
