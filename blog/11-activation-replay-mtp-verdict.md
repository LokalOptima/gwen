# Activation Replay and the MTP Verdict

*Blog post #11 in the GWEN series — when 42x faster rollback still isn't enough*

## The Reject Cost Problem

[Post #9](09-mtp-speculative-decoding.md) left MTP speculative decoding at 30% slower than baseline. The culprit was the reject path: when the draft token is wrong, we need to undo the DeltaNet recurrent state changes and re-advance the model. That checkpoint-and-re-forward approach cost **1.68 ms per rejection** — almost a full decode step wasted.

The math was brutal. With `forward_2tok` at 2.26 ms and `forward_mtp` at 0.18 ms, an accept cycle costs 2.44 ms for 2 tokens (1.22 ms/tok). But a reject cycle costs 2.44 + 1.68 + 0.18 = **4.30 ms for 1 token** — 2.6x worse than baseline's 1.66 ms/tok. You'd need >75% acceptance to break even.

The reject path breaks down as:
- Save 19.3 MB of DeltaNet state (checkpoint): ~0.02 ms
- Restore 19.3 MB on reject: ~0.02 ms
- Re-run `forward()` to advance state past token A: **1.66 ms** (the killer)

That re-forward is the real problem. It re-reads all 495 MB of model weights just to replay one token's state update.

## Prior Art: Activation Replay for SSMs

Before attempting a fix, I searched for how others handle this. The literature is clear:

**"The Mamba in the Llama"** (Junxiong et al., NeurIPS 2024) — Hybrid Mamba-Transformer models with speculative decoding. They observe that SSM state makes rollback expensive and propose caching intermediate activations to avoid re-computation.

**STree** (Gupta et al., NeurIPS 2025) — Speculative tree decoding for Mamba models. Their key insight: save SSM states at tree branch points and restore on reject, rather than re-running the forward pass.

**"Snakes and Ladders"** (Bhatt et al., EMNLP 2024) — Directly addresses speculative decoding overhead for state-space models. Recommends activation replay as the primary strategy.

The consensus: **save the state, don't re-compute it**.

## Three Attempts at Algebraic Undo

Before going to brute-force snapshots, I tried to be clever and compute the undo analytically. All three failed.

### Attempt 1: Algebraic Inversion from S_new

The DeltaNet recurrence is:
```
d = beta * (v - S @ k)       # delta
S_new = diag(exp(gate)) * S + k ⊗ d   # state update
```

In theory, you can invert this if you save `k`, `v`, `beta`, and `gate`:
```
d = (v - S_new @ k) * beta / (1 - beta)
S_old = (S_new - k ⊗ d) / diag(exp(gate))
```

Result: **Fiction prompt diverged at token 12.** The division by `exp(gate)` amplifies float32 rounding errors. With gate values around 0.1-2.0, the round-trip `S * exp(gate) / exp(gate)` introduces ~1 ULP error per element. Across 18 layers × 16 heads × 128×128 elements, errors accumulate fast.

### Attempt 2: Exact Delta Save

Modified `kernel_deltanet_decode` to output the exact `d` values computed during the forward pass, avoiding the algebraic reconstruction. The undo becomes:
```
S_old = (S_new - k ⊗ d_saved) * exp(-gate)
```

Result: **Nature prompt diverged at token 22.** Even with exact `d`, the multiply-then-divide-by-decay round-trip (`S * exp(gate) * exp(-gate)`) still isn't identity in float32. After 30 rejections, each touching 18 layers, the accumulated error crosses the threshold.

### Attempt 3: Direct S Snapshot (the one that works)

Stop trying to be clever. Just save the damn state.

Between processing token A and token B in `forward_body_2tok`, insert 18 device-to-device memcpy operations — one per DeltaNet layer — saving the S matrix after token A's update:

```cpp
// Inside forward_body_2tok, after token A's deltanet_decode:
size_t S_bytes = state.n_heads * state.state_size * state.state_size * sizeof(float);
cudaMemcpyAsync(dn_S_snapshot[dn_idx], state.S, S_bytes,
                cudaMemcpyDeviceToDevice, stream);
```

18 copies × 1 MB each = 18 MB total. At 896 GB/s, that's ~20 microseconds. And since it's all async memcpy baked into the CUDA graph, it overlaps with other work.

On reject, restore with the reverse copies:
```cpp
void InferenceState::undo_deltanet_token_b(cudaStream_t stream) {
    for (int i = 0; i < n_dn_layers; i++) {
        auto& state = deltanet_states[i];
        size_t S_bytes = state.n_heads * state.state_size * state.state_size * sizeof(float);
        cudaMemcpyAsync(state.S, dn_S_snapshot[i], S_bytes,
                         cudaMemcpyDeviceToDevice, stream);
    }
    // + conv1d undo (see below)
}
```

Result: **8/8 prompts produce identical output to baseline greedy decoding.** Zero precision loss, because D2D memcpy is bit-exact.

## Conv1d State Undo

DeltaNet also has a conv1d rolling buffer per layer (3 rows × 6144 elements). The `forward_body_2tok` shifts this buffer for both tokens A and B. On reject, we need to undo token B's shift.

This is a pure data-shuffling operation — no arithmetic, no precision loss. Before token B's conv1d, we save `conv_state[0]` (the row about to be evicted). On reject, we reverse the shift:

```cpp
__global__ void kernel_conv1d_undo_batch(
    float** conv_ptrs, const float* saved_rows,
    int dim, int n_layers)
{
    int layer = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (layer >= n_layers || idx >= dim) return;

    float* conv = conv_ptrs[layer];
    float row2 = conv[0 * dim + idx];   // current row 0 = old row 1
    float x_A  = conv[1 * dim + idx];   // current row 1 = token A's input
    conv[0 * dim + idx] = saved_rows[layer * dim + idx];  // restore evicted row
    conv[1 * dim + idx] = row2;          // old row 1 back to slot 1
    conv[2 * dim + idx] = x_A;           // token A back to slot 2
}
```

All 18 layers in a single kernel launch (2D grid: elements × layers). Cost: ~2 microseconds.

## The New Reject Path

Old approach:
```
restore checkpoint (19.3 MB)     ~0.02 ms
re-run forward() on token A      ~1.66 ms   ← the killer
run forward_mtp() for new draft   ~0.18 ms
                          Total:  ~1.86 ms
```

New approach:
```
restore S from snapshots (18 MB)  ~0.02 ms
conv1d undo (18 layers, 1 kernel) ~0.002 ms
swap mtp_hidden                    ~0.001 ms
run forward_mtp() for new draft    ~0.18 ms
                          Total:   ~0.20 ms
```

**The reject path dropped from 1.86 ms to 0.20 ms — a 9.3x improvement.** The reject *overhead* (excluding the MTP call common to both) dropped from 1.68 ms to 0.023 ms — **73x faster**.

This completely changes the break-even math:
- Accept cycle: 2.26 + 0.18 = 2.44 ms for 2 tokens → 1.22 ms/tok
- Reject cycle: 2.26 + 0.02 + 0.18 = 2.46 ms for 1 token → 2.46 ms/tok
- Break-even acceptance rate: **~65%** (down from 75%)

## Benchmark: The Final Numbers

With activation replay implemented, 500 tokens per prompt, 8 diverse prompts:

| Category | MTP tok/s | Base tok/s | Delta | Accept% |
|----------|-----------|------------|-------|---------|
| Scientific | 568 | 609 | -6.8% | 55.5% |
| Code | 589 | 586 | +0.5% | 62.5% |
| Fiction | 595 | 582 | +2.2% | 64.4% |
| Economics | 553 | 607 | -8.9% | 52.1% |
| Formal | 533 | 583 | -8.6% | 47.6% |
| Technical | 515 | 586 | -12.1% | 41.9% |
| Narrative | 577 | 606 | -4.8% | 59.4% |
| DevOps | 557 | 585 | -4.9% | 53.2% |
| **Average** | **561** | **593** | **-5.4%** | **54.6%** |

MTP is still net negative on average. But the picture changed dramatically from post #9:

- **Post #9** (old checkpoint): -30% average, no prompts positive
- **Post #11** (activation replay): -5.4% average, 2 prompts positive (Code, Fiction)

The problem is no longer the reject path — it's the acceptance rate.

## Why 55% Isn't Enough

At 54.6% average acceptance, we're below the 65% break-even threshold. The fundamental issue is that a single-layer MTP head on a 0.8B model just doesn't predict accurately enough across all domains:

- **High acceptance** (60-65%): Code, Fiction, Narrative — domains with predictable patterns, repetition, and strong local context
- **Low acceptance** (42-53%): Technical, Economics, Formal — domains with specialized vocabulary and less predictable token sequences

The MTP head sees exactly one transformer layer of context. For a 0.8B model, that's a lot of capacity relative to the main model — but the draft still needs to match the full 24-layer model's output exactly (greedy decoding requires exact match, no sampling tolerance).

## The Forward_2tok Bandwidth Tax

Even if acceptance were 100%, MTP can't do better than:
```
2.26 ms (forward_2tok) + 0.18 ms (MTP) = 2.44 ms for 2 tokens
= 1.22 ms/tok = 820 tok/s theoretical max
```

Compare to baseline's 1.66 ms/tok (603 tok/s). The 2-token forward reads weights once but does 2x the compute per weight element — so it's slightly less than 2x the single-token cost. This is a +36% speedup ceiling for single-draft speculative decode.

To go further, you'd need deeper speculation (predict 3+ tokens) or a batched dp4a kernel that truly amortizes weight reads across tokens.

## Lessons Learned

**1. Float32 is not exact.** This sounds obvious, but the specific failure mode was subtle: `exp(g) * exp(-g)` is NOT 1.0 in float32. Across 18 layers with 128×128 matrices, ~1 ULP error per element per rejection snowballs into divergent output after 20-30 rejections. Don't try to algebraically invert floating-point operations — save and restore instead.

**2. D2D memcpy is basically free.** 18 MB of device-to-device copies at 896 GB/s takes ~20 microseconds. When you're saving 1.66 ms of re-computation, that's a 83x return on investment. Modern GPU memory bandwidth makes "save everything" viable even at scale.

**3. Bake saves into the CUDA graph.** The S snapshot copies are async memcpy nodes in the CUDA graph that executes `forward_body_2tok`. Zero additional launch overhead. The graph runtime schedules them alongside compute kernels, so much of the copy time overlaps with other work.

**4. Profile before and after.** Without nsys traces, I wouldn't have identified the 1.66 ms re-forward as the dominant reject cost. And after the fix, profiling confirmed the saves are invisible in the trace — they overlap completely with GEMV execution.

**5. Acceptance rate is the real bottleneck.** All the engineering in the world on the reject path doesn't help if the draft model can't predict accurately enough. For Qwen3.5-0.8B's single-layer MTP head, 55% average acceptance with exact-match greedy decoding is the ceiling. Larger models with deeper MTP heads (Qwen3.5-3B has 2 MTP layers, 7B has 4) would likely see much higher acceptance rates.

## The Verdict

MTP speculative decoding on Qwen3.5-0.8B is a net negative for throughput. Activation replay solved the engineering problem (exact rollback in 0.04 ms instead of 1.68 ms re-forward), but the fundamental acceptance rate of ~55% is below the ~65% break-even point for single-draft speculation.

The infrastructure is there and correct — 8/8 diverse prompts produce identical output to baseline greedy. If a future model version ships with a better MTP head or if we implement multi-token speculation (predict 2-3 draft tokens), the activation replay mechanism is ready.

For now, GWEN's best decode speed remains **599 tok/s** with the straight single-token path. Not every optimization pays off, but you learn the most from the ones that don't.
