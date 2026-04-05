# Post 36: Fusing Gate Computation into the DeltaNet Kernel

Post 35's profiling showed DeltaNet element-wise ops consuming 10% of GPU time — sigmoid, softplus, add, and mul operating on [16]-element vectors in separate CUDA kernel launches. Each launch costs ~1μs for ~0.01μs of actual work. This post fuses all of it into the DeltaNet kernel.

## What Was Happening

Every DeltaNet layer (18 of them) computes two gating signals from the hidden state:

```
beta  = sigmoid(ssm_beta @ hidden)              # learning rate per head
alpha = exp(softplus(ssm_alpha @ hidden + dt_bias) * ssm_a)  # decay factor per head
```

Previously, the ggml compute graph had 6 separate nodes per layer:
1. `mul_mat` — ssm_beta × hidden → raw beta [16]
2. `sigmoid` — raw beta → beta
3. `mul_mat` — ssm_alpha × hidden → raw alpha [16]
4. `add` — raw alpha + dt_bias
5. `softplus` — log(1 + exp(alpha_biased))
6. `mul` — softplus × ssm_a

That's 6 × 18 = 108 element-wise kernel launches per forward pass, plus the 2 MMVQ launches for the matmuls. Each element-wise op processes a [16]-element vector — the kernel launch overhead dominated the actual compute by 100:1.

## The Fusion

Added a `FUSE_GATES` template parameter to `gated_delta_net_cuda`. When active, the kernel receives raw alpha/beta values (straight from the MMVQ output) plus dt_bias and ssm_a as additional pointers. The gate computation happens inline before the recurrence update:

```cuda
if constexpr (FUSE_GATES) {
    const float alpha_raw = *g_t;
    const float beta_raw  = *beta_t;

    // sigmoid(beta)
    beta_val = 1.0f / (1.0f + expf(-beta_raw));

    // exp(softplus(alpha + dt_bias) * ssm_a)
    const float alpha_biased = alpha_raw + dt_bias[h_idx];
    const float sp = alpha_biased > 20.0f ? alpha_biased : logf(1.0f + expf(alpha_biased));
    g_val_precomputed = expf(sp * ssm_a[h_idx]);
}
```

All threads in the block share the same `h_idx` (blockIdx.x), so `dt_bias[h_idx]` and `ssm_a[h_idx]` are a single L1-cached read. No shared memory needed.

## The Plumbing

The gate fusion touches 6 files across 3 layers:

**ggml layer:**
- `ggml.h` — declared `ggml_gated_delta_net_fused_gates()` taking 8 sources (q, k, v, alpha_raw, beta_raw, state, dt_bias, ssm_a) + l2_norm_eps
- `ggml.c` — implemented it: calls base `ggml_gated_delta_net()`, sets op_params[1] as fuse flag, assigns src[6]=dt_bias, src[7]=ssm_a

**CUDA kernel:**
- `gated_delta_net.cu` — added `FUSE_GATES` template parameter, `dt_bias`/`ssm_a` kernel args, inline gate math. Dispatch reads op_params[1] and routes to the appropriate template instantiation. Uses a macro to avoid repeating the launch code 8 times (KDA × L2_NORM × FUSE_GATES).

**Model layer:**
- `models.h` — added `dt_bias`/`ssm_a` optional params to `build_delta_net_fused()` and `build_delta_net()`
- `delta-net-base.cpp` — routes to `ggml_gated_delta_net_fused_gates()` when dt_bias/ssm_a are provided
- `qwen35.cpp` — when fused GDN is active, passes raw alpha/beta + constants instead of pre-computed gates. Moved `will_use_fused` determination earlier in `build_layer_attn_linear()` so the gate fusion decision is available before the element-wise ops.

## No Challenges

This was a clean implementation. The main design decision was whether to use shared memory or global reads for dt_bias/ssm_a. Since all threads in a block share the same head index, a single global read per value is L1-cached after the first access. No shared memory, no synchronization, no register pressure increase.

The softplus overflow guard (`alpha_biased > 20.0f ? alpha_biased : logf(1.0f + expf(alpha_biased))`) matches the standard numerical stability threshold — above 20.0, exp(x) dominates 1.0, so log(1+exp(x)) ≈ x.

## Results

```
                    Before fusion    After fusion    Delta
Baseline (tg64):    453              489             +8%
MTP avg:            615              638             +3.7%
MTP peak:           660              693             +5%
MTP worst:          576              596             +3.5%
```

The 8% baseline improvement is notable — the fusion benefits single-token decode (no MTP) equally since all 18 DeltaNet layers use the fused path. The CUDA graph node count drops by ~108 nodes per forward pass (4 element-wise + 2 reshape per layer × 18 layers), which reduces graph warmup time and property checking overhead.

Per-prompt breakdown:

| Prompt | Before | After | Delta |
|--------|--------|-------|-------|
| AI history | 580 | 601 | +3.6% |
| Narrative | 605 | 628 | +3.8% |
| Business | 606 | 631 | +4.1% |
| TCP/UDP | 616 | 638 | +3.6% |
| Quantum | 601 | 618 | +2.8% |
| Transformer | 635 | 657 | +3.5% |
| Python sort | 605 | 626 | +3.5% |
| Python class | 576 | 596 | +3.5% |
| Math word | 617 | 640 | +3.7% |
| Fibonacci | 623 | 648 | +4.0% |
| Fox repeat | 660 | 684 | +3.6% |
| Counting | 653 | 693 | +6.1% |

## Files Changed

| File | Change |
|------|--------|
| `ggml/include/ggml.h` | `ggml_gated_delta_net_fused_gates()` declaration |
| `ggml/src/ggml.c` | Implementation — sets op_params[1] and src[6:7] |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | `FUSE_GATES` template, inline gate math, dispatch macro |
| `src/models/models.h` | Added dt_bias/ssm_a params to build_delta_net_fused/build_delta_net |
| `src/models/delta-net-base.cpp` | Routes to fused_gates when dt_bias/ssm_a provided |
| `src/models/qwen35.cpp` | Skip element-wise ops, pass raw values + constants |
