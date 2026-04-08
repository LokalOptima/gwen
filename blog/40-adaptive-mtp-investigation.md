# Post 40: Adaptive MTP — Investigation and Dead End

Can we dynamically disable MTP when acceptance is low and re-enable when it
recovers? We built it, benchmarked it thoroughly, and found it hurts more than
it helps. This post documents the full investigation.

**Date**: 2026-04-08

**Branch**: `adaptive-MTP` (full implementation preserved for reference)

---

## Motivation

MTP speculative decode gives +33% throughput at ~80% acceptance, but what
happens on prompts where acceptance drops? If the 2-token batch + fixup
overhead exceeds the speculation benefit, MTP could theoretically slow
generation below baseline. An adaptive system that pauses MTP during
low-acceptance regions could provide a safety floor.

---

## What we built

### Async MTP launch/sync split

Split `decode_mtp()` into two phases:

- `decode_mtp_launch()` — copies hidden state, builds/reuses graph, dispatches
  `graph_compute_async`. Returns immediately.
- `mtp_sync_result()` — syncs scheduler, extracts argmax, advances KV cache.

The original `decode_mtp()` becomes a thin wrapper calling both. The intent was
to overlap MTP compute with the next main model decode on separate CUDA
scheduler streams.

### Sliding window acceptance tracker

O(1) ring buffer (`mtp_acceptance_tracker`) tracking the last 16 accept/reject
decisions. Provides instant acceptance rate queries.

### ACTIVE/MONITORING state machine

Two modes in `generate_mtp()`:

- **ACTIVE**: Standard speculative decode (2-token batch, accept/reject).
  Records acceptance in the tracker. Transitions to MONITORING when the sliding
  window acceptance drops below `PAUSE_THRESHOLD`.
- **MONITORING**: Single-token baseline decode. Periodically probes MTP to
  track virtual acceptance. Transitions back to ACTIVE when virtual acceptance
  exceeds `RESUME_THRESHOLD`.

Hysteresis via cooldown counter prevents oscillation.

---

## Benchmark methodology

Prior benchmarks used `--temp 0 --presence-penalty 0` (greedy, no anti-
repetition). This allows the model to loop freely, and MTP trivially predicts
loops — inflating both acceptance and throughput. Example: "Fox repeat" showed
98% acceptance at 688 tok/s, but the output was just "The quick brown fox..."
repeated 20 times.

All results below use **stochastic sampling with anti-repetition**:

```
--temp 0.7 --top-k 40 --presence-penalty 1.5 --seed <varies>
```

18 diverse prompts, 3 repetitions per prompt per mode (different seeds), 200
tokens each. RTX 5070 Ti. Three-way comparison: no-MTP baseline, MTP always-on,
MTP adaptive.

---

## Results: three-way comparison

| Prompt | No MTP | Always-on | Adaptive | Alw.spd | Ada.spd | Accept |
|---|---|---|---|---|---|---|
| Counting | 429.5 | 636.8 | 637.5 | +48.3% | +48.4% | 99.0% |
| Fox_repeat | 432.7 | 619.2 | 612.4 | +43.1% | +41.5% | 92.1% |
| Fibonacci | 433.5 | 591.2 | 590.6 | +36.4% | +36.2% | 88.8% |
| Math_word | 432.3 | 588.0 | 539.9 | +36.0% | +24.9% | 74.4% |
| Lists_enum | 431.4 | 574.1 | 575.4 | +33.1% | +33.4% | 80.5% |
| Logic_puzzle | 432.1 | 576.4 | 578.2 | +33.4% | +33.8% | 81.4% |
| AI_history | 433.5 | 577.7 | 482.4 | +33.3% | +11.3% | 64.3% |
| Dialogue | 432.4 | 575.3 | 574.2 | +33.0% | +32.8% | 79.5% |
| Mixed_lang | 432.2 | 603.2 | 523.3 | +39.6% | +22.2% | 76.0% |
| Python_sort | 432.0 | 570.9 | 573.7 | +32.2% | +32.8% | 76.8% |
| Quantum | 433.2 | 573.5 | 497.7 | +32.4% | +14.9% | 70.3% |
| TCP_UDP | 432.6 | 564.2 | 524.2 | +30.4% | +21.2% | 72.3% |
| Transformer | 431.8 | 558.9 | 525.0 | +29.4% | +21.6% | 72.2% |
| Poetry | 432.7 | 561.1 | 514.4 | +29.7% | +18.9% | 63.7% |
| Python_class | 421.2 | 553.8 | 436.4 | +31.5% | +3.6% | 55.3% |
| Narrative | 433.2 | 555.2 | 454.5 | +28.2% | +4.9% | 59.5% |
| Business | 434.1 | 556.1 | 475.5 | +28.1% | +9.5% | 67.1% |
| Gibberish | 432.9 | 520.0 | 427.7 | +20.1% | -1.2% | 39.0% |

**Averages:**

| Mode | tok/s | vs baseline |
|---|---|---|
| No MTP | 431.9 | — |
| MTP always-on | **575.3** | **+33.2%** |
| MTP adaptive | 548.6 | +22.8% |

Always-on MTP is faster than adaptive on 12/18 prompts. Adaptive is never
faster than always-on. On "Gibberish" (worst case), adaptive is -1.2% below
baseline while always-on is still +20.1%.

---

## Why adaptive hurts: monitoring is not free

### The overlap assumption was wrong

The design assumed MTP compute (~0.3ms) could overlap with the main decode
(~2.1ms) because MTP uses a separate `ggml_backend_sched`. Per-cycle profiling
of monitoring mode revealed otherwise:

```
CYCLE  18: MONITOR msync= 211 dispatch= 211 sync=2097 fsamp= 29 mlaunch= 41 total=2764 us
CYCLE  19: MONITOR msync= 210 dispatch= 210 sync=2097 fsamp= 30 mlaunch= 40 total=2742 us
```

`msync=211us` — MTP sync blocks for the full MTP compute time, sitting on the
critical path. Total cycle: 2750us vs baseline ~2130us = **+29% overhead**.

### Root cause: device-wide synchronization

`ggml_backend_sched_synchronize()` (ggml-backend.cpp:1808) iterates all
backends in the scheduler and calls `ggml_backend_synchronize()` on each:

```cpp
void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
}
```

Both the main and MTP schedulers are created with the **same backend pointers**
(llama-context.cpp:483). They share the same CUDA backend instance, which means
the same `cudaStream_t`. Syncing either scheduler calls
`cudaStreamSynchronize()` on that shared stream — waiting for ALL GPU work,
both main and MTP.

True async overlap would require creating a separate CUDA backend instance with
its own stream for MTP. ggml doesn't support multiple CUDA backend instances on
the same device.

### Reordering didn't help

Attempted reorder: dispatch main decode first (async), then sync MTP while main
runs. Result: `msync` absorbed the full main decode time (2131us) because the
device-wide sync waited for both.

### Probe-based monitoring: better but still a net loss

Replaced every-cycle MTP with periodic probes (every 16 tokens). Monitoring
overhead dropped from -14% to ~-1% vs baseline. But the monitoring cycles still
run at baseline speed (~433 tok/s) instead of MTP speed (~575 tok/s), making
adaptive strictly worse than always-on.

---

## Threshold sweep

Swept pause threshold from 0.55 down to 0.25 on the 9 worst-case prompts:

| Config | Avg tok/s | Monitor Cycles | vs Always-on |
|---|---|---|---|
| p=0.55/r=0.70 | 512.2 | 728 | -8.3% |
| p=0.45/r=0.60 | 543.7 | 204 | -2.7% |
| p=0.40/r=0.55 | 548.0 | 105 | -1.9% |
| p=0.35/r=0.50 | 550.0 | 69 | -1.6% |
| p=0.30/r=0.45 | 552.6 | 39 | -1.1% |
| p=0.25/r=0.40 | 558.7 | 17 | -0.02% |
| always-on | 558.8 | 0 | — |

Monotonic: every monitoring cycle costs throughput. At p=0.25 adaptive is
essentially always-on (17 total monitoring cycles across 27 runs).

---

## The presence-penalty=0 trap

Early benchmarks used `--temp 0 --presence-penalty 0`, showing +43.2% speedup
at 85.1% acceptance. Inspection revealed many prompts looped ("The quick brown
fox..." ×20, "Okay thanks" ping-pong, paragraph repeats). MTP trivially
predicts loops, inflating the numbers.

The correctness test (`test_correctness.sh`) requires `--presence-penalty 0`
because `generate_mtp`'s greedy path does raw argmax and ignores penalties.
With non-zero penalties, the standard and MTP sampling paths diverge. This is a
known limitation of the greedy MTP path.

Switching to stochastic sampling with penalties (temp=0.7, top_k=40,
presence_penalty=1.5) gives the honest numbers reported above.

---

## Conclusion

For this model (Qwen3.5-0.8B with well-tuned MTP head), **MTP always-on is
the optimal strategy**:

- +33.2% average speedup over baseline on realistic prompts
- Never slower than baseline on any prompt (worst case: +20.1%)
- Zero complexity — no state machine, no monitoring, no thresholds

The adaptive machinery would only help if MTP could be slower than baseline,
which requires acceptance consistently below ~40-45%. This MTP head, distilled
on the instruct model, never reaches that level.

The investigation yielded two useful findings for future work:

1. **ggml's synchronization is device-wide** — true async overlap between two
   ggml schedulers on the same GPU is impossible without creating separate CUDA
   backend instances with their own streams.
2. **Greedy benchmarks must use anti-repetition penalties** — the 0.8B model
   loops freely with penalty=0, inflating MTP acceptance rates by 5-10pp.

---

## Files (on `adaptive-MTP` branch)

- `include/llama.h` — `llama_decode_mtp_launch`, `llama_mtp_sync_result` API
- `src/llama-context.h` — async split member declarations
- `src/llama-context.cpp` — `decode_mtp_launch`, `mtp_sync_result` implementation
- `tools/completion/completion.cpp` — acceptance tracker, state machine,
  monitoring mode, env var overrides (`LLAMA_MTP_NO_ADAPTIVE`,
  `LLAMA_MTP_PAUSE`, `LLAMA_MTP_RESUME`)
- `scripts/bench_adaptive.sh` — three-way benchmark script
