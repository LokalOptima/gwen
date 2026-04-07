# Post 37: Thinking Loops — Investigation and Mitigation

The Qwen3.5-0.8B instruct model gets stuck in infinite repetition loops during
generation. This post documents how we found them, what causes them, what
doesn't fix them, and what does.

**Date**: 2026-04-05 through 2026-04-07

---

## Discovery

Thinking mode in Qwen3.5 requires `--jinja` — without it, the built-in `chatml`
template doesn't activate thinking and the model emits empty `<think></think>`
stubs. With `--jinja --reasoning on`, the model generates reasoning inside
`<think>` blocks before answering.

We ran 8 reasoning-heavy prompts (math proofs, logic puzzles, chess analysis)
with `-n 8192 --temp 0` and found:
- **All 8 prompts** produced unclosed `<think>` blocks (model never stopped thinking)
- **7/8** contained exact token-level repetition cycles

Example from `recursive_proof`: a 119-token block repeating 64 times, consuming
92.9% of the think budget:

> *Wait, actually, the standard definition is: A string is balanced if it is
> a sequence of `(` and `)` such that the number of `(` equals the number of `)`
> and for every `(`, there is a corresponding `)` immediately following it.*
> *Wait, let's check the standard definition again.*

Example from `prime_listing`: `473 / 13 = 36.38.` repeating 380 times (16
tokens per cycle, 74% of the think block).

Scripts: `scripts/test_think_loops.sh`, `scripts/analyze_think_loops.py`,
`scratch/investigate_loops.py`, `scratch/logprob_investigation.py`.

---

## The ratcheting mechanism

Using `llama-server` with `logprobs: true`, we measured token probabilities
across consecutive cycles of the `recursive_proof` loop:

| Token | Period 1 | Period 2 | Period 3 |
|---|---|---|---|
| `" Wait"` | **0.467** | 0.677 | 1.000 |
| `" definition"` | **0.249** | 0.999 | 1.000 |
| `" of"` | **0.206** | 0.999 | 0.999 |

The first cycle enters with moderate confidence. Each repetition in the KV cache
makes the next more certain. By period 3, every token is locked at P ≈ 1.000 with
gaps of 15+ nats to any alternative. The `</think>` token was available at
onset (logprob -13.54) but its probability only decreases with each cycle.

The `prime_listing` loop skips the ratchet — it enters at P > 0.999 from the
first cycle. Two distinct dynamics:

| | Semantic oscillation | Computational fixation |
|---|---|---|
| Period | Long (100+ tokens) | Short (16 tokens) |
| Lock-in | Gradual, 2-3 cycles | Instant |
| Entropy | Stays high | Collapses |

---

## What doesn't fix it

### Sampling parameters

Tested greedy, temp 0.7, DRY penalty, presence penalty. Loops persist across
all configs — different prompts loop under different settings, but the total
count stays around 7/8 with the fixed detector.

### Qwen's recommended settings

Qwen's model card recommends `temp=1.0, top_p=1.0, top_k=20,
presence_penalty=2.0` for non-thinking text. Testing with these exact settings
initially showed 0/8 loops — but that was a **detector bug**. The n-gram
analyzer used geometric period spacing (1.2x) which jumped from period 44 to 52,
missing the actual period of 50. After fixing the analyzer to check every
integer period:

```
LOOP: 6  |  WARN: 1  |  OK: 1  |  total: 8
```

The presence penalty forces enough surface variation ("Wait, I am repeating
myself again. Let's step back.") to dodge exact n-gram matching at some periods,
but the model still enters exact token-level cycles. The `sqrt2_irrational`
output contains a 103-token block repeating 7 times with 0 mismatches — and the
model literally says "Wait, I am repeating myself again" each cycle.

### Thinking OFF (`--reasoning off`)

Before our fix, `--reasoning off` was **silently ignored** by `llama-completion`.
The flag was only wired up in the server code. The completion tool never passed
`enable_thinking` to the jinja template, so it defaulted to `true`.

---

## Loops are not thinking-specific

Tested the instruct model without thinking and the base model with raw completion:

| Condition | Loops (4 prompts) |
|---|---|
| Instruct + thinking ON | 3/4 |
| Instruct + thinking OFF | 4/4 |
| Instruct raw (no template) | 0/4 (but 2 hit EOS early at <900 words) |
| Base model raw | 0/4 (but 3 hit EOS early at <350 words) |

The instruct model without chat template: `sqrt2` loops on `$$2s^2 = 4s^2$$`
in the response body (no `<think>` block). The base model: `chess` repeats
"Qxd4" 231 times. The non-looping cases just hit EOS before a cycle forms.

Thinking mode is the amplifier, not the cause — it encourages indefinite
generation with no natural stopping point.

---

## Not a quantization issue

BF16 instruct and F16 base loop identically to Q8_0:

| Condition | Q8_0 | BF16 |
|---|---|---|
| Instruct + thinking ON | 3/4 | 2/4 |
| Instruct + thinking OFF | 4/4 | 4/4 |

Same failure modes, same prompts. Full-precision weights don't help.

---

## Larger models

The 4B model (same 3:1 DeltaNet:FullAttn architecture, 32 layers) is better
but not immune. 5/8 prompts close their `<think>` blocks and produce answers
(vs 0/8 for 0.8B). The ones that don't close are heading toward the same loop
patterns, just slower:

- `pnp_debate` at 4B: numbering a list (246, 247, 248...) repeating the same sentence
- `recursive_proof` at 4B: "Wait, I need to be careful" oscillation, period=26 × 3

---

## This is a known problem

The Qwen3.5-0.8B model card states:

> "Qwen3.5-0.8B is more prone to entering thinking loops compared to other
> Qwen3.5 models, which may prevent it from terminating generation properly."

The 0.8B defaults to non-thinking mode for this reason. Greedy decoding is
explicitly warned against. GitHub issues document the same problem across the
Qwen3.5 family up to 122B. DeepSeek-R1's model card reports "endless
repetition" in R1-Zero.

Benchmarks don't catch it because they score the final answer (a number, a
letter, a `\boxed{}`), not the thinking tokens. The evaluation protocol allows
81,920 tokens of generation, but the scored output is tiny.

---

## The fix

### 1. Wire `--reasoning off` through to the completion tool

One-line change in `tools/completion/completion.cpp` — pass `enable_reasoning`
to the jinja template inputs:

```cpp
inputs.enable_thinking = params.enable_reasoning != 0;
```

Previously, `enable_thinking` defaulted to `true` and `--reasoning off` was
silently ignored.

### 2. Default to non-thinking mode

Changed `common/common.h`:
```cpp
int enable_reasoning = 0; // was -1 (auto)
```

Matches the Qwen3.5-0.8B model card: non-thinking is the default for this model.
`--reasoning on` still overrides when explicitly passed.

### 3. Hardcode Qwen3.5-0.8B recommended sampling defaults

From the [model card](https://huggingface.co/Qwen/Qwen3.5-0.8B), non-thinking
text task row:

| Parameter | llama.cpp default | Qwen recommended | New default |
|---|---|---|---|
| `top_k` | 40 | 20 | **20** |
| `top_p` | 0.95 | 1.00 | **1.00** |
| `min_p` | 0.05 | 0.00 | **0.00** |
| `temp` | 0.80 | 1.00 | **1.00** |
| `presence_penalty` | 0.00 | 2.00 | **2.00** |

### 4. Fix correctness tests

`test_correctness.sh` now explicitly passes `--presence-penalty 0` for
deterministic greedy comparison (MTP correctness needs clean greedy decoding).

### Result

With reasoning disabled and recommended sampling:
- 4 prompts complete in **3-6 seconds** (vs 20+ seconds stuck thinking)
- **0 loops** detected
- 36/36 MTP correctness tests pass

```
Tests: 36/36 pass (MTP correctness, all lengths).
```
