# Post 41: MTP Sidecar GGUF — Decoupling MTP from the Base Model

MTP weights no longer need to be baked into a custom GGUF. A standalone
sidecar file works with any stock Qwen3.5-0.8B GGUF, producing bit-identical
output with zero performance cost.

**Date**: 2026-04-08

---

## Motivation

Until now, MTP speculative decoding required a custom GGUF
(`Qwen3.5-0.8B-mtp-Q8_0.gguf`) that bundled:

- The 24-layer base model (Q8_0)
- The MTP transformer block as layer 24 (F16)
- `nextn_predict_layers = 1` in the GGUF metadata

Plus a separate GWRL binary (`lm_head_top50000.bin`) for the restricted
vocabulary LM head.

This coupling meant:

1. Can't use upstream GGUF files directly (any quantization, any source)
2. Need a custom conversion step to produce the MTP GGUF
3. Two different file formats (GGUF for the model + GWRL binary for LM head)

---

## Design

### Sidecar GGUF format

A single `*-mtp.gguf` file containing everything MTP needs:

**Tensors** (16 total, 79.1 MiB):
- 15 MTP layer tensors (`blk.24.*`) — F16 attention + FFN + MTP-specific
  projections (eh_proj, enorm, hnorm, shared_head_norm)
- 1 restricted LM head (`mtp_lm_head`) — Q6_K, 50K vocab

**Metadata**:
- `gwen.mtp_version` (uint32) = 1
- `gwen.mtp_token_ids` (array of 50000 int32) — restricted-to-full vocab mapping

### Auto-discovery

When loading a model with `nextn_predict_layers == 0`, the loader checks:

1. `LLAMA_MTP_GGUF` env var (explicit path)
2. `<model_path>` with `.gguf` replaced by `-mtp.gguf` (naming convention)

Example: loading `Qwen3.5-0.8B-Q8_0.gguf` auto-discovers
`Qwen3.5-0.8B-Q8_0-mtp.gguf` in the same directory.

### Loading sequence

The sidecar loads in `llama_model::load_mtp_sidecar()`, called after the
main model load completes:

1. Set `hparams.nextn_predict_layers = 1` and populate per-layer arrays
   (n_head, n_head_kv, n_ff, recurrent) for the MTP layer index
2. Extend `layers[]` and `dev_layer[]` to include layer 24 on GPU
3. Open sidecar GGUF, allocate a single GPU buffer, upload all tensors
4. Wire tensor pointers into `layers[24]` fields
5. Read token ID mapping from GGUF metadata
6. Copy `tok_embd` from CPU to GPU (needed for CUDA graph capture —
   stock GGUF puts embeddings on CPU, but MTP needs single-backend graphs)

The context constructor then picks up `mtp_sidecar_lm_head` and
`mtp_sidecar_token_ids` from the model, falling back to the GWRL binary
if no sidecar is present.

Skipped during `no_alloc` probe loads (the `-fit` memory estimation phase).

---

## Extraction script

`scripts/extract_mtp_gguf.py` extracts the sidecar from an existing combined
MTP GGUF:

```bash
./scripts/extract_mtp_gguf.py \
    --mtp-gguf ~/.cache/gwen/Qwen3.5-0.8B-mtp-Q8_0.gguf \
    --lm-head  ~/.cache/gwen/lm_head_top50000.bin \
    --output   ~/.cache/gwen/Qwen3.5-0.8B-mtp-head.gguf
```

Reads the combined GGUF for `blk.24.*` tensors, the GWRL binary for the
restricted LM head + token IDs, and writes a standalone sidecar GGUF.
Handles F16, F32, and quantized (Q6_K) tensor types correctly by converting
between GGML element shapes and GGUF byte shapes.

---

## Verification

### Correctness

12 prompts at 50 tokens, greedy (`--temp 0 --presence-penalty 0`):

| Path | Result |
|---|---|
| Combined MTP GGUF | 12/12 PASS |
| Stock GGUF + sidecar | 12/12 PASS |

Outputs are bit-identical between the two paths.

### Stochastic decode

Tested with `--temp 0.7 --top-k 40 --presence-penalty 1.5 --seed 42`:
599.5 tok/s, 75% acceptance, coherent output. Both greedy and stochastic
MTP paths work with the sidecar.

### Performance

200 tokens, greedy, same prompt:

| Path | tok/s | Accept rate | Accepted | Rejected |
|---|---|---|---|---|
| Combined MTP GGUF | 680.2 | 75.4% | 86 | 28 |
| Stock GGUF + sidecar | 680.8 | 75.4% | 86 | 28 |

Zero overhead — identical token counts, identical acceptance, within noise
on throughput.

---

## Files changed

- `scripts/extract_mtp_gguf.py` — sidecar extraction script (new)
- `src/llama-model.h` — `load_mtp_sidecar()` method, sidecar LM head/token
  ID storage
- `src/llama-model.cpp` — sidecar loading implementation (hparams setup,
  GPU buffer allocation, tensor upload, tok_embd GPU copy)
- `src/llama.cpp` — auto-discovery logic, `no_alloc` guard
- `src/llama-context.cpp` — sidecar LM head priority over GWRL binary fallback

---

## Usage

```bash
# Extract sidecar (one-time)
./scripts/extract_mtp_gguf.py

# Create naming-convention symlink
ln -s Qwen3.5-0.8B-mtp-head.gguf ~/.cache/gwen/Qwen3.5-0.8B-Q8_0-mtp.gguf

# Run with stock GGUF — sidecar auto-discovered
./build/bin/llama-completion -m ~/.cache/gwen/Qwen3.5-0.8B-Q8_0.gguf \
    -p "Hello world" -n 100

# Or explicit sidecar path
LLAMA_MTP_GGUF=~/.cache/gwen/Qwen3.5-0.8B-mtp-head.gguf \
./build/bin/llama-completion -m ~/.cache/gwen/Qwen3.5-0.8B-Q8_0.gguf \
    -p "Hello world" -n 100
```

Both the combined MTP GGUF and the stock + sidecar paths remain supported.
