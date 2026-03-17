# Post 23: Server-Side p_idk, Clean Pipeline, and Shared Memory

MTP v4 adds an "IDK" (I Don't Know) output token that absorbs OOV probability mass. The old approach computed p_idk in Python with a 248K×1024 matmul per batch — 62-hour ETA for the training corpus. This post covers moving p_idk to the CUDA server, cleaning up the training pipeline, and the shared memory optimization that brought training throughput from 11.5K to 14.4K tok/s.

## The p_idk Problem

The teacher distribution for IDK training needs the full-vocab partition function: `log_Z = logsumexp(hidden @ embed_248K.T)`. With K=4096 restricted tokens covering ~85% of the corpus, the remaining ~15% probability mass becomes p_idk — the probability that the correct token is outside our vocabulary.

Computing this in PyTorch on GPU meant a [batch, 248320] FP16 matmul per training batch. At 3.2K tok/s through `gwen_server`, pre-computing p_idk for 498M tokens would take 62 hours. Meanwhile, the `dev_server` sitting right there was pushing 20K+ tok/s with all the GPU infrastructure already warmed up.

## CUDA Kernels: logsumexp + p_idk

Two new kernels in `src/kernels/reduction.cu`:

**`gwen_logsumexp_rows`** — row-wise logsumexp for the 248K GEMM output. One block (256 threads) per row, two-pass: find max via warp+shared-mem reduction, then sum exp(x - max). Standard numerically-stable pattern, nothing fancy.

**`gwen_p_idk_from_logits`** — compute p_idk from the already-computed restricted logits plus log_Z. One warp per row: `p_idk = clamp(1 - sum(exp(restricted_logits - log_Z)), 0, 1)`. The restricted logits are already in GPU memory from the teacher GEMM, so this kernel just reads them and the freshly-computed log_Z.

Correctness: max error vs PyTorch `torch.logsumexp` reference was **3×10⁻⁶** — essentially exact.

## Extending /batch_logits

The dev_server's `/batch_logits` endpoint gained `?p_idk=1`:

1. Full token_embd (248K × 1024) dequantized to FP16 at startup (+485 MB GPU)
2. On request: chunked 248K GEMM (512 tokens per chunk) + logsumexp + p_idk reduction
3. Response: appends `[float32 p_idk[N]]` after the existing hidden + logits
4. Without `?p_idk=1`: identical response — fully backward compatible

Overhead at 32K tokens: **~9%** (14.8K → 13.6K tok/s). The 248K GEMM reads the 485 MB weight matrix once per chunk, and the logsumexp + p_idk kernels are negligible.

### Failed optimization: fused GEMM+logsumexp

Tried to avoid materializing the [chunk, 248K] intermediate by fusing the dot product and online logsumexp into one kernel. Two attempts:

1. **Shared-memory reduction** (16 threads/row, RPB=24 rows/block): `__syncthreads()` inside the 248K column loop = 248K barriers per block. Catastrophically slow.

2. **Warp-per-row** (32 threads/warp, 8 warps/block, no barriers): each warp independently streams through 485 MB of weights. No weight reuse between warps → L2 thrashing. 24× slower than CUTLASS.

The CUTLASS GEMM already has optimal tiling and L2 reuse for this shape. The ~9% overhead from the unfused approach is the practical floor without writing a custom CUTLASS epilogue.

## Clean Data Pipeline

Numbered the scattered scripts into a clear pipeline:

```
scripts/00_prepare_training_data.py   — raw text → data/train_tokens.bin
scripts/01_prepare_restricted_vocab.py — counts → data/restricted_vocab_{K}.bin
scripts/02_extract_val_cache.py        — dev_server → val_cache/ (hidden + p_idk)
```

Extracted `GwenClient` from the 1600-line `train_mtp.py` into `train/gwen_client.py`. Removed ~330 lines of dead code: `precompute_p_idk()`, `compute_teacher_dist_idk()`, `compute_teacher_dist_from_p_idk()`, the `precompute-idk` CLI subcommand, and the `full_embed_gpu` loading path.

Val cache extraction got: tqdm progress bars, restartability (skips existing batches via atomic tmp+rename writes), a background write queue, and a post-extraction verification pass.

## Shared Memory: Eliminating HTTP Overhead

Training at 11.5K tok/s with a 20K tok/s server? The gap was Python's `http.client` copying 320 MB per batch (256 MB logits + 64 MB hidden + 128 KB p_idk). Reading 320 MB through Python's HTTP stack takes ~1.2 seconds of pure overhead per batch.

**Fix**: the server `cudaMemcpy`s results directly into a pre-allocated POSIX shared memory region (`/dev/shm/gwen_batch`). The HTTP response carries only the 12-byte header (B, L, K). The client mmaps the same region and reads with `np.frombuffer` — zero Python HTTP overhead for the bulk data.

```
Before: GPU (1.66s) → cudaMemcpy → HTTP 320MB (1.2s) → numpy parse = 2.86s (11.5K tok/s)
After:  GPU (1.66s) → cudaMemcpy to shm → HTTP 12B → mmap read    = 1.96s (14.4K tok/s)
```

The `?shm=1` query parameter enables it; clients that don't ask for it get the normal HTTP response. The `no_logits=1` parameter was also added for the val cache extraction path, which only needs hidden + p_idk (saving 256 MB of unused logit transfer).

## train_mtp.py Cleanup

The IDK training loop now calls `batch_logits_with_p_idk()` for hidden + logits + p_idk in one server call. The teacher distribution is built inline:

```python
p_restricted = F.softmax(teacher_logits / T, dim=-1) * (1 - p_idk.unsqueeze(-1))
teacher_probs = torch.cat([p_restricted, p_idk.unsqueeze(-1)], dim=-1)
```

No more `full_embed_gpu` on the training GPU, no precompute step, no cache loading. Temperature decays from 2.0→1.0 over training. Default epochs changed from 3 to 1 (one pass over 500M tokens is sufficient for this distillation setup).

## Results

| Metric | Value |
|--------|-------|
| p_idk correctness | max error 3×10⁻⁶ vs PyTorch |
| Server overhead (p_idk) | 9% at 32K tokens |
| Training throughput | 14.4K tok/s (was 11.5K, +25%) |
| Training GPU memory saved | ~484 MB (no full_embed_gpu) |
| Server GPU memory | +728 MB (full embed + chunk buffer) |
| Epoch time (498M tokens) | ~9.5 hours |
| Old precompute time | 62 hours (eliminated) |
