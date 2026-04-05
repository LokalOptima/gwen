# Post 35: Profiling the Instruct Q8_0 Decode Pipeline

Detailed kernel-level analysis of the MTP speculative decode pipeline on Qwen3.5-0.8B (instruct) Q8_0, profiled on RTX 5070 Ti.

## Methodology

Profiled with `nsys` using `GGML_CUDA_DISABLE_GRAPHS=1` to expose individual kernel launches (normally hidden inside CUDA graph replay). Prompt "A", 200 tokens, greedy decode. The 88.5% acceptance rate means most cycles take the 2-token accept path.

Note: disabling CUDA graphs adds ~40% overhead (462 vs 615 tok/s) from kernel launch costs. The kernel time percentages remain accurate — it's the launch overhead that inflates wall time.

## The Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    MTP Speculative Decode Cycle                  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Main Model Decode (2-token batch on accept, 1 on reject) │   │
│  │                                                          │   │
│  │  For each of 24 layers:                                  │   │
│  │                                                          │   │
│  │  DeltaNet layers (×18):                                  │   │
│  │    ┌─────────┐  ┌──────────┐  ┌────────────────────────┐│   │
│  │    │get_rows │→│rms_norm  │→│quantize + MMVQ (QKV)   ││   │
│  │    │(embed)  │  │(1024)    │  │Q8_0, N=6144            ││   │
│  │    └─────────┘  └──────────┘  └────────────────────────┘│   │
│  │    ┌──────────────────┐  ┌─────────────────────────────┐│   │
│  │    │sigmoid, softplus,│→│gated_delta_net_cuda         ││   │
│  │    │scale, mul (gates)│  │(fused L2 norm + recurrence) ││   │
│  │    └──────────────────┘  └─────────────────────────────┘│   │
│  │    ┌──────────┐  ┌──────────────────────────────────────┐│   │
│  │    │rms_norm  │→│quantize + MMVQ (gate,up) + SiLU     ││   │
│  │    │(1024)    │  │+ MMVQ (down) — FFN                  ││   │
│  │    └──────────┘  └──────────────────────────────────────┘│   │
│  │                                                          │   │
│  │  Full attention layers (×6):                             │   │
│  │    ┌──────────┐  ┌──────────────────────────────────────┐│   │
│  │    │rms_norm  │→│quantize + MMVQ (Q,K,V)              ││   │
│  │    │(1024)    │  │Q8_0, N=4096/1024/1024               ││   │
│  │    └──────────┘  └──────────────────────────────────────┘│   │
│  │    ┌────────────┐  ┌───────┐  ┌────────────────────────┐│   │
│  │    │RoPE + KV   │→│softmax│→│MMVQ (output) + FFN     ││   │
│  │    │cache write │  │       │  │                        ││   │
│  │    └────────────┘  └───────┘  └────────────────────────┘│   │
│  │                                                          │   │
│  │  ┌──────────┐  ┌──────────────┐  ┌───────────┐          │   │
│  │  │rms_norm  │→│MMVQ lm_head  │→│argmax     │          │   │
│  │  │(output)  │  │Q8_0 → logits │  │(248K vocab)│          │   │
│  │  └──────────┘  └──────────────┘  └───────────┘          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                     │
│              accept/reject decision (CPU)                        │
│                           │                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  MTP Draft (1-token)                                      │   │
│  │                                                          │   │
│  │  ┌─────────┐ ┌────────────┐ ┌─────────────────────────┐ │   │
│  │  │embed +  │→│eh_proj     │→│1× attention layer (F16) │ │   │
│  │  │2×norm   │ │(concat+FC) │ │Q/K/V/O + RoPE + softmax│ │   │
│  │  └─────────┘ └────────────┘ └─────────────────────────┘ │   │
│  │  ┌──────────┐ ┌──────────────────┐ ┌──────────────────┐ │   │
│  │  │rms_norm  │→│MMVQ restricted   │→│argmax (50K)     │ │   │
│  │  │(shared)  │ │lm_head (Q6_K,50K)│ │                  │ │   │
│  │  └──────────┘ └──────────────────┘ └──────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Result: 1.88 tokens/cycle avg (88.5% accept)                   │
│  Cycle time: ~2.7ms main + ~0.3ms MTP = ~3.0ms                 │
└─────────────────────────────────────────────────────────────────┘
```

## GPU Kernel Breakdown

Profiled 200 tokens, 88.5% acceptance rate, 89,536 kernel launches:

| Category | Time (ms) | % GPU | Calls | Notes |
|----------|-----------|-------|-------|-------|
| **Weight matmuls (MMVQ)** | **139.7** | **53.2%** | 20,208 | Bandwidth-bound, dominates |
| Quantize inputs (Q8_1) | 19.8 | 7.5% | 20,424 | FP32→Q8_1 before each MMVQ |
| DeltaNet element-wise | 25.8 | 9.8% | 10,548 | sigmoid, softplus, scale, mul, conv, concat |
| Memory ops | 20.2 | 7.7% | 10,920 | get_rows, set_rows, copy |
| Normalization | 15.1 | 5.7% | 9,288 | RMSNorm 1024 + 256 |
| MTP head (F16 matmuls) | 12.7 | 4.8% | 2,268 | MTP attention + FFN |
| Argmax | 8.5 | 3.2% | 216 | Full vocab (248K) and restricted (50K) |
| Other element-wise | 12.0 | 4.6% | 11,452 | add, SiLU |
| DeltaNet recurrence | 5.9 | 2.2% | 1,944 | The actual S matrix update |
| Attention misc | 3.0 | 1.1% | 2,268 | RoPE, softmax, convert |
| **Total kernel** | **262.6** | **100%** | 89,536 | |
| **Launch overhead + sync** | **170.3** | | | 39% of wall time (eliminated by CUDA graphs) |
| **Wall time** | **432.9** | | | 615 tok/s with CUDA graphs |

### Top Individual Kernels

| Kernel | Time (ms) | % GPU | Calls | Avg (μs) |
|--------|-----------|-------|-------|----------|
| MMVQ Q8_0 batch=2 | 127.5 | 48.6% | 19,448 | 6.6 |
| quantize_q8_1 | 19.8 | 7.5% | 20,424 | 1.0 |
| concat QKV | 16.5 | 6.3% | 1,872 | 8.8 |
| MTP F16 matmuls | 12.4 | 4.7% | 2,160 | 5.7 |
| get_rows (embed) | 10.7 | 4.1% | 4,104 | 2.6 |
| RMSNorm (1024) | 10.6 | 4.0% | 5,832 | 1.8 |
| argmax | 8.5 | 3.2% | 216 | 39.2 |
| elem add | 8.0 | 3.0% | 7,012 | 1.1 |
| copy | 7.9 | 3.0% | 5,304 | 1.5 |
| MMVQ Q6_K lm_head | 7.2 | 2.8% | 108 | 66.9 |
| DeltaNet | 5.9 | 2.2% | 1,944 | 3.0 |

## DRAM Throughput (ncu)

| Matrix | Rows | DRAM % | SM % | Occupancy % |
|--------|------|--------|------|-------------|
| FFN (gate/up/down, N=3584) | 3584 | 59% | 29% | 71% |
| attn_qkv fused (N=6144) | 6144 | 56% | 38% | 50% |
| attn_norm / Q/K (N=1024) | 1024 | 48% | 21% | 63% |
| ssm_out (N=2048) | 2048 | 35% | 23% | 43% |
| ssm_alpha/beta (N=16) | 16 | **0.8%** | 0.5% | 0.9% |

## Key Observations

### 1. MMVQ dominates at 53% — and it's bandwidth-bound

The Q8_0 MMVQ kernel at batch=2 accounts for nearly half of all GPU time. ncu shows 48-59% DRAM throughput utilization for the large matrices (N≥1024). The theoretical peak on the 5070 Ti is 896 GB/s; reaching 59% means ~530 GB/s effective. This is already good for a latency-bound single-token kernel.

### 2. The ssm_alpha/beta matmuls are 99% wasted

The DeltaNet has 18 layers × 2 matrices (alpha, beta) with only N=16 rows of K=1024 columns. At 0.8% DRAM throughput, these kernels launch 4 warps of 32 threads to process 16 × 1024 × 1 byte = 16 KB of data. The kernel launch overhead (~5μs) dwarfs the actual compute (~0.1μs). These should be fused into the DeltaNet kernel which already processes the same data.

### 3. DeltaNet element-wise ops are 10% of GPU time

sigmoid, softplus, scale, mul for the gating mechanism — 72 kernel launches per layer forward pass (4 ops × 18 DeltaNet layers), each processing [16]-element vectors. These are pure launch overhead; the actual computation is trivial. Fusing into the DeltaNet kernel would eliminate ~10,000 kernel launches.

### 4. CUDA graphs save 39% of wall time

With graphs disabled: 433ms (462 tok/s). With graphs: ~325ms (615 tok/s). The 170ms overhead comes from 89,536 individual kernel launches at ~1.9μs each. CUDA graph replay replaces all of these with a single `cudaGraphLaunch`.

### 5. Quantize + MMVQ is the true bottleneck pair

Every MMVQ call requires a preceding `quantize_q8_1` to convert the FP32 input activation to Q8_1 format. Together they account for **60.7%** of GPU time. The quantization step alone is 7.5% — a fused dequant-matmul that reads Q8_0 weights and FP32 activations directly would eliminate this.

### 6. MTP head is cheap

The entire MTP draft path (F16 matmuls + restricted LM head + argmax) is only 7.9% of GPU time. The 0.3ms per draft is well-spent given the 88.5% acceptance rate.

## Optimization Targets

| Target | Current % | Expected savings | Approach |
|--------|-----------|------------------|----------|
| Fuse gate/beta ops into DeltaNet | 9.8% + 1.2% | ~10% kernel time | Compute sigmoid/softplus/scale/mul inside the DeltaNet kernel |
| Fuse ssm_alpha/beta into DeltaNet | 1.2% | ~1% + launch overhead | Read alpha/beta weights directly in the DeltaNet kernel |
| Multi-token speculation (3-tok) | N/A | ~30% throughput gain | E[tokens] = p²+p+1 = 2.52 at 88% accept |

The MMVQ bandwidth ceiling (59% DRAM) cannot be improved without hardware changes or a fundamentally different kernel approach. The remaining optimization space is in fusing the DeltaNet-adjacent operations and extending speculation depth.

## Files Changed

| File | Change |
|------|--------|
| `blog/35-q8-kernel-profile.md` | This post |
| `blog/README.md` | Added entry |
