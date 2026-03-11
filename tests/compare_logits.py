#!/usr/bin/env python3
"""Compare GWEN and llama.cpp logits from binary dumps."""
import struct
import numpy as np

def load_logits(path):
    with open(path, 'rb') as f:
        n_vocab = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(n_vocab * 4), dtype=np.float32)
    return data

gwen = load_logits('/tmp/gwen_logits.bin')
llama = load_logits('/tmp/llama_logits.bin')

print(f"Vocab size: gwen={len(gwen)}, llama={len(llama)}")
assert len(gwen) == len(llama)

# Basic stats
diff = gwen - llama
print(f"\nLogit differences:")
print(f"  Mean absolute error: {np.abs(diff).mean():.4f}")
print(f"  Max absolute error:  {np.abs(diff).max():.4f}")
print(f"  RMS error:           {np.sqrt((diff**2).mean()):.4f}")
print(f"  Correlation:         {np.corrcoef(gwen, llama)[0,1]:.6f}")

# Top-10 comparison
gwen_top = np.argsort(gwen)[::-1][:20]
llama_top = np.argsort(llama)[::-1][:20]

print(f"\nTop-20 GWEN:  ", [(int(i), f"{gwen[i]:.3f}") for i in gwen_top])
print(f"Top-20 llama: ", [(int(i), f"{llama[i]:.3f}") for i in llama_top])

# Check overlap in top-20
gwen_set = set(gwen_top.tolist())
llama_set = set(llama_top.tolist())
overlap = gwen_set & llama_set
print(f"\nTop-20 overlap: {len(overlap)}/20 tokens")

# KL divergence (from softmax)
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

p_llama = softmax(llama)
p_gwen = softmax(gwen)
kl = np.sum(p_llama * np.log(p_llama / (p_gwen + 1e-30) + 1e-30))
print(f"KL(llama||gwen): {kl:.4f}")

# Check specific tokens
for tok in [220, 2614, 198, 387, 997, 1414, 3377, 321]:
    print(f"  token {tok:6d}: gwen={gwen[tok]:8.4f}  llama={llama[tok]:8.4f}  diff={gwen[tok]-llama[tok]:+.4f}")

# Scatter plot of logit diffs by magnitude
print(f"\nLogit range: gwen [{gwen.min():.2f}, {gwen.max():.2f}], llama [{llama.min():.2f}, {llama.max():.2f}]")

# Check if there's a systematic offset
print(f"\nMean logit: gwen={gwen.mean():.4f}, llama={llama.mean():.4f}")
print(f"Median diff: {np.median(diff):.4f}")
