#!/usr/bin/env python3
"""Load GWEN's dumped x_norm and compute logits in F32 Python to verify."""
import struct, numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "Qwen3.5-0.8B-Q4_K_M.gguf"

# Load GWEN's dumped x_norm
x_norm = np.fromfile("/tmp/gwen_x_norm.bin", dtype=np.float32)
x_final = np.fromfile("/tmp/gwen_x_final.bin", dtype=np.float32)
print(f"x_norm shape: {x_norm.shape}")
print(f"x_norm[:10]: {' '.join(f'{v:.6f}' for v in x_norm[:10])}")
print(f"x_norm norm: {np.linalg.norm(x_norm):.4f}")
print(f"x_final[:10]: {' '.join(f'{v:.6f}' for v in x_final[:10])}")
print(f"x_final norm: {np.linalg.norm(x_final):.4f}")

# Now independently apply RMSNorm to x_final and verify it matches x_norm
def read_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")

def read_value(f, vtype):
    if vtype == 0: return struct.unpack("<B", f.read(1))[0]
    elif vtype == 1: return struct.unpack("<b", f.read(1))[0]
    elif vtype == 4: return struct.unpack("<I", f.read(4))[0]
    elif vtype == 5: return struct.unpack("<i", f.read(4))[0]
    elif vtype == 6: return struct.unpack("<f", f.read(4))[0]
    elif vtype == 7: return struct.unpack("<B", f.read(1))[0] != 0
    elif vtype == 8: return read_string(f)
    elif vtype == 9:
        atype = struct.unpack("<I", f.read(4))[0]
        alen = struct.unpack("<Q", f.read(8))[0]
        return [read_value(f, atype) for _ in range(alen)]
    elif vtype == 10: return struct.unpack("<Q", f.read(8))[0]
    elif vtype == 11: return struct.unpack("<q", f.read(8))[0]
    elif vtype == 12: return struct.unpack("<d", f.read(8))[0]
    else: raise ValueError(f"Unknown type: {vtype}")

with open(MODEL_PATH, "rb") as f:
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    metadata = {}
    for _ in range(n_kv):
        key = read_string(f)
        vtype = struct.unpack("<I", f.read(4))[0]
        val = read_value(f, vtype)
        metadata[key] = val
    tensors = {}
    for _ in range(n_tensors):
        name = read_string(f)
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]
        n_elems = 1
        for dd in dims: n_elems *= dd
        tensors[name] = {"dims": dims, "type": ttype, "offset": offset, "n_elems": n_elems}
    data_start = ((f.tell() + 31) // 32) * 32

import mmap
fd = open(MODEL_PATH, "rb")
mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)

# Load output_norm weight
t = tensors["output_norm.weight"]
raw = mm[data_start + t["offset"]:data_start + t["offset"] + t["n_elems"] * 4]
output_norm_w = np.frombuffer(raw, dtype=np.float32).copy()
print(f"\noutput_norm_w[:5]: {' '.join(f'{v:.4f}' for v in output_norm_w[:5])}")
print(f"output_norm_w mean: {np.mean(output_norm_w):.4f}")

# Verify: RMSNorm(x_final) should give x_norm
rms = np.sqrt(np.mean(x_final**2) + 1e-6)
x_norm_check = x_final / rms * output_norm_w
print(f"\nRMSNorm verification (should match x_norm):")
print(f"  check[:10]: {' '.join(f'{v:.6f}' for v in x_norm_check[:10])}")
print(f"  diff max: {np.max(np.abs(x_norm - x_norm_check)):.8f}")

# Load embedding data for computing logits
embd_t = tensors["token_embd.weight"]
embd_start = data_start + embd_t["offset"]
n_embed = 1024

def dequant_q6_k_row(data):
    ql = np.frombuffer(data[0:128], dtype=np.uint8)
    qh = np.frombuffer(data[128:192], dtype=np.uint8)
    scales = np.frombuffer(data[192:208], dtype=np.int8)
    d = float(np.frombuffer(data[208:210], dtype=np.float16)[0])
    result = np.zeros(256, dtype=np.float32)
    for i in range(256):
        sg = i // 16
        sc = int(scales[sg])
        ql_idx = i // 2
        ql_nibble = int(ql[ql_idx] & 0xF) if (i % 2 == 0) else int(ql[ql_idx] >> 4)
        qh_idx = i // 4
        qh_shift = (i % 4) * 2
        qh_bits = (int(qh[qh_idx]) >> qh_shift) & 0x3
        q_val = ql_nibble | (qh_bits << 4)
        result[i] = d * sc * (q_val - 32)
    return result

row_bytes = (n_embed // 256) * 210  # 4 blocks * 210 bytes = 840

# Compute logits for selected tokens using GWEN's x_norm
for tok in [0, 11, 138182]:
    tok_start = embd_start + tok * row_bytes
    row = np.zeros(n_embed, dtype=np.float32)
    for b in range(4):
        blk = mm[tok_start + b*210 : tok_start + (b+1)*210]
        row[b*256:(b+1)*256] = dequant_q6_k_row(blk)
    logit = np.dot(row, x_norm)
    print(f"Logit[{tok}] (using GWEN x_norm): {logit:.4f}")

# Also compute using the verified x_norm_check
print("\nUsing verified x_norm_check:")
for tok in [0, 11, 138182]:
    tok_start = embd_start + tok * row_bytes
    row = np.zeros(n_embed, dtype=np.float32)
    for b in range(4):
        blk = mm[tok_start + b*210 : tok_start + (b+1)*210]
        row[b*256:(b+1)*256] = dequant_q6_k_row(blk)
    logit = np.dot(row, x_norm_check)
    print(f"Logit[{tok}] (using F32 RMSNorm): {logit:.4f}")

# Top-5 using GWEN's x_norm (sample 1000 random tokens + known ones)
print("\nComputing top tokens from GWEN's x_norm...")
import random
candidates = list(range(100)) + [138182, 83369, 221501, 92659, 94189, 58212, 26479, 218209, 153727, 159508, 11, 9419]
candidates += random.sample(range(248320), 500)
candidates = sorted(set(candidates))

scored = []
for tok in candidates:
    tok_start = embd_start + tok * row_bytes
    row = np.zeros(n_embed, dtype=np.float32)
    for b in range(4):
        blk = mm[tok_start + b*210 : tok_start + (b+1)*210]
        row[b*256:(b+1)*256] = dequant_q6_k_row(blk)
    logit = np.dot(row, x_norm)
    scored.append((logit, tok))

scored.sort(reverse=True)
print("Top-10 (from candidates):")
for logit, tok in scored[:10]:
    print(f"  token={tok} logit={logit:.4f}")
