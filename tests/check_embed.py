#!/usr/bin/env python3
"""Quick check: dequantize embedding for token 9419 from GGUF and print."""
import struct, numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "Qwen3.5-0.8B-Q4_K_M.gguf"

def read_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")

def read_value(f, vtype):
    if vtype == 0: return struct.unpack("<B", f.read(1))[0]
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
    elif vtype == 1: return struct.unpack("<b", f.read(1))[0]
    else: raise ValueError(f"Unknown type: {vtype}")

with open(MODEL_PATH, "rb") as f:
    magic = struct.unpack("<I", f.read(4))[0]
    assert magic == 0x46554747
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

t = tensors["token_embd.weight"]
print(f"token_embd.weight: dims={t['dims']}, type={t['type']}, n_elems={t['n_elems']}")

# Type 14 = Q6_K, 210 bytes per block of 256 elements
assert t["type"] == 14
n_embed = t["dims"][0]  # 1024
vocab_size = t["dims"][1]  # 248320
blocks_per_row = n_embed // 256  # 4
row_bytes = blocks_per_row * 210  # 840

token_id = 9419
row_start = data_start + t["offset"] + token_id * row_bytes
row_data = mm[row_start:row_start + row_bytes]

# Dequantize Q6_K
result = np.zeros(n_embed, dtype=np.float32)
for b in range(blocks_per_row):
    offset = b * 210
    ql = np.frombuffer(row_data[offset:offset+128], dtype=np.uint8)
    qh = np.frombuffer(row_data[offset+128:offset+192], dtype=np.uint8)
    scales = np.frombuffer(row_data[offset+192:offset+208], dtype=np.int8)
    d = float(np.frombuffer(row_data[offset+208:offset+210], dtype=np.float16)[0])

    for i in range(256):
        sg = i // 16
        sc = int(scales[sg])
        ql_idx = i // 2
        ql_nibble = int(ql[ql_idx] & 0xF) if (i % 2 == 0) else int(ql[ql_idx] >> 4)
        qh_idx = i // 4
        qh_shift = (i % 4) * 2
        qh_bits = int(qh[qh_idx] >> qh_shift) & 0x3
        q_val = ql_nibble | (qh_bits << 4)
        result[b * 256 + i] = d * sc * (q_val - 32)

print(f"\nPython Q6_K dequant for token {token_id}:")
print(f"  first 10: {' '.join(f'{v:.6f}' for v in result[:10])}")
print(f"  norm: {np.linalg.norm(result):.6f}")

# Also check: what does llama.cpp's dequant look like?
# Show raw bytes for first block to help debug
print(f"\nRaw first block bytes:")
b0 = row_data[:210]
print(f"  ql[0:8]: {' '.join(f'{x:02x}' for x in b0[:8])}")
print(f"  qh[0:4]: {' '.join(f'{x:02x}' for x in b0[128:132])}")
print(f"  scales[0:4]: {' '.join(f'{x:02x}' for x in b0[192:196])}")
print(f"  d_fp16: {b0[208]:02x} {b0[209]:02x} = {float(np.frombuffer(b0[208:210], dtype=np.float16)[0]):.8f}")

# Element-by-element for first 10
print(f"\nElement-by-element dequant for first 10:")
for i in range(10):
    ql_idx = i // 2
    ql_byte = int(b0[ql_idx])
    ql_nibble = (ql_byte & 0xF) if (i % 2 == 0) else (ql_byte >> 4)
    qh_idx = i // 4
    qh_byte = int(b0[128 + qh_idx])
    qh_shift = (i % 4) * 2
    qh_bits = (qh_byte >> qh_shift) & 0x3
    q_val = ql_nibble | (qh_bits << 4)
    sg = i // 16
    sc = int(np.frombuffer(b0[192+sg:193+sg], dtype=np.int8)[0])
    d = float(np.frombuffer(b0[208:210], dtype=np.float16)[0])
    val = d * sc * (q_val - 32)
    print(f"  [{i}]: ql_byte={ql_byte:02x} ql_nib={ql_nibble} qh_byte={qh_byte:02x} qh_bits={qh_bits} q_val={q_val} sc={sc} d={d:.8f} → {val:.6f}")
