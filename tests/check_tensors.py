#!/usr/bin/env python3
"""Check tensor dimensions and key weight values in GGUF."""
import struct, numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "Qwen3.5-0.8B-Q4_K_M.gguf"

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

TYPE_NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K"}

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

# Print key tensor info for layer 0 (DeltaNet)
print("=== Layer 0 (DeltaNet) tensors ===")
for name in sorted(tensors.keys()):
    if name.startswith("blk.0.") or name in ("token_embd.weight", "output_norm.weight"):
        t = tensors[name]
        tname = TYPE_NAMES.get(t["type"], f"type_{t['type']}")
        print(f"  {name}: dims={t['dims']}, type={tname}, n_elems={t['n_elems']}")

# Print layer 3 (FullAttn) tensors too
print("\n=== Layer 3 (FullAttn) tensors ===")
for name in sorted(tensors.keys()):
    if name.startswith("blk.3."):
        t = tensors[name]
        tname = TYPE_NAMES.get(t["type"], f"type_{t['type']}")
        print(f"  {name}: dims={t['dims']}, type={tname}, n_elems={t['n_elems']}")

# Check if there's an output.weight tensor
print(f"\n'output.weight' exists: {'output.weight' in tensors}")
print(f"'token_embd.weight' dims: {tensors['token_embd.weight']['dims']}")
