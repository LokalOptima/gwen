#!/usr/bin/env python3
"""Verify the LM head GEMV by computing dot products in Python."""
import struct
import numpy as np
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

def dequant_q6_k_row(data, n):
    """Dequantize one row (n elements) of Q6_K."""
    nb = n // 256
    result = np.zeros(n, dtype=np.float32)
    for b in range(nb):
        offset = b * 210
        ql = np.frombuffer(data[offset:offset+128], dtype=np.uint8)
        qh = np.frombuffer(data[offset+128:offset+192], dtype=np.uint8)
        scales = np.frombuffer(data[offset+192:offset+208], dtype=np.int8)
        d = np.frombuffer(data[offset+208:offset+210], dtype=np.float16)[0]
        for i in range(256):
            sg = i // 16
            sc = scales[sg]
            ql_idx = i // 2
            ql_nibble = (ql[ql_idx] & 0xF) if (i % 2 == 0) else (ql[ql_idx] >> 4)
            qh_idx = i // 4
            qh_shift = (i % 4) * 2
            qh_bits = (qh[qh_idx] >> qh_shift) & 0x3
            q_val = ql_nibble | (qh_bits << 4)
            result[b*256 + i] = float(d) * sc * (q_val - 32)
    return result

def main():
    # Load GWEN's final x_norm
    x_norm = np.fromfile("/tmp/gwen_x_norm.bin", dtype=np.float32)
    print(f"x_norm shape: {x_norm.shape}, norm: {np.linalg.norm(x_norm):.4f}")
    print(f"x_norm[:10]: {x_norm[:10]}")

    # Parse GGUF to get embedding tensor
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
            for d in dims: n_elems *= d
            tensors[name] = {"dims": dims, "type": ttype, "offset": offset, "n_elems": n_elems}
        data_start = ((f.tell() + 31) // 32) * 32

    import mmap
    fd = open(MODEL_PATH, "rb")
    mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)

    embd_info = tensors["token_embd.weight"]
    embd_start = data_start + embd_info["offset"]
    dim = 1024
    row_bytes = (dim // 256) * 210  # Q6_K: 4 blocks of 210 bytes

    # Verify dot products for specific tokens
    for token_id in [0, 6, 11, 138182]:
        row_start = embd_start + token_id * row_bytes
        row_data = mm[row_start:row_start + row_bytes]
        embd_row = dequant_q6_k_row(row_data, dim)
        logit = np.dot(embd_row, x_norm)
        print(f"  token={token_id}: Python logit={logit:.4f}, embd_norm={np.linalg.norm(embd_row):.4f}")

    # Also check what llama.cpp's top token 11 logit should be:
    # If Python gives ~12.73 → GWEN x_norm is wrong
    # If Python gives ~-0.69 → LM head matches GWEN, hidden state is wrong

    mm.close()
    fd.close()

if __name__ == "__main__":
    main()
