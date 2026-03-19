#!/usr/bin/env python3
"""Debug: compute first layer output using Python and compare against gwen.

This script:
1. Reads the GGUF file
2. Dequantizes the embedding and first layer weights
3. Runs a reference forward pass for the first DeltaNet layer
4. Prints intermediate values for comparison
"""

import struct
import sys
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


def dequant_f32(data, n):
    return np.frombuffer(data[:n*4], dtype=np.float32)


def dequant_q8_0(data, n):
    """Dequantize Q8_0: 34 bytes per 32 values."""
    nb = n // 32
    result = np.zeros(n, dtype=np.float32)
    for b in range(nb):
        offset = b * 34
        d = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
        qs = np.frombuffer(data[offset+2:offset+34], dtype=np.int8)
        result[b*32:(b+1)*32] = float(d) * qs.astype(np.float32)
    return result


def dequant_q6_k(data, n):
    """Dequantize Q6_K: 210 bytes per 256 values."""
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


def dequant_q4_k(data, n):
    """Dequantize Q4_K: 144 bytes per 256 values."""
    nb = n // 256
    result = np.zeros(n, dtype=np.float32)
    for b in range(nb):
        offset = b * 144
        d = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
        dmin = np.frombuffer(data[offset+2:offset+4], dtype=np.float16)[0]
        sc_raw = np.frombuffer(data[offset+4:offset+16], dtype=np.uint8)
        qs = np.frombuffer(data[offset+16:offset+144], dtype=np.uint8)

        for i in range(256):
            sb = i // 32
            if sb < 4:
                sc_lo = sc_raw[sb] & 0x3F
                m_lo = sc_raw[sb + 4] & 0x3F
            else:
                sc_lo = (sc_raw[sb + 4] & 0xF) | ((sc_raw[sb - 4] >> 6) << 4)
                m_lo = (sc_raw[sb + 4] >> 4) | ((sc_raw[sb] >> 6) << 4)
            scale = float(d) * sc_lo
            min_val = float(dmin) * m_lo

            if i < 128:
                q_byte = qs[i // 2]
                q_val = (q_byte & 0xF) if (i % 2 == 0) else (q_byte >> 4)
            else:
                q_byte = qs[(i - 128) // 2 + 64]
                q_val = (q_byte & 0xF) if ((i - 128) % 2 == 0) else (q_byte >> 4)

            result[b*256 + i] = scale * q_val - min_val
    return result


def dequant_q5_k(data, n):
    """Dequantize Q5_K: 176 bytes per 256 values."""
    nb = n // 256
    result = np.zeros(n, dtype=np.float32)
    for b in range(nb):
        offset = b * 176
        d = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
        dmin = np.frombuffer(data[offset+2:offset+4], dtype=np.float16)[0]
        sc_raw = np.frombuffer(data[offset+4:offset+16], dtype=np.uint8)
        qh = np.frombuffer(data[offset+16:offset+48], dtype=np.uint8)
        qs = np.frombuffer(data[offset+48:offset+176], dtype=np.uint8)

        for i in range(256):
            sb = i // 32
            if sb < 4:
                sc_lo = sc_raw[sb] & 0x3F
                m_lo = sc_raw[sb + 4] & 0x3F
            else:
                sc_lo = (sc_raw[sb + 4] & 0xF) | ((sc_raw[sb - 4] >> 6) << 4)
                m_lo = (sc_raw[sb + 4] >> 4) | ((sc_raw[sb] >> 6) << 4)
            scale = float(d) * sc_lo
            min_val = float(dmin) * m_lo

            if i < 128:
                q_byte = qs[i // 2]
                q_lo = (q_byte & 0xF) if (i % 2 == 0) else (q_byte >> 4)
            else:
                q_byte = qs[(i - 128) // 2 + 64]
                q_lo = (q_byte & 0xF) if ((i - 128) % 2 == 0) else (q_byte >> 4)

            qh_byte_idx = i // 8
            qh_bit_idx = i % 8
            q_hi = (qh[qh_byte_idx] >> qh_bit_idx) & 1
            q_val = q_lo | (q_hi << 4)

            result[b*256 + i] = scale * q_val - min_val
    return result


DEQUANT_FNS = {
    0: dequant_f32,
    8: dequant_q8_0,
    12: dequant_q4_k,
    13: dequant_q5_k,
    14: dequant_q6_k,
}

BLOCK_SIZES = {0: 1, 8: 32, 12: 256, 13: 256, 14: 256}
BLOCK_BYTES = {0: 4, 8: 34, 12: 144, 13: 176, 14: 210}


def main():
    with open(MODEL_PATH, "rb") as f:
        # Parse header
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x46554747
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        # Parse metadata
        metadata = {}
        for _ in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            val = read_value(f, vtype)
            metadata[key] = val

        # Parse tensor info
        tensors = {}
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            n_elems = 1
            for d in dims:
                n_elems *= d
            tensors[name] = {"dims": dims, "type": ttype, "offset": offset, "n_elems": n_elems}

        # Data starts at 32-byte alignment after header
        data_start = ((f.tell() + 31) // 32) * 32

    # Memory-map and dequant needed tensors
    import mmap
    fd = open(MODEL_PATH, "rb")
    mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)

    def get_tensor_data(name):
        t = tensors[name]
        bs = BLOCK_SIZES[t["type"]]
        bb = BLOCK_BYTES[t["type"]]
        n_blocks = t["n_elems"] // bs
        nbytes = n_blocks * bb
        start = data_start + t["offset"]
        return mm[start:start+nbytes], t

    def dequant_tensor(name):
        data, t = get_tensor_data(name)
        fn = DEQUANT_FNS[t["type"]]
        return fn(data, t["n_elems"]).reshape([t["dims"][i] for i in range(len(t["dims"]))][::-1])

    # Token 9419 = "Hello"
    token_id = 9419

    # 1. Embedding lookup
    embd_data, embd_info = get_tensor_data("token_embd.weight")
    # Embedding is [1024, 248320] → ne[0]=1024, ne[1]=248320
    # Row `token_id` is at offset token_id * 1024 elements
    # For Q6_K: 1024 elements = 4 blocks of 256 = 4 * 210 = 840 bytes
    row_blocks = 1024 // 256
    row_bytes = row_blocks * 210
    row_start = token_id * row_bytes
    embd_row = dequant_q6_k(embd_data[row_start:row_start+row_bytes], 1024)
    print(f"Embedding[{token_id}] first 10 values: {embd_row[:10]}")
    print(f"  norm: {np.linalg.norm(embd_row):.4f}")

    # 2. RMSNorm with blk.0.attn_norm.weight
    norm_w = dequant_f32(get_tensor_data("blk.0.attn_norm.weight")[0], 1024)
    rms = np.sqrt(np.mean(embd_row**2) + 1e-6)
    x_norm = embd_row / rms * norm_w
    print(f"\nAfter RMSNorm (layer 0) first 10: {x_norm[:10]}")
    print(f"  norm: {np.linalg.norm(x_norm):.4f}")

    # 3. FFN gate projection to verify GEMV (skip attention for now)
    # ffn_gate: [1024, 3584] → 3584 rows of 1024 elements
    print("\n--- Testing FFN gate GEMV ---")
    ffn_gate_data, ffn_gate_info = get_tensor_data("blk.0.ffn_gate.weight")
    # Dequant first 3 rows
    for row_idx in range(3):
        row_bytes_q4k = (1024 // 256) * 144  # 4 Q4_K blocks
        row_data = ffn_gate_data[row_idx * row_bytes_q4k : (row_idx + 1) * row_bytes_q4k]
        row_vals = dequant_q4_k(row_data, 1024)
        dot = np.dot(row_vals, x_norm)
        print(f"  ffn_gate[{row_idx}] @ x_norm = {dot:.6f}")

    # 4. QKV projection
    print("\n--- Testing QKV projection (DeltaNet layer 0) ---")
    qkv_data, qkv_info = get_tensor_data("blk.0.attn_qkv.weight")
    # attn_qkv: [1024, 6144] → 6144 rows of 1024 elements
    # Q5_K: 1024 elements = 4 blocks of 256 = 4 * 176 = 704 bytes per row
    row_bytes_q5k = (1024 // 256) * 176
    for row_idx in range(3):
        row_data = qkv_data[row_idx * row_bytes_q5k : (row_idx + 1) * row_bytes_q5k]
        row_vals = dequant_q5_k(row_data, 1024)
        dot = np.dot(row_vals, x_norm)
        print(f"  qkv[{row_idx}] @ x_norm = {dot:.6f}")

    # Full QKV
    qkv_out = np.zeros(6144, dtype=np.float32)
    for row_idx in range(6144):
        row_data = qkv_data[row_idx * row_bytes_q5k : (row_idx + 1) * row_bytes_q5k]
        row_vals = dequant_q5_k(row_data, 1024)
        qkv_out[row_idx] = np.dot(row_vals, x_norm)

    print(f"\n  QKV output first 10: {qkv_out[:10]}")
    print(f"  QKV output norm: {np.linalg.norm(qkv_out):.4f}")

    # Conv1d (first token, state=0, just weight[3,:] * input)
    conv_data, conv_info = get_tensor_data("blk.0.ssm_conv1d.weight")
    conv_w = dequant_f32(conv_data, 4 * 6144).reshape(6144, 4)  # [dim, kernel]
    conv_out = conv_w[:, 3] * qkv_out  # Only last kernel position (state is 0)
    print(f"\n  Conv1d output first 10: {conv_out[:10]}")

    # SiLU
    silu_out = conv_out * (1.0 / (1.0 + np.exp(-conv_out)))
    print(f"  SiLU output first 10: {silu_out[:10]}")

    # Split Q, K, V
    q = silu_out[:2048]
    k = silu_out[2048:4096]
    v = silu_out[4096:6144]
    print(f"\n  Q[:5]: {q[:5]}")
    print(f"  K[:5]: {k[:5]}")
    print(f"  V[:5]: {v[:5]}")

    # L2 normalize Q and K (per head, 16 heads × 128 dims)
    for h in range(16):
        q[h*128:(h+1)*128] /= (np.linalg.norm(q[h*128:(h+1)*128]) + 1e-12)
        k[h*128:(h+1)*128] /= (np.linalg.norm(k[h*128:(h+1)*128]) + 1e-12)

    print(f"  Q normalized[:5]: {q[:5]}")

    mm.close()
    fd.close()
    print("\nDone. Use these values to verify GWEN's intermediate outputs.")


if __name__ == "__main__":
    main()
