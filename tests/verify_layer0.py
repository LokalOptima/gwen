#!/usr/bin/env python3
"""Verify full layer 0 forward pass against GWEN."""
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

BLOCK_SIZES = {0: 1, 8: 32, 12: 256, 13: 256, 14: 256}
BLOCK_BYTES = {0: 4, 8: 34, 12: 144, 13: 176, 14: 210}

def dequant_f32(data, n):
    return np.frombuffer(data[:n*4], dtype=np.float32).copy()

def dequant_q8_0(data, n):
    nb = n // 32
    result = np.zeros(n, dtype=np.float32)
    for b in range(nb):
        offset = b * 34
        d = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
        qs = np.frombuffer(data[offset+2:offset+34], dtype=np.int8)
        result[b*32:(b+1)*32] = float(d) * qs.astype(np.float32)
    return result

def dequant_q4_k(data, n):
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

def dequant_q6_k(data, n):
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

DEQUANT_FNS = {0: dequant_f32, 8: dequant_q8_0, 12: dequant_q4_k, 13: dequant_q5_k, 14: dequant_q6_k}

class GGUFModel:
    def __init__(self, path):
        with open(path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            assert magic == 0x46554747
            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            self.metadata = {}
            for _ in range(n_kv):
                key = read_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                val = read_value(f, vtype)
                self.metadata[key] = val
            self.tensors = {}
            for _ in range(n_tensors):
                name = read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                ttype = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                n_elems = 1
                for d in dims: n_elems *= d
                self.tensors[name] = {"dims": dims, "type": ttype, "offset": offset, "n_elems": n_elems}
            self.data_start = ((f.tell() + 31) // 32) * 32

        import mmap
        self._fd = open(path, "rb")
        self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)

    def get_raw(self, name):
        t = self.tensors[name]
        bs = BLOCK_SIZES[t["type"]]
        bb = BLOCK_BYTES[t["type"]]
        n_blocks = t["n_elems"] // bs
        nbytes = n_blocks * bb
        start = self.data_start + t["offset"]
        return self._mm[start:start+nbytes], t

    def get(self, name):
        data, t = self.get_raw(name)
        fn = DEQUANT_FNS[t["type"]]
        flat = fn(data, t["n_elems"])
        # Reshape: GGUF dims are [ne0, ne1, ...], ne0 is fastest
        # For 2D: [ne0, ne1] means ne1 rows of ne0 elements
        dims = t["dims"]
        if len(dims) == 1:
            return flat
        elif len(dims) == 2:
            return flat.reshape(dims[1], dims[0])  # [rows, cols]
        return flat

    def gemv(self, name, x):
        """Compute W @ x where W is [out_features, in_features]."""
        data, t = self.get_raw(name)
        dims = t["dims"]
        in_features = dims[0]
        out_features = dims[1]
        fn = DEQUANT_FNS[t["type"]]
        bs = BLOCK_SIZES[t["type"]]
        bb = BLOCK_BYTES[t["type"]]
        blocks_per_row = in_features // bs
        row_bytes = blocks_per_row * bb

        result = np.zeros(out_features, dtype=np.float32)
        for row in range(out_features):
            row_data = data[row * row_bytes:(row + 1) * row_bytes]
            row_vals = fn(row_data, in_features)
            result[row] = np.dot(row_vals, x)
        return result

def rmsnorm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x**2) + eps)
    return x / rms * weight

def silu(x):
    return x / (1.0 + np.exp(-x))

def main():
    model = GGUFModel(MODEL_PATH)
    token_id = 9419
    n_embed = 1024

    # 1. Embedding
    embd_data, embd_info = model.get_raw("token_embd.weight")
    row_bytes = (n_embed // 256) * 210
    embd = dequant_q6_k(embd_data[token_id * row_bytes:(token_id + 1) * row_bytes], n_embed)
    print(f"Embedding norm: {np.linalg.norm(embd):.4f}")
    print(f"embd[:5]: {embd[:5]}")

    # 2. RMSNorm
    norm_w = dequant_f32(model.get_raw("blk.0.attn_norm.weight")[0], n_embed)
    x_norm = rmsnorm(embd, norm_w)
    print(f"\nRMSNorm norm: {np.linalg.norm(x_norm):.4f}")
    print(f"x_norm[:5]: {x_norm[:5]}")

    # 3. QKV projection
    print("\nComputing QKV GEMV (6144 rows)...")
    qkv = model.gemv("blk.0.attn_qkv.weight", x_norm)
    print(f"QKV norm: {np.linalg.norm(qkv):.4f}")
    print(f"qkv[:5]: {qkv[:5]}")

    # 4. Gate projection
    print("\nComputing gate GEMV (2048 rows)...")
    gate_z = model.gemv("blk.0.attn_gate.weight", x_norm)
    print(f"gate_z norm: {np.linalg.norm(gate_z):.4f}")
    print(f"gate_z[:5]: {gate_z[:5]}")

    # 5. Conv1d (first token, state=0)
    conv_data, conv_info = model.get_raw("blk.0.ssm_conv1d.weight")
    conv_w = dequant_f32(conv_data, 4 * 6144).reshape(6144, 4)
    conv_out = conv_w[:, 3] * qkv
    print(f"\nConv1d[:5]: {conv_out[:5]}")

    # 6. SiLU
    silu_out = silu(conv_out)
    print(f"SiLU[:5]: {silu_out[:5]}")

    # 7. Split Q, K, V
    q = silu_out[:2048].copy()
    k = silu_out[2048:4096].copy()
    v = silu_out[4096:6144].copy()

    # 8. L2 normalize Q and K per head
    for h in range(16):
        q[h*128:(h+1)*128] /= (np.linalg.norm(q[h*128:(h+1)*128]) + 1e-12)
        k[h*128:(h+1)*128] /= (np.linalg.norm(k[h*128:(h+1)*128]) + 1e-12)

    # 9. Gate and beta (Q8_0 GEMV)
    alpha_data, alpha_info = model.get_raw("blk.0.ssm_alpha.weight")
    beta_data, beta_info = model.get_raw("blk.0.ssm_beta.weight")
    ssm_a = dequant_f32(model.get_raw("blk.0.ssm_a")[0], 16)
    dt_bias = dequant_f32(model.get_raw("blk.0.ssm_dt.bias")[0], 16)

    # alpha/beta projections: [16, 1024] @ x_norm → [16]
    alpha_full = dequant_q8_0(alpha_data, 16 * n_embed).reshape(16, n_embed)
    beta_full = dequant_q8_0(beta_data, 16 * n_embed).reshape(16, n_embed)

    gates = np.zeros(16, dtype=np.float32)
    betas = np.zeros(16, dtype=np.float32)
    for h in range(16):
        alpha_proj = np.dot(alpha_full[h], x_norm)
        beta_proj = np.dot(beta_full[h], x_norm)
        sp = np.log(1.0 + np.exp(alpha_proj + dt_bias[h]))  # softplus
        gates[h] = ssm_a[h] * sp
        betas[h] = 1.0 / (1.0 + np.exp(-beta_proj))  # sigmoid

    print(f"\ngates[:4]: {gates[:4]}")
    print(f"betas[:4]: {betas[:4]}")

    # 10. DeltaNet (state=0)
    # output_h[j] = (k_h . q_h) * beta_h * v_h[j]
    attn_out = np.zeros(2048, dtype=np.float32)
    for h in range(16):
        k_h = k[h*128:(h+1)*128]
        q_h = q[h*128:(h+1)*128]
        v_h = v[h*128:(h+1)*128]
        dot_kq = np.dot(k_h, q_h)
        attn_out[h*128:(h+1)*128] = dot_kq * betas[h] * v_h

    print(f"\nDeltaNet output[:10]: {attn_out[:10]}")

    # 11. Gated RMSNorm
    ssm_norm_w = dequant_f32(model.get_raw("blk.0.ssm_norm.weight")[0], 128)
    gated_out = np.zeros(2048, dtype=np.float32)
    for h in range(16):
        x_h = attn_out[h*128:(h+1)*128]
        g_h = gate_z[h*128:(h+1)*128]
        normed = rmsnorm(x_h, ssm_norm_w)
        silu_g = silu(g_h)
        gated_out[h*128:(h+1)*128] = normed * silu_g

    print(f"Gated RMSNorm[:10]: {gated_out[:10]}")

    # 12. Output projection
    print("\nComputing output projection (1024 rows)...")
    out_proj = model.gemv("blk.0.ssm_out.weight", gated_out)
    print(f"out_proj[:10]: {out_proj[:10]}")

    # 13. Residual
    x = out_proj + embd
    print(f"\nAfter residual (attn + embed):")
    print(f"x[:10]: {x[:10]}")
    print(f"x norm: {np.linalg.norm(x):.4f}")

    # 14. Post-attention RMSNorm
    post_norm_w = dequant_f32(model.get_raw("blk.0.post_attention_norm.weight")[0], n_embed)
    x_norm2 = rmsnorm(x, post_norm_w)
    print(f"\nPost-attn RMSNorm x_norm2[:5]: {x_norm2[:5]}")

    # 15. FFN
    print("\nComputing FFN gate (3584 rows)...")
    ffn_gate_out = model.gemv("blk.0.ffn_gate.weight", x_norm2)
    print("Computing FFN up (3584 rows)...")
    ffn_up_out = model.gemv("blk.0.ffn_up.weight", x_norm2)
    ffn_swiglu = silu(ffn_gate_out) * ffn_up_out
    print(f"SwiGLU[:5]: {ffn_swiglu[:5]}")

    print("Computing FFN down (1024 rows)...")
    ffn_down_out = model.gemv("blk.0.ffn_down.weight", ffn_swiglu)
    print(f"FFN down[:5]: {ffn_down_out[:5]}")

    # 16. Residual
    x = ffn_down_out + x
    print(f"\nFinal layer 0 output:")
    print(f"x[:10]: {x[:10]}")
    print(f"x norm: {np.linalg.norm(x):.4f}")

    # Compare with GWEN:
    # GWEN layer 0 after residual: x norm=3.0592
    # GWEN layer 0 x[:10]: ???

    # Also compute early logit for this x
    output_norm_w = dequant_f32(model.get_raw("output_norm.weight")[0], n_embed)
    x_final_norm = rmsnorm(x, output_norm_w)

    # Logit for token 0 and token 11
    for tid in [0, 11]:
        embd_row = dequant_q6_k(embd_data[tid * row_bytes:(tid + 1) * row_bytes], n_embed)
        logit = np.dot(embd_row, x_final_norm)
        print(f"  Early logit[{tid}] after layer 0: {logit:.4f}")

if __name__ == "__main__":
    main()
