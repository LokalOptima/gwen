#!/usr/bin/env python3
"""Verify multi-layer forward pass against GWEN, layer by layer."""
import struct
import time
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

def dequant_q8_0_fast(data, n):
    """Fast vectorized Q8_0 dequantization."""
    nb = n // 32
    raw = np.frombuffer(data[:nb*34], dtype=np.uint8).reshape(nb, 34)
    d = raw[:, :2].view(np.float16).astype(np.float32).reshape(nb, 1)
    qs = raw[:, 2:34].view(np.int8).astype(np.float32)
    return (d * qs).reshape(-1)[:n]

def dequant_q4_k_fast(data, n):
    """Faster Q4_K dequantization using partial vectorization."""
    nb = n // 256
    result = np.zeros(n, dtype=np.float32)
    for b in range(nb):
        offset = b * 144
        d = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
        dmin = np.frombuffer(data[offset+2:offset+4], dtype=np.float16)[0]
        sc_raw = np.frombuffer(data[offset+4:offset+16], dtype=np.uint8)
        qs = np.frombuffer(data[offset+16:offset+144], dtype=np.uint8)

        # Precompute scales and mins for all 8 sub-blocks
        scales = np.zeros(8, dtype=np.float32)
        mins = np.zeros(8, dtype=np.float32)
        for sb in range(8):
            if sb < 4:
                sc_lo = int(sc_raw[sb]) & 0x3F
                m_lo = int(sc_raw[sb + 4]) & 0x3F
            else:
                sc_lo = (int(sc_raw[sb + 4]) & 0xF) | ((int(sc_raw[sb - 4]) >> 6) << 4)
                m_lo = (int(sc_raw[sb + 4]) >> 4) | ((int(sc_raw[sb]) >> 6) << 4)
            scales[sb] = float(d) * sc_lo
            mins[sb] = float(dmin) * m_lo

        # Extract 4-bit values
        # Low half (elements 0-127): qs[0:64], two nibbles each
        lo_bytes = qs[:64]
        lo_even = (lo_bytes & 0xF).astype(np.float32)
        lo_odd = (lo_bytes >> 4).astype(np.float32)
        lo = np.empty(128, dtype=np.float32)
        lo[0::2] = lo_even
        lo[1::2] = lo_odd

        # High half (elements 128-255): qs[64:128]
        hi_bytes = qs[64:128]
        hi_even = (hi_bytes & 0xF).astype(np.float32)
        hi_odd = (hi_bytes >> 4).astype(np.float32)
        hi = np.empty(128, dtype=np.float32)
        hi[0::2] = hi_even
        hi[1::2] = hi_odd

        # Apply scales and mins per sub-block (32 elements each)
        for sb in range(4):
            s, e = sb * 32, (sb + 1) * 32
            result[b*256 + s:b*256 + e] = scales[sb] * lo[s:e] - mins[sb]
        for sb in range(4, 8):
            s, e = (sb-4) * 32, (sb-3) * 32
            result[b*256 + 128 + s:b*256 + 128 + e] = scales[sb] * hi[s:e] - mins[sb]

    return result

def dequant_q5_k_fast(data, n):
    nb = n // 256
    result = np.zeros(n, dtype=np.float32)
    for b in range(nb):
        offset = b * 176
        d = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
        dmin = np.frombuffer(data[offset+2:offset+4], dtype=np.float16)[0]
        sc_raw = np.frombuffer(data[offset+4:offset+16], dtype=np.uint8)
        qh = np.frombuffer(data[offset+16:offset+48], dtype=np.uint8)
        qs = np.frombuffer(data[offset+48:offset+176], dtype=np.uint8)

        scales = np.zeros(8, dtype=np.float32)
        mins = np.zeros(8, dtype=np.float32)
        for sb in range(8):
            if sb < 4:
                sc_lo = int(sc_raw[sb]) & 0x3F
                m_lo = int(sc_raw[sb + 4]) & 0x3F
            else:
                sc_lo = (int(sc_raw[sb + 4]) & 0xF) | ((int(sc_raw[sb - 4]) >> 6) << 4)
                m_lo = (int(sc_raw[sb + 4]) >> 4) | ((int(sc_raw[sb]) >> 6) << 4)
            scales[sb] = float(d) * sc_lo
            mins[sb] = float(dmin) * m_lo

        for i in range(256):
            sb = i // 32
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
            result[b*256 + i] = scales[sb] * q_val - mins[sb]
    return result

def dequant_q6_k_fast(data, n):
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

DEQUANT_FNS = {0: dequant_f32, 8: dequant_q8_0_fast, 12: dequant_q4_k_fast, 13: dequant_q5_k_fast, 14: dequant_q6_k_fast}

class GGUFModel:
    def __init__(self, path):
        with open(path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            assert magic == 0x46554747
            struct.unpack("<I", f.read(4))[0]  # version
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
                for dd in dims: n_elems *= dd
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

    def get_f32(self, name):
        data, t = self.get_raw(name)
        assert t["type"] == 0
        return np.frombuffer(data[:t["n_elems"]*4], dtype=np.float32).copy()

    def gemv(self, name, x):
        """Compute W @ x where W is stored row-major in quantized format."""
        data, t = self.get_raw(name)
        dims = t["dims"]
        in_features = dims[0]
        out_features = dims[1]
        fn = DEQUANT_FNS[t["type"]]
        bs = BLOCK_SIZES[t["type"]]
        bb = BLOCK_BYTES[t["type"]]
        row_bytes = (in_features // bs) * bb

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
    return x / (1.0 + np.exp(-np.clip(x, -80, 80)))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))

def softplus(x):
    return np.log(1.0 + np.exp(np.clip(x, -80, 80)))

FULL_ATTN_LAYERS = {3, 7, 11, 15, 19, 23}

def compute_deltanet_layer(model, x, layer_idx):
    prefix = f"blk.{layer_idx}."
    n_embed = 1024
    ssm_inner = 2048
    ssm_n_heads = 16
    ssm_state_size = 128

    # RMSNorm
    norm_w = model.get_f32(prefix + "attn_norm.weight")
    x_norm = rmsnorm(x, norm_w)

    # QKV projection
    qkv = model.gemv(prefix + "attn_qkv.weight", x_norm)

    # Gate projection
    gate_z = model.gemv(prefix + "attn_gate.weight", x_norm)

    # Conv1d (first token, state=0)
    conv_data, _ = model.get_raw(prefix + "ssm_conv1d.weight")
    conv_w = dequant_f32(conv_data, 4 * 6144).reshape(6144, 4)
    qkv = conv_w[:, 3] * qkv  # only weight[3] * input for first token

    # SiLU
    qkv = silu(qkv)

    # Split
    q = qkv[:ssm_inner].copy()
    k = qkv[ssm_inner:2*ssm_inner].copy()
    v = qkv[2*ssm_inner:3*ssm_inner].copy()

    # L2 normalize Q and K
    for h in range(ssm_n_heads):
        s = h * ssm_state_size
        e = s + ssm_state_size
        q[s:e] /= (np.linalg.norm(q[s:e]) + 1e-12)
        k[s:e] /= (np.linalg.norm(k[s:e]) + 1e-12)

    # Gate and beta
    ssm_a = model.get_f32(prefix + "ssm_a")
    dt_bias = model.get_f32(prefix + "ssm_dt.bias")
    alpha_data, alpha_info = model.get_raw(prefix + "ssm_alpha.weight")
    beta_data, beta_info = model.get_raw(prefix + "ssm_beta.weight")
    alpha_full = dequant_q8_0_fast(alpha_data, ssm_n_heads * n_embed).reshape(ssm_n_heads, n_embed)
    beta_full = dequant_q8_0_fast(beta_data, ssm_n_heads * n_embed).reshape(ssm_n_heads, n_embed)

    gates = np.zeros(ssm_n_heads, dtype=np.float32)
    betas = np.zeros(ssm_n_heads, dtype=np.float32)
    for h in range(ssm_n_heads):
        alpha_proj = np.dot(alpha_full[h], x_norm)
        beta_proj = np.dot(beta_full[h], x_norm)
        sp = softplus(alpha_proj + dt_bias[h])
        gates[h] = ssm_a[h] * sp
        betas[h] = sigmoid(beta_proj)

    # DeltaNet (state=0, first token)
    attn_out = np.zeros(ssm_inner, dtype=np.float32)
    for h in range(ssm_n_heads):
        s = h * ssm_state_size
        e = s + ssm_state_size
        dot_kq = np.dot(k[s:e], q[s:e])
        attn_out[s:e] = dot_kq * betas[h] * v[s:e]

    # Gated RMSNorm
    ssm_norm_w = model.get_f32(prefix + "ssm_norm.weight")
    gated_out = np.zeros(ssm_inner, dtype=np.float32)
    for h in range(ssm_n_heads):
        s = h * ssm_state_size
        e = s + ssm_state_size
        normed = rmsnorm(attn_out[s:e], ssm_norm_w)
        silu_g = silu(gate_z[s:e])
        gated_out[s:e] = normed * silu_g

    # Output projection
    out_proj = model.gemv(prefix + "ssm_out.weight", gated_out)

    # Residual
    x = out_proj + x

    # FFN
    post_norm_w = model.get_f32(prefix + "post_attention_norm.weight")
    x_norm2 = rmsnorm(x, post_norm_w)

    ffn_gate_out = model.gemv(prefix + "ffn_gate.weight", x_norm2)
    ffn_up_out = model.gemv(prefix + "ffn_up.weight", x_norm2)
    ffn_swiglu = silu(ffn_gate_out) * ffn_up_out
    ffn_down_out = model.gemv(prefix + "ffn_down.weight", ffn_swiglu)

    x = ffn_down_out + x
    return x

def compute_full_attn_layer(model, x, layer_idx):
    prefix = f"blk.{layer_idx}."
    n_embed = 1024
    n_head = 8
    n_head_kv = 2
    head_dim = 256

    # RMSNorm
    norm_w = model.get_f32(prefix + "attn_norm.weight")
    x_norm = rmsnorm(x, norm_w)

    # Q+gate projection (interleaved output)
    q_gate = model.gemv(prefix + "attn_q.weight", x_norm)
    # Deinterleave: [Q_h0(256), gate_h0(256), Q_h1(256), gate_h1(256), ...]
    q_all = np.zeros(n_head * head_dim, dtype=np.float32)
    gate_all = np.zeros(n_head * head_dim, dtype=np.float32)
    for h in range(n_head):
        q_all[h*head_dim:(h+1)*head_dim] = q_gate[h*head_dim*2 : h*head_dim*2+head_dim]
        gate_all[h*head_dim:(h+1)*head_dim] = q_gate[h*head_dim*2+head_dim : (h+1)*head_dim*2]

    # K and V projections
    k_all = model.gemv(prefix + "attn_k.weight", x_norm)
    v_all = model.gemv(prefix + "attn_v.weight", x_norm)

    # Per-head Q/K RMSNorm
    q_norm_w = model.get_f32(prefix + "attn_q_norm.weight")
    k_norm_w = model.get_f32(prefix + "attn_k_norm.weight")
    for h in range(n_head):
        s, e = h * head_dim, (h+1) * head_dim
        q_all[s:e] = rmsnorm(q_all[s:e], q_norm_w)
    for h in range(n_head_kv):
        s, e = h * head_dim, (h+1) * head_dim
        k_all[s:e] = rmsnorm(k_all[s:e], k_norm_w)

    # RoPE at pos=0 → identity, skip

    # Attention at pos=0 with seq_len=1: attn_out = V for each head
    # GQA: Q heads 0-3 → KV head 0, Q heads 4-7 → KV head 1
    attn_out = np.zeros(n_head * head_dim, dtype=np.float32)
    for qh in range(n_head):
        kv_h = qh // (n_head // n_head_kv)
        attn_out[qh*head_dim:(qh+1)*head_dim] = v_all[kv_h*head_dim:(kv_h+1)*head_dim]

    # Gated attention: output = attn_out * sigmoid(gate)
    gated_out = attn_out * sigmoid(gate_all)

    # Output projection
    out_proj = model.gemv(prefix + "attn_output.weight", gated_out)

    # Residual
    x = out_proj + x

    # FFN
    post_norm_w = model.get_f32(prefix + "post_attention_norm.weight")
    x_norm2 = rmsnorm(x, post_norm_w)

    ffn_gate_out = model.gemv(prefix + "ffn_gate.weight", x_norm2)
    ffn_up_out = model.gemv(prefix + "ffn_up.weight", x_norm2)
    ffn_swiglu = silu(ffn_gate_out) * ffn_up_out
    ffn_down_out = model.gemv(prefix + "ffn_down.weight", ffn_swiglu)

    x = ffn_down_out + x
    return x

def main():
    model = GGUFModel(MODEL_PATH)
    n_embed = 1024
    token_id = 9419

    # Embedding
    embd_data, embd_info = model.get_raw("token_embd.weight")
    row_bytes = (n_embed // 256) * 210
    x = dequant_q6_k_fast(embd_data[token_id * row_bytes:(token_id + 1) * row_bytes], n_embed)
    print(f"Embedding norm: {np.linalg.norm(x):.4f}")

    # Early logit helper
    output_norm_w = model.get_f32("output_norm.weight")
    def early_logits(x_cur):
        xn = rmsnorm(x_cur, output_norm_w)
        logit0 = np.dot(dequant_q6_k_fast(embd_data[0:row_bytes], n_embed), xn)
        logit11 = np.dot(dequant_q6_k_fast(embd_data[11*row_bytes:12*row_bytes], n_embed), xn)
        return logit0, logit11

    # Process layers
    for layer_idx in range(24):
        t0 = time.time()
        if layer_idx in FULL_ATTN_LAYERS:
            x = compute_full_attn_layer(model, x, layer_idx)
            layer_type = "FullAttn"
        else:
            x = compute_deltanet_layer(model, x, layer_idx)
            layer_type = "DeltaNet"
        t1 = time.time()

        l0, l11 = early_logits(x)
        print(f"[Python] After {layer_type} layer {layer_idx}: x norm={np.linalg.norm(x):.4f}, "
              f"early_logit[0]={l0:.2f}, early_logit[11]={l11:.2f} ({t1-t0:.1f}s)")

if __name__ == "__main__":
    main()
