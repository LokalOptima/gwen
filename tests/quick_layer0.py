#!/usr/bin/env python3
"""Quick test: compute layer 0 intermediate values and compare with GWEN debug output."""
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

BLOCK_SIZES = {0: 1, 8: 32, 12: 256, 13: 256, 14: 256}
BLOCK_BYTES = {0: 4, 8: 34, 12: 144, 13: 176, 14: 210}

def dequant_f32(data, n):
    return np.frombuffer(data[:n*4], dtype=np.float32).copy()

def dequant_q8_0_fast(data, n):
    nb = n // 32
    raw = np.frombuffer(data[:nb*34], dtype=np.uint8).reshape(nb, 34)
    d = raw[:, :2].view(np.float16).astype(np.float32).reshape(nb, 1)
    qs = raw[:, 2:34].view(np.int8).astype(np.float32)
    return (d * qs).reshape(-1)[:n]

def dequant_q4_k_row(data):
    """Dequantize one Q4_K block (256 elements) from 144 bytes."""
    d = np.frombuffer(data[0:2], dtype=np.float16).astype(np.float32)[0]
    dmin = np.frombuffer(data[2:4], dtype=np.float16).astype(np.float32)[0]
    sc_raw = np.frombuffer(data[4:16], dtype=np.uint8)
    qs = np.frombuffer(data[16:144], dtype=np.uint8)

    result = np.zeros(256, dtype=np.float32)
    for sb in range(8):
        if sb < 4:
            sc_lo = int(sc_raw[sb]) & 0x3F
            m_lo = int(sc_raw[sb + 4]) & 0x3F
        else:
            sc_lo = (int(sc_raw[sb + 4]) & 0xF) | ((int(sc_raw[sb - 4]) >> 6) << 4)
            m_lo = (int(sc_raw[sb + 4]) >> 4) | ((int(sc_raw[sb]) >> 6) << 4)
        scale = float(d) * sc_lo
        min_val = float(dmin) * m_lo

        if sb < 4:
            for i in range(32):
                idx = sb * 32 + i
                q_byte = qs[idx // 2]
                q_val = (q_byte & 0xF) if (idx % 2 == 0) else (q_byte >> 4)
                result[idx] = scale * q_val - min_val
        else:
            for i in range(32):
                idx = (sb - 4) * 32 + i + 128
                q_byte = qs[idx // 2]
                q_val = (q_byte & 0xF) if ((idx - 128) % 2 == 0) else (q_byte >> 4)
                result[idx] = scale * q_val - min_val
    return result

def dequant_q5_k_row(data):
    """Dequantize one Q5_K block (256 elements) from 176 bytes."""
    d = np.frombuffer(data[0:2], dtype=np.float16).astype(np.float32)[0]
    dmin = np.frombuffer(data[2:4], dtype=np.float16).astype(np.float32)[0]
    sc_raw = np.frombuffer(data[4:16], dtype=np.uint8)
    qh = np.frombuffer(data[16:48], dtype=np.uint8)
    qs = np.frombuffer(data[48:176], dtype=np.uint8)

    result = np.zeros(256, dtype=np.float32)
    for sb in range(8):
        if sb < 4:
            sc_lo = int(sc_raw[sb]) & 0x3F
            m_lo = int(sc_raw[sb + 4]) & 0x3F
        else:
            sc_lo = (int(sc_raw[sb + 4]) & 0xF) | ((int(sc_raw[sb - 4]) >> 6) << 4)
            m_lo = (int(sc_raw[sb + 4]) >> 4) | ((int(sc_raw[sb]) >> 6) << 4)
        scale = float(d) * sc_lo
        min_val = float(dmin) * m_lo

    for i in range(256):
        sb = i // 32
        if sb < 4:
            sc_lo = int(sc_raw[sb]) & 0x3F
            m_lo = int(sc_raw[sb + 4]) & 0x3F
        else:
            sc_lo = (int(sc_raw[sb + 4]) & 0xF) | ((int(sc_raw[sb - 4]) >> 6) << 4)
            m_lo = (int(sc_raw[sb + 4]) >> 4) | ((int(sc_raw[sb]) >> 6) << 4)
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
        q_hi = (int(qh[qh_byte_idx]) >> qh_bit_idx) & 1
        q_val = q_lo | (q_hi << 4)
        result[i] = scale * q_val - min_val
    return result

def dequant_q6_k_row(data):
    """Dequantize one Q6_K block (256 elements) from 210 bytes."""
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

# GGUF type to dequant block function
DEQUANT_BLOCK = {
    0: lambda d: np.frombuffer(d[:256*4], dtype=np.float32)[:256].copy(),  # F32 - 256 elements
    12: dequant_q4_k_row,
    13: dequant_q5_k_row,
    14: dequant_q6_k_row,
}

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

    def gemv_row(self, name, x, row_idx):
        """Compute one row of W @ x."""
        data, t = self.get_raw(name)
        in_features = t["dims"][0]
        bs = BLOCK_SIZES[t["type"]]
        bb = BLOCK_BYTES[t["type"]]
        blocks_per_row = in_features // bs
        row_bytes = blocks_per_row * bb

        offset = row_idx * row_bytes
        row_data = data[offset:offset + row_bytes]

        # Dequantize row
        dequant_fn = DEQUANT_BLOCK[t["type"]]
        row_vals = np.zeros(in_features, dtype=np.float32)
        for b in range(blocks_per_row):
            blk_data = row_data[b*bb:(b+1)*bb]
            row_vals[b*bs:(b+1)*bs] = dequant_fn(blk_data)

        return np.dot(row_vals, x)

    def gemv(self, name, x):
        """Full GEMV: W @ x."""
        data, t = self.get_raw(name)
        in_features = t["dims"][0]
        out_features = t["dims"][1]
        bs = BLOCK_SIZES[t["type"]]
        bb = BLOCK_BYTES[t["type"]]
        blocks_per_row = in_features // bs
        row_bytes = blocks_per_row * bb

        dequant_fn = DEQUANT_BLOCK[t["type"]]
        result = np.zeros(out_features, dtype=np.float32)

        for r in range(out_features):
            offset = r * row_bytes
            row_data = data[offset:offset + row_bytes]
            row_vals = np.zeros(in_features, dtype=np.float32)
            for b in range(blocks_per_row):
                blk_data = row_data[b*bb:(b+1)*bb]
                row_vals[b*bs:(b+1)*bs] = dequant_fn(blk_data)
            result[r] = np.dot(row_vals, x)

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

# ============================================================

model = GGUFModel(MODEL_PATH)
n_embed = 1024
ssm_inner = 2048
ssm_n_heads = 16
ssm_state_size = 128
token_id = 9419

# Step 1: Embedding
embd_data, embd_info = model.get_raw("token_embd.weight")
row_bytes = (n_embed // 256) * 210
x = np.zeros(n_embed, dtype=np.float32)
for b in range(4):
    blk_data = embd_data[token_id * row_bytes + b*210 : token_id * row_bytes + (b+1)*210]
    x[b*256:(b+1)*256] = dequant_q6_k_row(blk_data)

print(f"Embedding: {' '.join(f'{v:.6f}' for v in x[:10])}")
print(f"  norm: {np.linalg.norm(x):.6f}")

# Step 2: RMSNorm
norm_w = model.get_f32("blk.0.attn_norm.weight")
x_norm = rmsnorm(x, norm_w)
print(f"\nRMSNorm: {' '.join(f'{v:.6f}' for v in x_norm[:10])}")
print(f"  norm: {np.linalg.norm(x_norm):.6f}")

# Step 3: QKV projection (first 10 rows only for speed, then also compute row 0)
import time
print(f"\nComputing first few QKV rows...")
t0 = time.time()
qkv_first5 = np.array([model.gemv_row("blk.0.attn_qkv.weight", x_norm, r) for r in range(5)])
t1 = time.time()
print(f"QKV[0:5]: {' '.join(f'{v:.6f}' for v in qkv_first5)}")
print(f"  (took {t1-t0:.2f}s for 5 rows)")

# Full QKV
print(f"\nComputing full QKV (6144 rows)...")
t0 = time.time()
qkv = model.gemv("blk.0.attn_qkv.weight", x_norm)
t1 = time.time()
print(f"QKV[:10]: {' '.join(f'{v:.6f}' for v in qkv[:10])}")
print(f"QKV norm: {np.linalg.norm(qkv):.6f}")
print(f"  (took {t1-t0:.1f}s)")

# Step 4: Gate/Z projection
print(f"\nComputing gate Z projection (2048 rows)...")
gate_z = model.gemv("blk.0.attn_gate.weight", x_norm)
print(f"gate_z[:5]: {' '.join(f'{v:.6f}' for v in gate_z[:5])}")

# Step 5: Conv1d
conv_data, _ = model.get_raw("blk.0.ssm_conv1d.weight")
conv_w = dequant_f32(conv_data, 4 * 6144).reshape(6144, 4)
qkv = conv_w[:, 3] * qkv
print(f"\nAfter conv1d: {' '.join(f'{v:.6f}' for v in qkv[:10])}")

# Step 6: SiLU
qkv = silu(qkv)
print(f"After SiLU: {' '.join(f'{v:.6f}' for v in qkv[:10])}")

# Step 7: Split
Q = qkv[:ssm_inner].copy()
K = qkv[ssm_inner:2*ssm_inner].copy()
V = qkv[2*ssm_inner:3*ssm_inner].copy()
print(f"\nQ[:5]: {' '.join(f'{v:.6f}' for v in Q[:5])}")
print(f"K[:5]: {' '.join(f'{v:.6f}' for v in K[:5])}")
print(f"V[:5]: {' '.join(f'{v:.6f}' for v in V[:5])}")

# Step 8: L2 normalize
for h in range(ssm_n_heads):
    s, e = h*ssm_state_size, (h+1)*ssm_state_size
    Q[s:e] /= max(np.linalg.norm(Q[s:e]), 1e-6)
    K[s:e] /= max(np.linalg.norm(K[s:e]), 1e-6)
print(f"\nQ_norm[:5]: {' '.join(f'{v:.6f}' for v in Q[:5])}")
print(f"K_norm[:5]: {' '.join(f'{v:.6f}' for v in K[:5])}")

# Step 9: Gate/beta
ssm_a = model.get_f32("blk.0.ssm_a")
dt_bias = model.get_f32("blk.0.ssm_dt.bias")
alpha_data, _ = model.get_raw("blk.0.ssm_alpha.weight")
beta_data, _ = model.get_raw("blk.0.ssm_beta.weight")
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

print(f"\nssm_a[:4]: {' '.join(f'{v:.6f}' for v in ssm_a[:4])}")
print(f"dt_bias[:4]: {' '.join(f'{v:.6f}' for v in dt_bias[:4])}")
print(f"gates[:4]: {' '.join(f'{v:.6f}' for v in gates[:4])}")
print(f"betas[:4]: {' '.join(f'{v:.6f}' for v in betas[:4])}")

# Step 10: DeltaNet (state=0)
attn_out = np.zeros(ssm_inner, dtype=np.float32)
for h in range(ssm_n_heads):
    s, e = h*ssm_state_size, (h+1)*ssm_state_size
    dot_kq = np.dot(K[s:e], Q[s:e])
    attn_out[s:e] = dot_kq * betas[h] * V[s:e]
print(f"\nattn_out[:10]: {' '.join(f'{v:.6f}' for v in attn_out[:10])}")

# Step 11: Gated RMSNorm
ssm_norm_w = model.get_f32("blk.0.ssm_norm.weight")
gated_out = np.zeros(ssm_inner, dtype=np.float32)
for h in range(ssm_n_heads):
    s, e = h*ssm_state_size, (h+1)*ssm_state_size
    normed = rmsnorm(attn_out[s:e], ssm_norm_w)
    silu_g = silu(gate_z[s:e])
    gated_out[s:e] = normed * silu_g
print(f"gated_out[:10]: {' '.join(f'{v:.6f}' for v in gated_out[:10])}")

# Step 12: Output projection
print(f"\nComputing output projection (1024 rows)...")
out_proj = model.gemv("blk.0.ssm_out.weight", gated_out)
print(f"out_proj[:10]: {' '.join(f'{v:.6f}' for v in out_proj[:10])}")

# Step 13: Residual
x = x + out_proj
print(f"\nAfter residual: {' '.join(f'{v:.6f}' for v in x[:10])}")
print(f"  norm: {np.linalg.norm(x):.6f}")

# Early logit check
output_norm_w = model.get_f32("output_norm.weight")
xn = rmsnorm(x, output_norm_w)
logit0 = np.dot(np.zeros(n_embed), xn)  # placeholder
logit11 = np.dot(np.zeros(n_embed), xn)  # placeholder

# Compute logit[0] and logit[11] from token_embd
for b in range(4):
    row0 = dequant_q6_k_row(embd_data[0*row_bytes + b*210 : 0*row_bytes + (b+1)*210])
    row11 = dequant_q6_k_row(embd_data[11*row_bytes + b*210 : 11*row_bytes + (b+1)*210])
    logit0 += np.dot(row0, xn[b*256:(b+1)*256])
    logit11 += np.dot(row11, xn[b*256:(b+1)*256])
print(f"\nEarly logit[0]: {logit0:.4f}")
print(f"Early logit[11]: {logit11:.4f}")
print("\n=== Compare with GWEN ===")
print("GWEN embed: 0.015656 -0.028915 -0.019272 0.007229 -0.028915 -0.008430 0.001204 0.003614 -0.010841 0.013252")
print("GWEN x_norm: 1.654297 -1.977539 -1.562500 0.666504 -2.076172 -0.581055 0.080383 0.246460 -0.734863 0.849121")
print("GWEN qkv: 0.506348 -0.277832 0.562500 0.210938 0.618164 0.002024 0.324951 -0.441406 1.380859 -0.154541")
print("GWEN conv: -0.037567 0.001526 0.002087 0.048615 -0.049500 -0.000009 0.023163 -0.029526 -0.109924 0.013359")
print("GWEN silu: -0.018433 0.000763 0.001044 0.024902 -0.024139 -0.000004 0.011719 -0.014542 -0.051941 0.006725")
print("GWEN gate: -1.062671 -0.000890 -0.005490 -4.908503")
print("GWEN beta: 0.693206 0.499433 0.591787 0.595164")
print("GWEN out_proj: 0.245483 0.023636 -0.010986 0.064514 -0.037933 0.008942 -0.165527 -0.078308 0.157104 0.090027")
print("GWEN x_res: 0.261230 -0.005280 -0.030258 0.071716 -0.066833 0.000511 -0.164307 -0.074707 0.146240 0.103271")
print("GWEN early_logit[0]=-3.99, early_logit[11]=-3.29")
