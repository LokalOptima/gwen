#!/usr/bin/env python3
"""
Prepare all model artifacts from HF safetensors (single source of truth).

Steps:
  1. Convert HF safetensors → F16 GGUF  (llama.cpp convert_hf_to_gguf.py)
  2. Quantize F16 → Q4_K_M              (llama-quantize)
  3. Patch ssm_alpha/beta to Q8_0        (GWEN kernels require Q8_0 for these)
  4. Extract Q6K embeddings for training (data/embed_tokens_q6k.npy)
  5. Extract output norm for training    (data/output_norm.npy)
  6. Cross-validate extracts vs HF safetensors

Usage:
    uv run scripts/02_prepare_models.py
    uv run scripts/02_prepare_models.py --hf-dir ~/models/hf/Qwen3.5-0.8B-Base
"""

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LLAMA_CPP = Path.home() / "git" / "llama.cpp"
CONVERT_SCRIPT = LLAMA_CPP / "convert_hf_to_gguf.py"
QUANTIZE_BIN = LLAMA_CPP / "build" / "bin" / "llama-quantize"
GGUF_DIR = Path.home() / "models" / "gguf"
DATA_DIR = Path("data")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def fail(msg: str) -> None:
    log(f"FATAL: {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# GGUF helpers (minimal parser)
# ---------------------------------------------------------------------------

# (block_elements, block_bytes) — matches GWEN's GGMLType enum
_TYPE_BLOCK = {
    0:  (1, 4),     # F32
    1:  (1, 2),     # F16
    2:  (32, 20),   # Q4_0
    3:  (32, 22),   # Q4_1
    6:  (32, 24),   # Q5_0
    7:  (32, 26),   # Q5_1
    8:  (32, 34),   # Q8_0
    9:  (32, 36),   # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    15: (256, 292), # Q8_K
}
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q6_K = 14


def _tensor_size(ggml_type, n_elements):
    if ggml_type not in _TYPE_BLOCK:
        raise ValueError(f"Unknown GGML type {ggml_type}")
    blk_els, blk_bytes = _TYPE_BLOCK[ggml_type]
    assert n_elements % blk_els == 0
    return (n_elements // blk_els) * blk_bytes


def _skip_gguf_value(f, vtype):
    if vtype in (0, 1, 7): f.read(1)
    elif vtype in (2, 3): f.read(2)
    elif vtype in (4, 5, 6): f.read(4)
    elif vtype in (10, 11, 12): f.read(8)
    elif vtype == 8:
        slen = struct.unpack("<Q", f.read(8))[0]
        f.read(slen)
    elif vtype == 9:
        atype = struct.unpack("<I", f.read(4))[0]
        alen = struct.unpack("<Q", f.read(8))[0]
        for _ in range(alen):
            _skip_gguf_value(f, atype)
    else:
        raise ValueError(f"Unknown GGUF value type {vtype}")


def _skip_gguf_value_offset(buf, off, vtype):
    if vtype in (0, 1, 7): return off + 1
    elif vtype in (2, 3): return off + 2
    elif vtype in (4, 5, 6): return off + 4
    elif vtype in (10, 11, 12): return off + 8
    elif vtype == 8:
        slen = struct.unpack_from("<Q", buf, off)[0]
        return off + 8 + slen
    elif vtype == 9:
        arr_type = struct.unpack_from("<I", buf, off)[0]; off += 4
        arr_len = struct.unpack_from("<Q", buf, off)[0]; off += 8
        for _ in range(arr_len):
            off = _skip_gguf_value_offset(buf, off, arr_type)
        return off
    else:
        raise ValueError(f"Unknown GGUF value type {vtype}")


def read_gguf_tensor(gguf_path, tensor_name):
    """Read a single named tensor from GGUF. Returns (ggml_type, dims, raw_bytes)."""
    with open(gguf_path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x46554747, f"Bad GGUF magic: {magic:#x}"
        _ = struct.unpack("<I", f.read(4))[0]  # version
        n_tensors, n_meta = struct.unpack("<QQ", f.read(16))

        for _ in range(n_meta):
            klen = struct.unpack("<Q", f.read(8))[0]
            f.read(klen)
            vtype = struct.unpack("<I", f.read(4))[0]
            _skip_gguf_value(f, vtype)

        found = None
        for _ in range(n_tensors):
            nlen = struct.unpack("<Q", f.read(8))[0]
            name = f.read(nlen).decode()
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            if name == tensor_name:
                found = (dtype, dims, offset)

        if found is None:
            raise KeyError(f"Tensor {tensor_name!r} not found in {gguf_path}")

        dtype, dims, offset = found
        n_el = 1
        for d in dims:
            n_el *= d
        size = _tensor_size(dtype, n_el)
        data_start = ((f.tell() + 31) // 32) * 32
        f.seek(data_start + offset)
        raw = f.read(size)
    return dtype, dims, raw


def read_gguf_tensors_all(path):
    """Read ALL tensors from GGUF. Returns dict of name -> (type, dims, raw_bytes)."""
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x46554747
        _ = struct.unpack("<I", f.read(4))[0]
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]
            f.read(klen)
            vtype = struct.unpack("<I", f.read(4))[0]
            _skip_gguf_value(f, vtype)

        tensor_infos = []
        for _ in range(n_tensors):
            nlen = struct.unpack("<Q", f.read(8))[0]
            name = f.read(nlen).decode()
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ggml_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensor_infos.append((name, dims, ggml_type, offset))

        data_start = ((f.tell() + 31) // 32) * 32
        tensors = {}
        for name, dims, ggml_type, offset in tensor_infos:
            n_el = 1
            for d in dims:
                n_el *= d
            size = _tensor_size(ggml_type, n_el)
            f.seek(data_start + offset)
            tensors[name] = (ggml_type, dims, f.read(size))
    return tensors


# ---------------------------------------------------------------------------
# Q6_K dequantization
# ---------------------------------------------------------------------------

def dequant_q6k(raw, n_rows, n_cols):
    """Vectorized Q6_K dequant → FP16 matrix."""
    blocks_per_row = n_cols // 256
    n_blocks = n_rows * blocks_per_row
    all_blocks = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 210)

    ql_all = all_blocks[:, :128]
    qh_all = all_blocks[:, 128:192]
    sc_all = all_blocks[:, 192:208].view(np.int8)
    d_all = all_blocks[:, 208:210].view(np.float16).astype(np.float32).ravel()

    result = np.zeros((n_blocks, 256), dtype=np.float32)
    for tid in range(256):
        half_idx = tid // 128
        j = tid % 128
        quarter = j // 32
        pos = j % 32

        ql_byte_idx = half_idx * 64 + (quarter & 1) * 32 + pos
        if quarter >= 2:
            ql_nibble = (ql_all[:, ql_byte_idx] >> 4).astype(np.int32)
        else:
            ql_nibble = (ql_all[:, ql_byte_idx] & 0xF).astype(np.int32)

        qh_byte_idx = half_idx * 32 + pos
        qh_bits = ((qh_all[:, qh_byte_idx] >> (quarter * 2)) & 0x3).astype(np.int32)

        q_val = ql_nibble | (qh_bits << 4)
        scale_idx = half_idx * 8 + quarter * 2 + pos // 16
        scale = sc_all[:, scale_idx].astype(np.float32)
        result[:, tid] = d_all * scale * (q_val - 32)

    return result.reshape(n_rows, n_cols).astype(np.float16)


# ---------------------------------------------------------------------------
# Q8_0 quantization
# ---------------------------------------------------------------------------

def quantize_q8_0(values):
    """Quantize float32 array to Q8_0 format bytes."""
    assert len(values) % 32 == 0
    n_blocks = len(values) // 32
    result = bytearray()
    for i in range(n_blocks):
        block = values[i * 32 : (i + 1) * 32]
        amax = np.max(np.abs(block))
        d = amax / 127.0 if amax > 0 else 0.0
        d_f16 = np.float16(d)
        d_f32 = float(d_f16)
        if d_f32 > 0:
            qs = np.clip(np.round(block / d_f32), -128, 127).astype(np.int8)
        else:
            qs = np.zeros(32, dtype=np.int8)
        result += struct.pack("<e", d_f16)
        result += qs.tobytes()
    return bytes(result)


# ---------------------------------------------------------------------------
# GGUF patching (ssm_alpha/beta Q4_K → Q8_0)
# ---------------------------------------------------------------------------

def patch_ssm_weights(q4_path, f16_path, out_path):
    """Replace ssm_alpha/beta in Q4_K_M GGUF with Q8_0 from F16 GGUF."""
    log(f"Reading F16 GGUF for ssm weights: {f16_path}")
    f16_tensors = read_gguf_tensors_all(str(f16_path))
    targets = [n for n in f16_tensors if "ssm_alpha" in n or "ssm_beta" in n]
    log(f"  Found {len(targets)} tensors to patch")

    q8_data = {}
    for name in targets:
        ggml_type, dims, raw = f16_tensors[name]
        n_el = 1
        for d in dims:
            n_el *= d
        f32_vals = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        q8_data[name] = quantize_q8_0(f32_vals)

    log(f"Reading Q4_K_M GGUF: {q4_path}")
    with open(str(q4_path), "rb") as f:
        src = f.read()

    off = 0
    off += 4  # magic
    off += 4  # version
    n_tensors = struct.unpack_from("<Q", src, off)[0]; off += 8
    n_kv = struct.unpack_from("<Q", src, off)[0]; off += 8
    header = bytearray(src[:off])

    kv_start = off
    for _ in range(n_kv):
        klen = struct.unpack_from("<Q", src, off)[0]; off += 8
        off += klen
        vtype = struct.unpack_from("<I", src, off)[0]; off += 4
        off = _skip_gguf_value_offset(src, off, vtype)
    kv_data = src[kv_start:off]

    tensor_infos = []
    for _ in range(n_tensors):
        name_len = struct.unpack_from("<Q", src, off)[0]; off += 8
        name = src[off:off+name_len].decode(); off += name_len
        n_dims = struct.unpack_from("<I", src, off)[0]; off += 4
        dims = [struct.unpack_from("<Q", src, off + i*8)[0] for i in range(n_dims)]
        off += n_dims * 8
        ggml_type = struct.unpack_from("<I", src, off)[0]; off += 4
        data_offset = struct.unpack_from("<Q", src, off)[0]; off += 8
        tensor_infos.append({"name": name, "dims": dims,
                             "ggml_type": ggml_type, "data_offset": data_offset})

    orig_data_start = ((off + 31) // 32) * 32

    for ti in tensor_infos:
        n_el = 1
        for d in ti["dims"]:
            n_el *= d
        size = _tensor_size(ti["ggml_type"], n_el)
        ti["data"] = src[orig_data_start + ti["data_offset"]:
                         orig_data_start + ti["data_offset"] + size]

    patched = 0
    for ti in tensor_infos:
        if ti["name"] in q8_data:
            ti["ggml_type"] = GGML_TYPE_Q8_0
            ti["data"] = q8_data[ti["name"]]
            patched += 1

    # Rebuild file
    out = bytearray()
    out += header
    out += kv_data

    ti_section = bytearray()
    current_offset = 0
    for ti in tensor_infos:
        current_offset = ((current_offset + 31) // 32) * 32
        ti_section += struct.pack("<Q", len(ti["name"]))
        ti_section += ti["name"].encode()
        ti_section += struct.pack("<I", len(ti["dims"]))
        for d in ti["dims"]:
            ti_section += struct.pack("<Q", d)
        ti_section += struct.pack("<I", ti["ggml_type"])
        ti_section += struct.pack("<Q", current_offset)
        ti["new_offset"] = current_offset
        current_offset += len(ti["data"])

    out += ti_section
    new_data_start = ((len(out) + 31) // 32) * 32
    out += b"\x00" * (new_data_start - len(out))

    for ti in tensor_infos:
        target = new_data_start + ti["new_offset"]
        if len(out) < target:
            out += b"\x00" * (target - len(out))
        out += ti["data"]

    with open(str(out_path), "wb") as f:
        f.write(out)

    log(f"  Patched {patched} tensors → {out_path} ({len(out) / 1024 / 1024:.1f} MB)")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(embed_path, norm_path, hf_dir):
    """Verify GGUF-extracted weights match HF safetensors."""
    from safetensors import safe_open

    log("Cross-validating against HF safetensors...")
    sf_path = next(hf_dir.glob("*.safetensors"))

    with safe_open(str(sf_path), framework="pt") as f:
        for k in f.keys():
            if "embed_tokens" in k:
                hf_embed = f.get_tensor(k).float().numpy()
                break
        for k in f.keys():
            if k in ("model.norm.weight", "model.language_model.norm.weight"):
                hf_norm = f.get_tensor(k).float().numpy() + 1.0  # Qwen additive offset
                break

    gguf_embed = np.load(str(embed_path))
    gguf_norm = np.load(str(norm_path))

    # Embedding check
    test_tokens = [0, 100, 760, 3841, 13477, 50000, 100000, 200000]
    cos_vals = []
    for tok in test_tokens:
        a = gguf_embed[tok].astype(np.float32)
        b = hf_embed[tok]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        cos_vals.append(cos)

    min_cos = min(cos_vals)
    log(f"  Embeddings: min cos={min_cos:.6f} across {len(test_tokens)} tokens")
    if min_cos < 0.99:
        fail(f"Embedding cross-validation FAILED (min cos={min_cos:.6f})")

    # Norm check
    norm_diff = np.max(np.abs(gguf_norm - hf_norm))
    log(f"  Output norm: max diff={norm_diff:.8f}")
    if norm_diff > 1e-4:
        fail(f"Output norm cross-validation FAILED (max diff={norm_diff:.8f})")

    log("  ✓ Cross-validation passed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare model artifacts from HF safetensors")
    parser.add_argument("--hf-dir", type=Path,
                        default=Path.home() / "models" / "hf" / "Qwen3.5-0.8B-Base",
                        help="HF safetensors directory")
    parser.add_argument("--gguf-dir", type=Path, default=GGUF_DIR,
                        help="Output directory for GGUFs")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Output directory for training data extracts")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip HF→GGUF conversion (use existing F16 GGUF)")
    args = parser.parse_args()

    hf_dir = args.hf_dir
    gguf_dir = args.gguf_dir
    data_dir = args.data_dir

    if not hf_dir.exists():
        fail(f"HF directory not found: {hf_dir}\n"
             f"Download with: uvx hf download Qwen/Qwen3.5-0.8B-Base --local-dir {hf_dir}")

    gguf_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    f16_gguf = gguf_dir / "Qwen3.5-0.8B-Base-F16.gguf"
    q4_gguf = gguf_dir / "Qwen3.5-0.8B-Base-Q4_K_M.gguf"
    patched_gguf = gguf_dir / "Qwen3.5-0.8B-Base-Q4_K_M-patched.gguf"
    embed_path = data_dir / "embed_tokens_q6k.npy"
    norm_path = data_dir / "output_norm.npy"

    # Step 1: HF → F16 GGUF
    if args.skip_convert and f16_gguf.exists():
        log(f"Step 1: Skipping conversion, using existing {f16_gguf}")
    else:
        log(f"Step 1: Converting {hf_dir} → {f16_gguf}")
        if not CONVERT_SCRIPT.exists():
            fail(f"convert_hf_to_gguf.py not found at {CONVERT_SCRIPT}")
        cmd = ["uv", "run", "--with", "transformers", "--with", "torch",
               "--with", "sentencepiece", "--with", "gguf", "--with", "protobuf",
               str(CONVERT_SCRIPT), str(hf_dir),
               "--outfile", str(f16_gguf), "--outtype", "f16"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log(r.stderr)
            fail("HF → GGUF conversion failed")
        log(f"  → {f16_gguf} ({f16_gguf.stat().st_size / 1024 / 1024:.1f} MB)")

    # Step 2: F16 → Q4_K_M
    log(f"Step 2: Quantizing {f16_gguf.name} → {q4_gguf.name}")
    if not QUANTIZE_BIN.exists():
        fail(f"llama-quantize not found at {QUANTIZE_BIN}")
    cmd = [str(QUANTIZE_BIN), str(f16_gguf), str(q4_gguf), "Q4_K_M"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(r.stderr)
        fail("Quantization failed")
    log(f"  → {q4_gguf} ({q4_gguf.stat().st_size / 1024 / 1024:.1f} MB)")

    # Step 3: Patch ssm_alpha/beta to Q8_0
    log(f"Step 3: Patching ssm_alpha/beta → Q8_0")
    patch_ssm_weights(q4_gguf, f16_gguf, patched_gguf)

    # Step 4: Extract Q6K embeddings
    log(f"Step 4: Extracting Q6K embeddings from {patched_gguf.name}")
    dtype, dims, raw = read_gguf_tensor(str(patched_gguf), "token_embd.weight")
    if dtype != GGML_TYPE_Q6_K:
        fail(f"Expected Q6_K (14) for token_embd, got {dtype}")
    n_cols, n_rows = dims[0], dims[1]
    embed = dequant_q6k(raw, n_rows, n_cols)
    np.save(str(embed_path), embed)
    log(f"  → {embed_path} ({embed.shape}, {embed_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Step 5: Extract output norm
    log(f"Step 5: Extracting output_norm from {patched_gguf.name}")
    dtype, dims, raw = read_gguf_tensor(str(patched_gguf), "output_norm.weight")
    if dtype != 0:
        fail(f"Expected F32 (0) for output_norm, got {dtype}")
    norm_raw = np.frombuffer(raw, dtype=np.float32)
    # GGUF stores multiplicative form (mean ~4.3), not additive offset
    mean_val = norm_raw.mean()
    if abs(mean_val) < 0.5:
        log(f"  Additive offset form (mean={mean_val:.4f}), converting to 1+w")
        norm_weight = norm_raw + 1.0
    else:
        log(f"  Multiplicative form (mean={mean_val:.4f})")
        norm_weight = norm_raw
    np.save(str(norm_path), norm_weight.astype(np.float32))
    log(f"  → {norm_path}")

    # Step 6: Cross-validate
    log(f"Step 6: Cross-validation")
    cross_validate(embed_path, norm_path, hf_dir)

    # Summary
    log("\n" + "=" * 60)
    log("All artifacts prepared successfully:")
    log(f"  F16 GGUF:     {f16_gguf}")
    log(f"  Q4_K_M GGUF:  {q4_gguf}")
    log(f"  Patched GGUF: {patched_gguf}  ← use this for inference/training")
    log(f"  Embeddings:   {embed_path}")
    log(f"  Output norm:  {norm_path}")
    log("=" * 60)


if __name__ == "__main__":
    main()
