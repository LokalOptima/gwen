#!/usr/bin/env python3
"""Quantize Qwen3.5 HuggingFace BF16 weights to GWEN FP8 E4M3 format (.gwfp8).

Usage:
    uv run scripts/quantize_fp8.py [--model Qwen/Qwen3.5-0.8B-Base] [--output model.gwfp8]

The script:
1. Loads BF16 weights from HuggingFace SafeTensors
2. Maps HF tensor names to GWEN's internal names (matching GGUF convention)
3. Applies the same QKV reordering as llama.cpp's converter (in_proj_qkvz split, V head reorder)
4. Quantizes weight matrices to FP8 E4M3 with per-row FP32 scales
5. Keeps small tensors (norms, biases, A_log, conv1d) as F32
6. Writes a mmap-friendly .gwfp8 binary file
"""

import argparse
import struct
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np

# GWFP8 file format constants
GWFP8_MAGIC = b"GWF8"
GWFP8_VERSION = 1

# Dtype codes in GWFP8 format
DTYPE_FP8_E4M3 = 0
DTYPE_F32 = 1
DTYPE_F16 = 2

# Scale modes
SCALE_NONE = 0     # no scaling (F32/F16 tensors)
SCALE_PER_ROW = 1  # one F32 scale per row


def quantize_to_fp8_e4m3(tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a 2D tensor to FP8 E4M3 with per-row scaling.

    Returns (fp8_data, scales) where:
    - fp8_data: uint8 numpy array (FP8 E4M3 bit pattern)
    - scales: float32 numpy array with one scale per row
    """
    assert tensor.ndim == 2, f"Expected 2D tensor, got {tensor.ndim}D"
    tensor = tensor.float()

    # Per-row max absolute value
    row_max = tensor.abs().amax(dim=1)
    # E4M3 max representable value is 448.0
    scales = (row_max / 448.0).clamp(min=1e-12)

    # Scale each row
    scaled = tensor / scales.unsqueeze(1)

    # Clamp to E4M3 range and convert
    # torch.float8_e4m3fn is available in PyTorch 2.1+
    fp8 = scaled.to(torch.float8_e4m3fn)

    # Get the raw bytes as uint8
    fp8_bytes = fp8.view(torch.uint8).numpy()
    scales_np = scales.numpy().astype(np.float32)

    return fp8_bytes, scales_np


def get_hf_tensor_name_map(n_layers: int, hparams: dict) -> dict[str, tuple[str, bool]]:
    """Build mapping from HF tensor name → (GWEN internal name, is_weight_matrix).

    GWEN internal names follow GGUF convention: blk.{i}.{name}.weight
    is_weight_matrix: True for large matrices that should be quantized to FP8

    The mapping handles both DeltaNet and FullAttn layers.
    Full attention layers: indices where (i+1) % full_attn_interval == 0
    """
    full_attn_interval = hparams.get("full_attention_interval", 4)
    name_map = {}

    # Global tensors — support both "model.layers" and "model.language_model.layers" prefixes
    for pfx in ["model", "model.language_model"]:
        name_map[f"{pfx}.embed_tokens.weight"] = ("token_embd.weight", True)
        name_map[f"{pfx}.norm.weight"] = ("output_norm.weight", False)
    # For models with untied embeddings (9B)
    name_map["lm_head.weight"] = ("output.weight", True)
    name_map["model.language_model.lm_head.weight"] = ("output.weight", True)

    for i in range(n_layers):
        # Try both prefixes
        for pfx in [f"model.layers.{i}", f"model.language_model.layers.{i}"]:
            hf = pfx
            gw = f"blk.{i}"
            is_full_attn = ((i + 1) % full_attn_interval == 0)

            # Shared: norms
            name_map[f"{hf}.input_layernorm.weight"] = (f"{gw}.attn_norm.weight", False)
            name_map[f"{hf}.post_attention_layernorm.weight"] = (f"{gw}.post_attention_norm.weight", False)

            # Shared: FFN
            name_map[f"{hf}.mlp.gate_proj.weight"] = (f"{gw}.ffn_gate.weight", True)
            name_map[f"{hf}.mlp.up_proj.weight"] = (f"{gw}.ffn_up.weight", True)
            name_map[f"{hf}.mlp.down_proj.weight"] = (f"{gw}.ffn_down.weight", True)

            if is_full_attn:
                # Full attention projections
                name_map[f"{hf}.self_attn.q_proj.weight"] = (f"{gw}.attn_q.weight", True)
                name_map[f"{hf}.self_attn.k_proj.weight"] = (f"{gw}.attn_k.weight", True)
                name_map[f"{hf}.self_attn.v_proj.weight"] = (f"{gw}.attn_v.weight", True)
                name_map[f"{hf}.self_attn.o_proj.weight"] = (f"{gw}.attn_output.weight", True)
                name_map[f"{hf}.self_attn.q_norm.weight"] = (f"{gw}.attn_q_norm.weight", False)
                name_map[f"{hf}.self_attn.k_norm.weight"] = (f"{gw}.attn_k_norm.weight", False)
            else:
                # DeltaNet: in_proj_qkvz is split into attn_qkv + attn_gate
                # This is handled specially in process_tensors()
                name_map[f"{hf}.linear_attn.in_proj_qkvz.weight"] = ("SPLIT_QKVZ", True)
                # Separate QKV (Qwen3.5 may use this instead of qkvz)
                name_map[f"{hf}.linear_attn.in_proj_qkv.weight"] = (f"{gw}.attn_qkv.weight", True)
                name_map[f"{hf}.linear_attn.in_proj_z.weight"] = (f"{gw}.attn_gate.weight", True)

                name_map[f"{hf}.linear_attn.conv1d.weight"] = (f"{gw}.ssm_conv1d.weight", False)
                name_map[f"{hf}.linear_attn.A_log"] = (f"{gw}.ssm_a", False)  # transformed: -exp(A_log)
                name_map[f"{hf}.linear_attn.dt_bias"] = (f"{gw}.ssm_dt.bias", False)
                name_map[f"{hf}.linear_attn.in_proj_a.weight"] = (f"{gw}.ssm_alpha.weight", True)
                name_map[f"{hf}.linear_attn.in_proj_b.weight"] = (f"{gw}.ssm_beta.weight", True)
                name_map[f"{hf}.linear_attn.norm.weight"] = (f"{gw}.ssm_norm.weight", False)
                name_map[f"{hf}.linear_attn.out_proj.weight"] = (f"{gw}.ssm_out.weight", True)

    return name_map


def split_qkvz(tensor: torch.Tensor, hparams: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Split in_proj_qkvz into QKV and Z (gate), reordering from interleaved to grouped.

    HF stores: [q0,k0,v0,z0, q1,k1,v1,z1, ...] per head group
    GWEN wants: QKV = [q_all, k_all, v_all], Z = [z_all]

    Matches llama.cpp's convert_hf_to_gguf.py Qwen3NextModel.modify_tensors()
    """
    head_k_dim = hparams["linear_key_head_dim"]
    head_v_dim = hparams["linear_value_head_dim"]
    num_v_heads = hparams["linear_num_value_heads"]
    num_k_heads = hparams["linear_num_key_heads"]
    hidden_size = hparams["hidden_size"]
    num_v_per_k = num_v_heads // num_k_heads

    split_sizes = [
        head_k_dim,                    # q partition
        head_k_dim,                    # k partition
        num_v_per_k * head_v_dim,      # v partition
        num_v_per_k * head_v_dim,      # z partition
    ]

    # Reshape: [total_out, hidden] → [hidden, num_k_heads, q+k+v+z]
    data = tensor.permute(1, 0).contiguous()
    data = data.view(-1, num_k_heads, sum(split_sizes))

    q, k, v, z = torch.split(data, split_sizes, dim=-1)
    q = q.contiguous().view(hidden_size, -1)
    k = k.contiguous().view(hidden_size, -1)
    v = v.contiguous().view(hidden_size, -1)
    z = z.contiguous().view(hidden_size, -1)

    qkv = torch.cat([q, k, v], dim=-1).permute(1, 0).contiguous()
    z = z.permute(1, 0).contiguous()

    return qkv, z


def reorder_v_heads(tensor: torch.Tensor, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int) -> torch.Tensor:
    """Reorder V heads from grouped (by K head) to tiled order.

    Matches llama.cpp's _LinearAttentionVReorderBase._reorder_v_heads()
    Only needed when num_k_heads != num_v_heads (4B/9B models).
    """
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)


def transform_norm_weight(tensor: torch.Tensor, name: str, hparams: dict) -> torch.Tensor:
    """Apply llama.cpp's norm weight transformation: weight += 1.

    llama.cpp adds 1 to norm weights (except linear_attn.norm) during conversion.
    This means the GGUF norm weights have bias=1 baked in.
    We must do the same to match llama.cpp's inference behavior.
    """
    if name.endswith("norm.weight") and "linear_attn.norm" not in name:
        return tensor + 1.0
    return tensor


def write_gwfp8(output_path: str, tensors: list[tuple[str, int, list[int], int, np.ndarray, np.ndarray]]):
    """Write tensors to GWFP8 format.

    Each tensor entry: (name, dtype, shape, scale_mode, scales, data)
    """
    ALIGN = 64  # alignment for mmap

    with open(output_path, "wb") as f:
        # Header
        f.write(GWFP8_MAGIC)
        f.write(struct.pack("<I", GWFP8_VERSION))
        f.write(struct.pack("<I", len(tensors)))

        # Placeholder for header_size (will fill in after writing tensor headers)
        header_size_pos = f.tell()
        f.write(struct.pack("<Q", 0))

        # Tensor headers
        tensor_header_info = []
        for name, dtype, shape, scale_mode, scales, data in tensors:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", dtype))
            f.write(struct.pack("<I", len(shape)))
            for s in shape:
                f.write(struct.pack("<Q", s))
            f.write(struct.pack("<I", scale_mode))
            # Scale count
            n_scales = len(scales) if scales is not None else 0
            f.write(struct.pack("<I", n_scales))
            # Scale data offset + data offset (placeholders)
            scale_offset_pos = f.tell()
            f.write(struct.pack("<Q", 0))  # scale_offset
            data_offset_pos = f.tell()
            f.write(struct.pack("<Q", 0))  # data_offset
            data_size = len(data.tobytes()) if data is not None else 0
            f.write(struct.pack("<Q", data_size))

            tensor_header_info.append((scale_offset_pos, data_offset_pos, scales, data))

        # Record header size
        header_end = f.tell()
        # Align to ALIGN
        pad = (ALIGN - (header_end % ALIGN)) % ALIGN
        f.write(b"\x00" * pad)
        data_start = f.tell()

        # Go back and write header_size
        f.seek(header_size_pos)
        f.write(struct.pack("<Q", data_start))
        f.seek(data_start)

        # Write tensor data (aligned)
        for scale_offset_pos, data_offset_pos, scales, data in tensor_header_info:
            # Align
            pos = f.tell()
            pad = (ALIGN - (pos % ALIGN)) % ALIGN
            f.write(b"\x00" * pad)

            # Write scales
            scale_pos = f.tell()
            if scales is not None and len(scales) > 0:
                f.write(scales.tobytes())

            # Align data
            pos = f.tell()
            pad = (ALIGN - (pos % ALIGN)) % ALIGN
            f.write(b"\x00" * pad)

            # Write data
            data_pos = f.tell()
            if data is not None:
                f.write(data.tobytes())

            # Go back and fill in offsets
            cur = f.tell()
            f.seek(scale_offset_pos)
            f.write(struct.pack("<Q", scale_pos))
            f.seek(data_offset_pos)
            f.write(struct.pack("<Q", data_pos))
            f.seek(cur)

    file_size = os.path.getsize(output_path)
    print(f"Wrote {output_path}: {len(tensors)} tensors, {file_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Quantize Qwen3.5 to FP8 E4M3")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output", default=None,
                        help="Output .gwfp8 file path")
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/models"),
                        help="Directory to cache downloaded model")
    args = parser.parse_args()

    if args.output is None:
        model_name = args.model.replace("/", "-")
        args.output = os.path.join(args.cache_dir, f"{model_name}-fp8.gwfp8")

    # Load model config
    model_dir = args.model
    if not os.path.isdir(model_dir):
        # Download from HuggingFace
        print(f"Downloading {args.model}...")
        import subprocess
        result = subprocess.run(
            ["uvx", "hf", "download", args.model, "--local-dir",
             os.path.join(args.cache_dir, args.model.replace("/", "--"))],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Download failed: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        model_dir = os.path.join(args.cache_dir, args.model.replace("/", "--"))
        print(f"Downloaded to {model_dir}")

    # Read config.json (may be nested under text_config for multimodal models)
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        raw_config = json.load(f)
    hparams = raw_config.get("text_config", raw_config)

    n_layers = hparams["num_hidden_layers"]
    full_attn_interval = hparams.get("full_attention_interval", 4)
    num_k_heads = hparams.get("linear_num_key_heads", hparams.get("num_attention_heads", 8))
    num_v_heads = hparams.get("linear_num_value_heads", num_k_heads)
    need_v_reorder = (num_k_heads != num_v_heads)

    print(f"Model: {args.model}")
    print(f"  Layers: {n_layers}, Full attn interval: {full_attn_interval}")
    print(f"  K heads: {num_k_heads}, V heads: {num_v_heads}, V reorder: {need_v_reorder}")
    print(f"  Hidden: {hparams['hidden_size']}, FFN: {hparams['intermediate_size']}")

    # Build name map
    name_map = get_hf_tensor_name_map(n_layers, hparams)

    # Load SafeTensors
    from safetensors import safe_open

    safetensor_files = sorted(Path(model_dir).glob("*.safetensors"))
    if not safetensor_files:
        print(f"No .safetensors files found in {model_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading from {len(safetensor_files)} safetensor file(s)...")

    output_tensors = []
    total_fp8_bytes = 0
    total_f32_bytes = 0

    for sf_path in safetensor_files:
        with safe_open(str(sf_path), framework="pt", device="cpu") as sf:
            for hf_name in sf.keys():
                tensor = sf.get_tensor(hf_name)

                if hf_name not in name_map:
                    print(f"  SKIP: {hf_name} (not in name map)")
                    continue

                gwen_name, is_weight_matrix = name_map[hf_name]

                # --- Special handling ---

                # Split QKVZ into QKV + Z
                if gwen_name == "SPLIT_QKVZ":
                    layer_idx = int(hf_name.split(".")[2])
                    prefix = f"blk.{layer_idx}"

                    qkv, z = split_qkvz(tensor, hparams)

                    # V head reorder for asymmetric models (4B/9B)
                    if need_v_reorder:
                        head_k_dim = hparams["linear_key_head_dim"]
                        head_v_dim = hparams["linear_value_head_dim"]
                        num_v_per_k = num_v_heads // num_k_heads
                        q_dim = head_k_dim * num_k_heads
                        k_dim = head_k_dim * num_k_heads
                        q_part = qkv[:q_dim]
                        k_part = qkv[q_dim:q_dim + k_dim]
                        v_part = qkv[q_dim + k_dim:]
                        v_part = reorder_v_heads(v_part, 0, num_k_heads, num_v_per_k, head_v_dim)
                        qkv = torch.cat([q_part, k_part, v_part], dim=0)
                        z = reorder_v_heads(z, 0, num_k_heads, num_v_per_k, head_v_dim)

                    # Quantize QKV
                    fp8_data, scales = quantize_to_fp8_e4m3(qkv)
                    output_tensors.append((
                        f"{prefix}.attn_qkv.weight", DTYPE_FP8_E4M3,
                        list(qkv.shape), SCALE_PER_ROW, scales, fp8_data
                    ))
                    total_fp8_bytes += fp8_data.nbytes

                    # Quantize Z (gate)
                    fp8_data, scales = quantize_to_fp8_e4m3(z)
                    output_tensors.append((
                        f"{prefix}.attn_gate.weight", DTYPE_FP8_E4M3,
                        list(z.shape), SCALE_PER_ROW, scales, fp8_data
                    ))
                    total_fp8_bytes += fp8_data.nbytes

                    print(f"  SPLIT: {hf_name} → {prefix}.attn_qkv + {prefix}.attn_gate")
                    continue

                # Transform A_log → -exp(A_log) (same as llama.cpp)
                if hf_name.endswith(".A_log"):
                    tensor = -torch.exp(tensor)

                # Conv1d: squeeze extra dims
                if "conv1d" in hf_name:
                    tensor = tensor.squeeze()

                # Norm weight transformation: +1 (matches llama.cpp)
                tensor = transform_norm_weight(tensor, hf_name, hparams)

                # dt_bias: rename to match GGUF convention
                # (already handled in name_map)

                # V head reorder for specific DeltaNet tensors (4B/9B only)
                if need_v_reorder and "linear_attn." in hf_name:
                    head_k_dim = hparams["linear_key_head_dim"]
                    head_v_dim = hparams["linear_value_head_dim"]
                    num_v_per_k = num_v_heads // num_k_heads

                    if "in_proj_a" in hf_name or "in_proj_b" in hf_name:
                        tensor = reorder_v_heads(tensor, 0, num_k_heads, num_v_per_k, 1)
                    elif "out_proj" in hf_name:
                        tensor = reorder_v_heads(tensor, 1, num_k_heads, num_v_per_k, head_v_dim)
                    elif "conv1d" in hf_name:
                        qk_channels = head_k_dim * num_k_heads * 2
                        qk_part = tensor[:qk_channels]
                        v_part = tensor[qk_channels:]
                        v_part = reorder_v_heads(v_part, 0, num_k_heads, num_v_per_k, head_v_dim)
                        tensor = torch.cat([qk_part, v_part], dim=0)

                # Quantize or store as-is
                if is_weight_matrix and tensor.ndim == 2:
                    fp8_data, scales = quantize_to_fp8_e4m3(tensor)
                    output_tensors.append((
                        gwen_name, DTYPE_FP8_E4M3,
                        list(tensor.shape), SCALE_PER_ROW, scales, fp8_data
                    ))
                    total_fp8_bytes += fp8_data.nbytes
                    print(f"  FP8:  {hf_name} → {gwen_name}  {list(tensor.shape)}")
                else:
                    f32_data = tensor.float().numpy().astype(np.float32)
                    output_tensors.append((
                        gwen_name, DTYPE_F32,
                        list(tensor.shape), SCALE_NONE, None, f32_data
                    ))
                    total_f32_bytes += f32_data.nbytes
                    print(f"  F32:  {hf_name} → {gwen_name}  {list(tensor.shape)}")

    print(f"\nTotal: {len(output_tensors)} tensors")
    print(f"  FP8: {total_fp8_bytes / 1024 / 1024:.1f} MB")
    print(f"  F32: {total_f32_bytes / 1024 / 1024:.1f} MB")

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_gwfp8(args.output, output_tensors)


if __name__ == "__main__":
    main()
