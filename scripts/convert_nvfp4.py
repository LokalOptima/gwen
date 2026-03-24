#!/usr/bin/env python3
"""Convert ModelOpt NVFP4 safetensors checkpoint → GWEN GWFP4 binary format.

GWFP4 format:
  Magic "GWF4" (4 bytes)
  Version (U32)
  N_tensors (U32)
  Header_size (U32)
  [Tensor headers, 64-byte aligned]
  [Tensor data, 64-byte aligned]

Each quantized weight has 3 components:
  - data: [out_features, in_features/2] uint8 (packed FP4 E2M1)
  - scales: [out_features, in_features/16] uint8 (E4M3 micro-scales)
  - scale2: float32 (per-tensor global scale)

Non-quantized weights (norms, embeddings) stored as F32 or BF16.

Usage:
    uv run scripts/convert_nvfp4.py \
        --input ~/models/Qwen3.5-4B-NVFP4 \
        --output ~/models/Qwen3.5-4B.gwfp4
"""
import argparse
import struct
import numpy as np
from pathlib import Path
from safetensors import safe_open
import torch
import json


# GWFP4 dtype codes
DTYPE_FP4_E2M1 = 0   # packed FP4 (2 per byte) with E4M3 block scales
DTYPE_F32 = 1
DTYPE_BF16 = 2
DTYPE_F16 = 3         # FP16 (for embeddings/non-quantized weights)

# HF → GWEN tensor name mapping for Qwen3.5 DeltaNet layers
DELTANET_MAP = {
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.in_proj_a": "ssm_alpha",
    "linear_attn.in_proj_b": "ssm_beta",
    "linear_attn.out_proj": "ssm_out",
    "linear_attn.A_log": "ssm_a",
    "linear_attn.dt_bias": "ssm_dt_bias",
    "linear_attn.conv1d": "ssm_conv1d",
    "linear_attn.norm": "ssm_norm",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "post_attn_norm",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}

# HF → GWEN tensor name mapping for Qwen3.5 FullAttn layers
FULLATTN_MAP = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "self_attn.q_norm": "attn_q_norm",
    "self_attn.k_norm": "attn_k_norm",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "post_attn_norm",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}

# Full attention layers (0-indexed): every 4th starting at 3
def is_full_attn(layer_idx, n_layers):
    full_attn_interval = 4
    return (layer_idx % full_attn_interval) == (full_attn_interval - 1)


def convert_name(hf_name, n_layers):
    """Convert HF tensor name to GWEN name."""
    # Global tensors
    if hf_name == "model.language_model.embed_tokens.weight":
        return "token_embd.weight"
    if hf_name == "model.language_model.norm.weight":
        return "output_norm.weight"

    # Layer tensors
    if hf_name.startswith("model.language_model.layers."):
        parts = hf_name.split(".")
        layer_idx = int(parts[3])
        # Reconstruct the suffix after "layers.N."
        suffix = ".".join(parts[4:])

        # Remove .weight suffix for mapping
        base_suffix = suffix.replace(".weight", "")

        if is_full_attn(layer_idx, n_layers):
            mapping = FULLATTN_MAP
        else:
            mapping = DELTANET_MAP

        for hf_key, gwen_key in mapping.items():
            if base_suffix == hf_key or base_suffix.startswith(hf_key + "."):
                return f"blk.{layer_idx}.{gwen_key}.weight"

    # MTP tensors
    if hf_name.startswith("mtp."):
        return "mtp." + hf_name[4:]

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="ModelOpt NVFP4 checkpoint directory")
    parser.add_argument("--output", required=True, help="Output .gwfp4 file path")
    args = parser.parse_args()

    input_dir = Path(args.input)

    # Load config
    with open(input_dir / "config.json") as f:
        config = json.load(f)
    text_cfg = config.get("text_config", config)
    n_layers = text_cfg["num_hidden_layers"]

    # Load quant config
    quant_cfg_path = input_dir / "hf_quant_config.json"
    if quant_cfg_path.exists():
        with open(quant_cfg_path) as f:
            quant_cfg = json.load(f)
        print(f"Quant config: {json.dumps(quant_cfg['quantization'], indent=2)}")

    # Open safetensors
    st_files = sorted(input_dir.glob("model*.safetensors"))
    print(f"Found {len(st_files)} safetensors file(s)")

    # Collect all tensors
    tensors = {}  # name → (data_bytes, dtype, shape, scales_bytes, scale2)
    skipped = []

    for st_path in st_files:
        with safe_open(str(st_path), framework="pt") as f:
            keys = list(f.keys())

            # Group quantized tensors: .weight, .weight_scale, .weight_scale_2, .input_scale
            # Process in groups
            base_keys = set()
            for k in keys:
                if k.endswith(".weight"):
                    base = k[:-7]  # strip ".weight"
                    base_keys.add(base)
                elif k.endswith(".weight_scale"):
                    base = k[:-13]
                    base_keys.add(base)

            for k in sorted(keys):
                # Skip scale/input_scale tensors (handled with their base weight)
                if k.endswith(".weight_scale") or k.endswith(".weight_scale_2") or k.endswith(".input_scale"):
                    continue

                # Skip visual encoder entirely
                if "visual" in k:
                    skipped.append(k)
                    continue

                # Handle bare tensors (no .weight suffix): A_log, dt_bias, etc.
                if not k.endswith(".weight") and not k.endswith("_scale") and not k.endswith("_scale_2"):
                    gwen_name = convert_name(k, n_layers)
                    if gwen_name is not None:
                        t = f.get_tensor(k)
                        # A_log → -exp(A_log) to match GGUF convention (always negative)
                        if "A_log" in k:
                            t = -torch.exp(t.float())
                        t_f32 = t.float().contiguous()
                        tensors[gwen_name] = {
                            "data": t_f32.numpy().tobytes(),
                            "dtype": DTYPE_F32,
                            "shape": list(t.shape),
                            "scales": None,
                            "scale2": 0.0,
                            "scales_shape": None,
                        }
                        print(f"  F32: {gwen_name} {list(t.shape)}")
                    continue

                if k.endswith(".weight"):
                    base = k[:-7]
                    t = f.get_tensor(k)

                    # Check if this is a quantized tensor (has .weight_scale)
                    scale_key = base + ".weight_scale"
                    scale2_key = base + ".weight_scale_2"

                    if scale_key in keys:
                        # Quantized FP4 tensor
                        scales = f.get_tensor(scale_key)
                        scale2 = float(f.get_tensor(scale2_key).item())

                        gwen_name = convert_name(k, n_layers)
                        if gwen_name is None:
                            skipped.append(k)
                            continue

                        # t is uint8 packed FP4 — may have leading dim of 1
                        data = t.squeeze().contiguous()
                        # E4M3 scales as raw bytes
                        scales_bytes = scales.contiguous().view(torch.uint8).numpy().tobytes()

                        shape = list(data.shape)
                        real_shape = [shape[0], shape[1] * 2]

                        tensors[gwen_name] = {
                            "data": data.numpy().tobytes(),
                            "dtype": DTYPE_FP4_E2M1,
                            "shape": real_shape,
                            "scales": scales_bytes,
                            "scale2": scale2,
                            "scales_shape": list(scales.shape),
                        }
                        print(f"  FP4: {gwen_name} [{real_shape[0]}×{real_shape[1]}] "
                              f"scales={list(scales.shape)} scale2={scale2:.6f}")
                    else:
                        # Non-quantized tensor (BF16 or F32)
                        gwen_name = convert_name(k, n_layers)
                        if gwen_name is None:
                            skipped.append(k)
                            continue

                        # Large 2D tensors (embeddings, MLP weights) → FP16 for GEMV
                        # Small tensors and conv1d (norms, biases, conv weights) → F32
                        if t.numel() > 4096 and t.dim() <= 2 and "conv" not in k:
                            t_fp16 = t.half().contiguous()
                            tensors[gwen_name] = {
                                "data": t_fp16.numpy().tobytes(),
                                "dtype": DTYPE_F16,
                                "shape": list(t.shape),
                                "scales": None,
                                "scale2": 0.0,
                                "scales_shape": None,
                            }
                            print(f"  F16: {gwen_name} {list(t.shape)}")
                        else:
                            t_f32 = t.float().contiguous()
                            # Qwen3.5 norms use (1 + weight) scaling; add 1.0 to match GGUF convention
                            if "norm" in k and "weight" in k:
                                t_f32 = t_f32 + 1.0
                            tensors[gwen_name] = {
                                "data": t_f32.numpy().tobytes(),
                                "dtype": DTYPE_F32,
                                "shape": list(t.shape),
                                "scales": None,
                                "scale2": 0.0,
                                "scales_shape": None,
                            }
                            print(f"  F32: {gwen_name} {list(t.shape)}")

    print(f"\nConverted {len(tensors)} tensors, skipped {len(skipped)}")
    if skipped:
        print(f"Skipped: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    # Write GWFP4 file
    write_gwfp4(args.output, tensors, text_cfg)
    print(f"\nWrote {args.output}")


def write_gwfp4(path, tensors, config):
    """Write GWFP4 binary file."""
    MAGIC = b"GWF4"
    VERSION = 1
    ALIGN = 64

    def align_to(offset, alignment):
        return (offset + alignment - 1) & ~(alignment - 1)

    # Build headers
    n_tensors = len(tensors)

    # Encode config as JSON in a special tensor
    config_json = json.dumps(config).encode("utf-8")

    # Pre-compute header size
    # Header: magic(4) + version(4) + n_tensors(4) + header_size(4) + config_len(4) + config_data
    # Per tensor: name_len(4) + name + dtype(4) + ndims(4) + shape(8*ndims) +
    #             has_scales(4) + scale2(4) + scales_shape(if has_scales: 4+8*ndims) +
    #             data_offset(8) + data_size(8) + scales_offset(8) + scales_size(8)
    header_bytes = bytearray()
    header_bytes += MAGIC
    header_bytes += struct.pack("<I", VERSION)
    header_bytes += struct.pack("<I", n_tensors)
    header_bytes += struct.pack("<I", 0)  # placeholder for header_size
    header_bytes += struct.pack("<I", len(config_json))
    header_bytes += config_json

    # Per-tensor headers (offsets filled later)
    tensor_header_offsets = {}
    for name, info in sorted(tensors.items()):
        tensor_header_offsets[name] = len(header_bytes)
        name_bytes = name.encode("utf-8")
        header_bytes += struct.pack("<I", len(name_bytes))
        header_bytes += name_bytes
        header_bytes += struct.pack("<I", info["dtype"])
        ndims = len(info["shape"])
        header_bytes += struct.pack("<I", ndims)
        for dim in info["shape"]:
            header_bytes += struct.pack("<Q", dim)
        has_scales = 1 if info["scales"] is not None else 0
        header_bytes += struct.pack("<I", has_scales)
        header_bytes += struct.pack("<f", info["scale2"])
        if has_scales:
            s_ndims = len(info["scales_shape"])
            header_bytes += struct.pack("<I", s_ndims)
            for dim in info["scales_shape"]:
                header_bytes += struct.pack("<Q", dim)
        # Placeholders for offsets
        header_bytes += struct.pack("<Q", 0)  # data_offset
        header_bytes += struct.pack("<Q", 0)  # data_size
        header_bytes += struct.pack("<Q", 0)  # scales_offset
        header_bytes += struct.pack("<Q", 0)  # scales_size

    header_size = align_to(len(header_bytes), ALIGN)

    # Update header_size field
    struct.pack_into("<I", header_bytes, 12, header_size)

    # Compute data offsets
    data_offset = header_size
    offsets = {}
    for name, info in sorted(tensors.items()):
        data_offset = align_to(data_offset, ALIGN)
        data_size = len(info["data"])
        scales_offset = 0
        scales_size = 0
        if info["scales"] is not None:
            scales_offset = align_to(data_offset + data_size, ALIGN)
            scales_size = len(info["scales"])
            next_offset = scales_offset + scales_size
        else:
            next_offset = data_offset + data_size
        offsets[name] = (data_offset, data_size, scales_offset, scales_size)
        data_offset = next_offset

    # Fill in offsets in headers
    for name, info in sorted(tensors.items()):
        off = tensor_header_offsets[name]
        # Find the offset placeholder: it's at the end of this tensor's header
        # Navigate: name_len(4) + name + dtype(4) + ndims(4) + shape(8*ndims) +
        #           has_scales(4) + scale2(4) + [scales_shape if has_scales] + offsets(32)
        name_len = len(name.encode("utf-8"))
        ndims = len(info["shape"])
        pos = off + 4 + name_len + 4 + 4 + 8 * ndims + 4 + 4
        if info["scales"] is not None:
            s_ndims = len(info["scales_shape"])
            pos += 4 + 8 * s_ndims
        d_off, d_size, s_off, s_size = offsets[name]
        struct.pack_into("<Q", header_bytes, pos, d_off)
        struct.pack_into("<Q", header_bytes, pos + 8, d_size)
        struct.pack_into("<Q", header_bytes, pos + 16, s_off)
        struct.pack_into("<Q", header_bytes, pos + 24, s_size)

    # Write file
    total_size = data_offset
    with open(path, "wb") as f:
        # Write header (padded to alignment)
        f.write(header_bytes)
        f.write(b"\x00" * (header_size - len(header_bytes)))

        # Write tensor data
        for name, info in sorted(tensors.items()):
            d_off, d_size, s_off, s_size = offsets[name]
            # Pad to alignment
            current = f.tell()
            if current < d_off:
                f.write(b"\x00" * (d_off - current))
            f.write(info["data"])

            if info["scales"] is not None:
                current = f.tell()
                if current < s_off:
                    f.write(b"\x00" * (s_off - current))
                f.write(info["scales"])

    file_size = Path(path).stat().st_size
    print(f"File size: {file_size / 1024 / 1024:.1f} MB ({n_tensors} tensors)")


if __name__ == "__main__":
    main()
