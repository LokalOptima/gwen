#!/usr/bin/env python3
"""
Extract Multi-Token Prediction (MTP) weights from Qwen3.5-0.8B safetensors model
and save them as a simple binary file for GWEN inference.

Binary format (GWMT):
    Magic: b"GWMT" (4 bytes)
    Version: 1 (uint32 LE)
    N_tensors: 15 (uint32 LE)
    For each tensor:
        name_len: uint32 LE
        name: bytes[name_len] (UTF-8, no null terminator)
        dtype: uint32 (0 = F32, 1 = F16, 8 = Q8_0)
        ndims: uint32 LE
        shape: uint64[ndims] LE (row-major order)
        data_size: uint64 LE (bytes)
        data: raw bytes[data_size]

    Q8_0 format: blocks of 32 values, each block = float16 scale (2 bytes) + int8[32] (32 bytes) = 34 bytes
    Shape stored as original logical shape; data_size = ceil(n_elements/32) * 34

Usage:
    uv run --with safetensors --with numpy --with ml-dtypes scripts/extract_mtp_weights.py --model-dir ~/models/hf/Qwen3.5-0.8B
    uv run --with safetensors --with numpy --with ml-dtypes scripts/extract_mtp_weights.py --model-dir ~/models/hf/Qwen3.5-0.8B --quantize q8_0
    uv run --with safetensors --with numpy --with ml-dtypes scripts/extract_mtp_weights.py --model-dir ~/models/hf/Qwen3.5-0.8B --output ~/models/gguf/Qwen3.5-0.8B-mtp-q8.bin --quantize q8_0
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import ml_dtypes  # noqa: F401 — registers bfloat16 with numpy
from safetensors import safe_open


# Expected MTP tensors in canonical order, with (name, expected_shape)
EXPECTED_MTP_TENSORS = [
    ("mtp.fc.weight",                                     (1024, 2048)),
    ("mtp.pre_fc_norm_embedding.weight",                   (1024,)),
    ("mtp.pre_fc_norm_hidden.weight",                      (1024,)),
    ("mtp.layers.0.self_attn.q_proj.weight",               (4096, 1024)),  # Q + gate interleaved
    ("mtp.layers.0.self_attn.k_proj.weight",               (512, 1024)),
    ("mtp.layers.0.self_attn.v_proj.weight",               (512, 1024)),
    ("mtp.layers.0.self_attn.o_proj.weight",               (1024, 2048)),
    ("mtp.layers.0.self_attn.q_norm.weight",               (256,)),
    ("mtp.layers.0.self_attn.k_norm.weight",               (256,)),
    ("mtp.layers.0.input_layernorm.weight",                (1024,)),
    ("mtp.layers.0.post_attention_layernorm.weight",       (1024,)),
    ("mtp.layers.0.mlp.gate_proj.weight",                  (3584, 1024)),
    ("mtp.layers.0.mlp.up_proj.weight",                    (3584, 1024)),
    ("mtp.layers.0.mlp.down_proj.weight",                  (1024, 3584)),
    ("mtp.norm.weight",                                    (1024,)),
]

EXPECTED_NAMES = {name for name, _ in EXPECTED_MTP_TENSORS}

# GWMT dtype codes (match GGMLType enum)
DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPE_Q8_0 = 8

# Q8_0 block: 32 values → half d (2 bytes) + int8[32] (32 bytes) = 34 bytes
Q8_0_BLOCK_SIZE = 32
Q8_0_BLOCK_BYTES = 34


def is_norm_tensor(name: str) -> bool:
    """Check if a tensor is a norm weight (should stay F32)."""
    return "norm" in name


def quantize_q8_0(tensor: np.ndarray) -> bytes:
    """Quantize a tensor to Q8_0 format (GGML block_q8_0).

    Each block: half d (scale) + int8[32] (quantized values).
    d = max(abs(block)) / 127
    qs[i] = round(block[i] / d)
    """
    flat = tensor.astype(np.float32).ravel()
    n = len(flat)
    n_blocks = (n + Q8_0_BLOCK_SIZE - 1) // Q8_0_BLOCK_SIZE

    # Pad to multiple of block size
    if n % Q8_0_BLOCK_SIZE != 0:
        flat = np.pad(flat, (0, Q8_0_BLOCK_SIZE - n % Q8_0_BLOCK_SIZE))

    blocks = flat.reshape(n_blocks, Q8_0_BLOCK_SIZE)
    result = bytearray(n_blocks * Q8_0_BLOCK_BYTES)

    for i in range(n_blocks):
        block = blocks[i]
        amax = np.max(np.abs(block))
        d = amax / 127.0 if amax != 0 else 0.0
        d_fp16 = np.float16(d)
        if d != 0:
            qs = np.clip(np.round(block / d), -128, 127).astype(np.int8)
        else:
            qs = np.zeros(Q8_0_BLOCK_SIZE, dtype=np.int8)

        offset = i * Q8_0_BLOCK_BYTES
        struct.pack_into("<e", result, offset, float(d_fp16))
        result[offset + 2 : offset + 2 + Q8_0_BLOCK_SIZE] = qs.tobytes()

    return bytes(result)


def find_safetensors_files(model_dir: Path) -> list[Path]:
    """Find safetensors files, handling both single and sharded models."""
    # Check for index file (sharded model)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        # Get unique shard filenames
        shard_files = sorted(set(index["weight_map"].values()))
        paths = [model_dir / f for f in shard_files]
        for p in paths:
            if not p.exists():
                print(f"Error: Shard file not found: {p}", file=sys.stderr)
                sys.exit(1)
        return paths

    # Check for single model file
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]

    # Fallback: glob for any safetensors files
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        print(f"Error: No safetensors files found in {model_dir}", file=sys.stderr)
        sys.exit(1)
    return files


def load_mtp_tensors(model_dir: Path) -> dict[str, np.ndarray]:
    """Load all MTP tensors from safetensors files."""
    safetensors_files = find_safetensors_files(model_dir)
    print(f"Found {len(safetensors_files)} safetensors file(s):")
    for f in safetensors_files:
        print(f"  {f.name}")
    print()

    mtp_tensors: dict[str, np.ndarray] = {}

    for st_path in safetensors_files:
        with safe_open(str(st_path), framework="numpy") as f:
            for name in f.keys():
                if name.startswith("mtp."):
                    mtp_tensors[name] = f.get_tensor(name)

    return mtp_tensors


def validate_tensors(tensors: dict[str, np.ndarray]) -> None:
    """Validate that we have exactly the expected tensors with correct shapes."""
    found_names = set(tensors.keys())

    # Check for missing tensors
    missing = EXPECTED_NAMES - found_names
    if missing:
        print(f"Warning: Missing expected MTP tensors:", file=sys.stderr)
        for name in sorted(missing):
            print(f"  {name}", file=sys.stderr)
        sys.exit(1)

    # Check for unexpected tensors
    extra = found_names - EXPECTED_NAMES
    if extra:
        print(f"Warning: Found unexpected MTP tensors (will be included):")
        for name in sorted(extra):
            t = tensors[name]
            print(f"  {name}: {t.shape} {t.dtype}")
        print()

    # Validate shapes
    for name, expected_shape in EXPECTED_MTP_TENSORS:
        if name not in tensors:
            continue
        actual_shape = tuple(tensors[name].shape)
        if actual_shape != expected_shape:
            print(f"Error: Shape mismatch for {name}: "
                  f"expected {expected_shape}, got {actual_shape}", file=sys.stderr)
            sys.exit(1)


def convert_tensor(name: str, tensor: np.ndarray, quantize: str = "f16") -> tuple[np.ndarray | bytes, int]:
    """Convert tensor to target dtype. Returns (converted_data, dtype_code).

    For Q8_0, returns raw bytes instead of ndarray.
    """
    if is_norm_tensor(name):
        # Norm weights: keep as F32, apply Gemma-style (1+w) offset
        # Qwen3.5 uses Qwen3_5RMSNorm which computes: output * (1 + weight)
        # GGUF stores the effective scale (1+w), so we do the same for consistency
        return np.ascontiguousarray(tensor.astype(np.float32) + 1.0), DTYPE_F32
    elif quantize == "q8_0":
        # Linear weights: quantize to Q8_0 (block_q8_0 format)
        return quantize_q8_0(tensor), DTYPE_Q8_0
    else:
        # Linear weights: convert BF16 -> FP16
        return np.ascontiguousarray(tensor.astype(np.float16)), DTYPE_F16


def write_gwmt(output_path: Path, tensors: dict[str, np.ndarray], quantize: str = "f16") -> int:
    """Write tensors in GWMT binary format."""
    # Use canonical order from EXPECTED_MTP_TENSORS, then any extras alphabetically
    ordered_names = [name for name, _ in EXPECTED_MTP_TENSORS if name in tensors]
    extra_names = sorted(set(tensors.keys()) - EXPECTED_NAMES)
    ordered_names.extend(extra_names)

    n_tensors = len(ordered_names)

    with open(output_path, "wb") as f:
        # Header
        f.write(b"GWMT")                                    # Magic
        f.write(struct.pack("<I", 1))                        # Version
        f.write(struct.pack("<I", n_tensors))                # N_tensors

        # Tensors
        total_data_bytes = 0
        for name in ordered_names:
            tensor = tensors[name]
            converted, dtype_code = convert_tensor(name, tensor, quantize)

            if isinstance(converted, bytes):
                data = converted
            else:
                data = converted.tobytes()

            name_bytes = name.encode("utf-8")
            # Store original logical shape (even for Q8_0, shape reflects the tensor dimensions)
            shape = tensor.shape
            ndims = len(shape)

            # name_len + name
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            # dtype
            f.write(struct.pack("<I", dtype_code))

            # ndims + shape
            f.write(struct.pack("<I", ndims))
            for dim in shape:
                f.write(struct.pack("<Q", dim))

            # data_size + data
            f.write(struct.pack("<Q", len(data)))
            f.write(data)

            total_data_bytes += len(data)

    return total_data_bytes


def main():
    parser = argparse.ArgumentParser(
        description="Extract MTP weights from Qwen3.5-0.8B safetensors model"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path.home() / "models" / "hf" / "Qwen3.5-0.8B",
        help="Path to HuggingFace model directory (default: ~/models/hf/Qwen3.5-0.8B)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: <model-dir>/mtp_weights.bin)",
    )
    parser.add_argument(
        "--quantize",
        choices=["f16", "q8_0"],
        default="f16",
        help="Quantization for linear weights (default: f16). Norm weights always stay F32.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        print(f"Error: Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = model_dir / "mtp_weights.bin"
    else:
        output_path = output_path.expanduser().resolve()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize = args.quantize

    print(f"Model directory: {model_dir}")
    print(f"Output file:     {output_path}")
    print(f"Quantization:    {quantize}")
    print()

    # Load MTP tensors
    print("Loading MTP tensors from safetensors...")
    mtp_tensors = load_mtp_tensors(model_dir)
    print(f"Found {len(mtp_tensors)} MTP tensor(s)")
    print()

    if not mtp_tensors:
        print("Error: No MTP tensors found in model!", file=sys.stderr)
        sys.exit(1)

    # Validate
    print("Validating tensor shapes...")
    validate_tensors(mtp_tensors)
    print("All shapes match expected dimensions.")
    print()

    # Print summary
    dtype_names = {DTYPE_F32: "F32", DTYPE_F16: "F16", DTYPE_Q8_0: "Q8_0"}
    print(f"{'Tensor':<55} {'Shape':<20} {'Src dtype':<10} {'Out dtype':<10} {'Size':>10}")
    print("=" * 110)
    total_out_bytes = 0
    ordered_names = [name for name, _ in EXPECTED_MTP_TENSORS if name in mtp_tensors]
    extra_names = sorted(set(mtp_tensors.keys()) - EXPECTED_NAMES)
    ordered_names.extend(extra_names)

    for name in ordered_names:
        tensor = mtp_tensors[name]
        _, dtype_code = convert_tensor(name, tensor, quantize)
        out_dtype_name = dtype_names[dtype_code]
        src_dtype_name = str(tensor.dtype)

        if dtype_code == DTYPE_F32:
            out_bytes = tensor.size * 4
        elif dtype_code == DTYPE_Q8_0:
            n_blocks = (tensor.size + Q8_0_BLOCK_SIZE - 1) // Q8_0_BLOCK_SIZE
            out_bytes = n_blocks * Q8_0_BLOCK_BYTES
        else:
            out_bytes = tensor.size * 2

        total_out_bytes += out_bytes
        shape_str = "x".join(str(d) for d in tensor.shape)
        size_str = f"{out_bytes / 1024:.1f} KB" if out_bytes < 1024 * 1024 else f"{out_bytes / 1024 / 1024:.2f} MB"
        print(f"  {name:<53} {shape_str:<20} {src_dtype_name:<10} {out_dtype_name:<10} {size_str:>10}")

    print("=" * 110)
    print(f"  Total output data: {total_out_bytes:,} bytes ({total_out_bytes / 1024 / 1024:.2f} MB)")
    print()

    # Write binary file
    print(f"Writing GWMT binary to {output_path}...")
    data_bytes = write_gwmt(output_path, mtp_tensors, quantize)
    file_size = output_path.stat().st_size
    print(f"Done! File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  Header + metadata overhead: {file_size - data_bytes:,} bytes")
    print(f"  Tensor data: {data_bytes:,} bytes")


if __name__ == "__main__":
    main()
