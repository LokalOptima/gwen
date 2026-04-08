#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["gguf", "numpy"]
# ///
"""Extract MTP head from a combined GGUF into a standalone sidecar GGUF.

Usage:
    ./scripts/extract_mtp_gguf.py [--mtp-gguf PATH] [--lm-head PATH] [--output PATH]

Defaults:
    --mtp-gguf  ~/.cache/gwen/Qwen3.5-0.8B-mtp-Q8_0.gguf
    --lm-head   ~/.cache/gwen/lm_head_top50000.bin
    --output    ~/.cache/gwen/Qwen3.5-0.8B-mtp-head.gguf
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
from gguf.constants import GGML_QUANT_SIZES


def make_tensor_array(data_bytes: bytes, ggml_shape: list[int], raw_type: GGMLQuantizationType) -> tuple[np.ndarray, GGMLQuantizationType | None]:
    """Convert raw tensor bytes to a numpy array the GGUF writer accepts.

    For native numpy types (F32, F16): returns a typed array with correct shape.
    For quantized types: returns uint8 in byte-shape (numpy order) + raw_dtype.
    """
    if raw_type == GGMLQuantizationType.F32:
        return np.frombuffer(data_bytes, dtype=np.float32).reshape(list(reversed(ggml_shape))), None
    if raw_type == GGMLQuantizationType.F16:
        return np.frombuffer(data_bytes, dtype=np.float16).reshape(list(reversed(ggml_shape))), None
    # Quantized: reshape uint8 to (rows..., bytes_per_innermost_dim)
    block_size, type_size = GGML_QUANT_SIZES[raw_type]
    numpy_shape = list(reversed(ggml_shape))  # outermost first
    numpy_shape[-1] = numpy_shape[-1] // block_size * type_size  # elements → bytes
    return np.frombuffer(data_bytes, dtype=np.uint8).reshape(numpy_shape), raw_type


def main():
    cache = Path.home() / ".cache" / "gwen"
    parser = argparse.ArgumentParser(description="Extract MTP sidecar GGUF")
    parser.add_argument("--mtp-gguf", type=Path, default=cache / "Qwen3.5-0.8B-mtp-Q8_0.gguf")
    parser.add_argument("--lm-head", type=Path, default=cache / "lm_head_top50000.bin")
    parser.add_argument("--output", type=Path, default=cache / "Qwen3.5-0.8B-mtp-head.gguf")
    args = parser.parse_args()

    # --- Read MTP tensors from combined GGUF ---
    print(f"Reading {args.mtp_gguf}")
    reader = GGUFReader(str(args.mtp_gguf))

    mtp_tensors = [t for t in reader.tensors if t.name.startswith("blk.24.")]
    if not mtp_tensors:
        print("ERROR: No MTP tensors found (blk.24.*)", file=sys.stderr)
        sys.exit(1)

    total_mtp = sum(t.n_bytes for t in mtp_tensors)
    print(f"  Found {len(mtp_tensors)} MTP tensors ({total_mtp / 1024 / 1024:.1f} MiB)")

    # --- Read restricted LM head binary ---
    lm_head_data = None
    token_ids = None
    lm_head_shape = None
    lm_head_type = None
    lm_head_nbytes = 0

    if args.lm_head.exists():
        print(f"Reading {args.lm_head}")
        with open(args.lm_head, "rb") as f:
            magic = f.read(4)
            if magic != b"GWRL":
                print(f"ERROR: Invalid LM head magic: {magic}", file=sys.stderr)
                sys.exit(1)
            version, K, n_embed, gtype, row_bytes = struct.unpack("5I", f.read(20))
            assert version == 1
            token_ids = np.frombuffer(f.read(K * 4), dtype=np.int32).copy()
            lm_head_data = f.read(K * row_bytes)
            lm_head_shape = [n_embed, K]  # ggml shape: [cols, rows]
            lm_head_type = GGMLQuantizationType(gtype)
            lm_head_nbytes = len(lm_head_data)
        print(f"  Restricted LM head: {K} tokens, type={lm_head_type.name}, {lm_head_nbytes / 1024 / 1024:.1f} MiB")
    else:
        print(f"  No LM head binary at {args.lm_head}, skipping")

    # --- Write sidecar GGUF ---
    print(f"Writing {args.output}")
    writer = GGUFWriter(str(args.output), arch="gwen-mtp")

    # Metadata
    writer.add_uint32("gwen.mtp_version", 1)
    if token_ids is not None:
        writer.add_array("gwen.mtp_token_ids", token_ids.tolist())

    # MTP layer tensors (preserve exact types from source)
    for t in mtp_tensors:
        raw_type = GGMLQuantizationType(t.tensor_type)
        ggml_shape = [int(x) for x in t.shape]
        arr, dtype = make_tensor_array(bytes(t.data), ggml_shape, raw_type)
        writer.add_tensor(t.name, arr, raw_dtype=dtype)

    # Restricted LM head tensor
    if lm_head_data is not None:
        arr, dtype = make_tensor_array(lm_head_data, lm_head_shape, lm_head_type)
        writer.add_tensor("mtp_lm_head", arr, raw_dtype=dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    output_size = args.output.stat().st_size
    print(f"Done: {args.output} ({output_size / 1024 / 1024:.1f} MiB)")
    print(f"  {len(mtp_tensors)} MTP tensors + {'1 LM head' if lm_head_data else 'no LM head'}")


if __name__ == "__main__":
    main()
