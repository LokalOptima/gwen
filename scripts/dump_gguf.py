#!/usr/bin/env python3
"""
Dump GGUF file metadata and tensor info for debugging.

Usage:
    uv run scripts/dump_gguf.py Qwen3.5-9B-UD-Q4_K_XL.gguf
    uv run scripts/dump_gguf.py Qwen3.5-9B-UD-Q4_K_XL.gguf --tensors    # show all tensors
    uv run scripts/dump_gguf.py Qwen3.5-9B-UD-Q4_K_XL.gguf --layer 3    # show specific layer
"""

import argparse
import struct
import sys
from pathlib import Path

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_NAMES = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16",
    4: "uint32", 5: "int32", 6: "float32", 7: "bool",
    8: "string", 9: "array", 10: "uint64", 11: "int64", 12: "float64",
}

# GGML tensor types
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16",
}

GGML_TYPE_SIZES = {
    0: 4,     # F32
    1: 2,     # F16
    12: 0.5625,  # Q4_K: 144 bytes per 256 values = 0.5625 bytes/value
    13: 0.6875,  # Q5_K: 176 bytes per 256 values
    14: 0.8125,  # Q6_K: 208 bytes per 256 values
    8: 1.0625,   # Q8_0: 34 bytes per 32 values
    30: 2,    # BF16
}


def read_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def read_value(f, vtype):
    if vtype == 0:  # uint8
        return struct.unpack("<B", f.read(1))[0]
    elif vtype == 1:  # int8
        return struct.unpack("<b", f.read(1))[0]
    elif vtype == 2:  # uint16
        return struct.unpack("<H", f.read(2))[0]
    elif vtype == 3:  # int16
        return struct.unpack("<h", f.read(2))[0]
    elif vtype == 4:  # uint32
        return struct.unpack("<I", f.read(4))[0]
    elif vtype == 5:  # int32
        return struct.unpack("<i", f.read(4))[0]
    elif vtype == 6:  # float32
        return struct.unpack("<f", f.read(4))[0]
    elif vtype == 7:  # bool
        return struct.unpack("<B", f.read(1))[0] != 0
    elif vtype == 8:  # string
        return read_string(f)
    elif vtype == 9:  # array
        atype = struct.unpack("<I", f.read(4))[0]
        alen = struct.unpack("<Q", f.read(8))[0]
        return [read_value(f, atype) for _ in range(alen)]
    elif vtype == 10:  # uint64
        return struct.unpack("<Q", f.read(8))[0]
    elif vtype == 11:  # int64
        return struct.unpack("<q", f.read(8))[0]
    elif vtype == 12:  # float64
        return struct.unpack("<d", f.read(8))[0]
    else:
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def dump_gguf(path: str, show_tensors: bool = False, layer_filter: int = None):
    with open(path, "rb") as f:
        # Header
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            print(f"Error: Not a GGUF file (magic: 0x{magic:08X})", file=sys.stderr)
            sys.exit(1)

        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        print(f"GGUF v{version}")
        print(f"Tensors: {n_tensors}")
        print(f"Metadata KV pairs: {n_kv}")
        print()

        # Read metadata
        print("=== Metadata ===")
        metadata = {}
        for _ in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            value = read_value(f, vtype)
            metadata[key] = value

            # Format output
            if isinstance(value, list) and len(value) > 20:
                display = f"[{value[0]}, {value[1]}, ..., {value[-1]}] (len={len(value)})"
            elif isinstance(value, str) and len(value) > 100:
                display = f'"{value[:100]}..."'
            else:
                display = repr(value)
            print(f"  {key} ({GGUF_TYPE_NAMES.get(vtype, '?')}): {display}")

        print()

        # Read tensor info
        print("=== Tensors ===")
        tensors = []
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]

            n_elements = 1
            for d in dims:
                n_elements *= d

            type_name = GGML_TYPE_NAMES.get(ttype, f"type_{ttype}")
            bytes_per_elem = GGML_TYPE_SIZES.get(ttype, 0)
            size_bytes = int(n_elements * bytes_per_elem) if bytes_per_elem else 0

            tensors.append({
                "name": name,
                "dims": dims,
                "type": type_name,
                "type_id": ttype,
                "offset": offset,
                "n_elements": n_elements,
                "size_bytes": size_bytes,
            })

        # Filter and display
        for t in tensors:
            if layer_filter is not None:
                if f"blk.{layer_filter}." not in t["name"]:
                    continue
            if show_tensors or layer_filter is not None:
                dims_str = "×".join(str(d) for d in t["dims"])
                size_mb = t["size_bytes"] / 1024 / 1024
                print(f"  {t['name']:<45} [{dims_str}]  {t['type']:<6}  {size_mb:.2f} MB")

        # Summary
        if not show_tensors and layer_filter is None:
            # Show summary by type
            type_counts = {}
            type_sizes = {}
            for t in tensors:
                tn = t["type"]
                type_counts[tn] = type_counts.get(tn, 0) + 1
                type_sizes[tn] = type_sizes.get(tn, 0) + t["size_bytes"]

            print(f"  Total: {len(tensors)} tensors")
            total_mb = sum(type_sizes.values()) / 1024 / 1024
            print(f"  Total size: {total_mb:.1f} MB")
            print()
            print("  By type:")
            for tn in sorted(type_counts.keys()):
                mb = type_sizes[tn] / 1024 / 1024
                print(f"    {tn:<6}: {type_counts[tn]:>4} tensors, {mb:>8.1f} MB")

            # Layer type analysis
            print()
            print("  Layer types (from full_attention_interval=4):")
            n_layers = metadata.get("qwen35.block_count", 24)
            for i in range(n_layers):
                layer_tensors = [t for t in tensors if f"blk.{i}." in t["name"]]
                has_ssm = any("ssm" in t["name"] for t in layer_tensors)
                layer_type = "DeltaNet" if has_ssm else "FullAttn"
                n_t = len(layer_tensors)
                layer_mb = sum(t["size_bytes"] for t in layer_tensors) / 1024 / 1024
                print(f"    Layer {i:>2}: {layer_type:<10} ({n_t} tensors, {layer_mb:.1f} MB)")

        print(f"\n  Use --tensors to see all tensors, --layer N to see a specific layer")


def main():
    parser = argparse.ArgumentParser(description="Dump GGUF file info")
    parser.add_argument("path", help="Path to GGUF file")
    parser.add_argument("--tensors", action="store_true", help="Show all tensors")
    parser.add_argument("--layer", type=int, help="Show specific layer")
    args = parser.parse_args()

    dump_gguf(args.path, args.tensors, args.layer)


if __name__ == "__main__":
    main()
