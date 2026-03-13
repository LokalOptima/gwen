#!/usr/bin/env python3
"""Test that GEMM-batched extract matches GEMV single-sequence extract."""

import struct
import sys
import urllib.request

SERVER = "http://127.0.0.1:8090"

def test_compare(tokens: list[int]):
    """Send tokens to /compare_extract, check results."""
    L = len(tokens)
    body = struct.pack(f"<I{L}i", L, *tokens)
    req = urllib.request.Request(
        f"{SERVER}/compare_extract",
        data=body,
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(req) as resp:
        import json
        result = json.loads(resp.read())
    return result


def test_batch_extract(tokens_list: list[list[int]]):
    """Send B sequences to /batch_extract, compare with individual /extract calls."""
    B = len(tokens_list)
    L = len(tokens_list[0])

    # Batch extract
    flat_tokens = [t for seq in tokens_list for t in seq]
    body = struct.pack(f"<II{B*L}i", B, L, *flat_tokens)
    req = urllib.request.Request(
        f"{SERVER}/batch_extract",
        data=body,
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    hdr = struct.unpack("<III", data[:12])
    assert hdr[0] == B and hdr[1] == L
    n_embed = hdr[2]
    batch_hidden = data[12:]

    # Individual extracts via GEMV path
    for i, tokens in enumerate(tokens_list):
        body_single = struct.pack(f"<I{L}i", L, *tokens)
        req = urllib.request.Request(
            f"{SERVER}/compare_extract",
            data=body_single,
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(req) as resp:
            import json
            result = json.loads(resp.read())
        print(f"  seq {i}: {result}")

    print(f"  Batch extracted {B}×{L} = {B*L} tokens, {len(batch_hidden)} bytes")


if __name__ == "__main__":
    print("=== GEMV vs GEMM comparison ===")

    # Test 1: short sequence
    result = test_compare([1, 2, 3, 4, 5])
    print(f"L=5:   {result}")

    # Test 2: medium sequence
    result = test_compare(list(range(100, 164)))
    print(f"L=64:  {result}")

    # Test 3: longer sequence
    result = test_compare(list(range(1000, 1256)))
    print(f"L=256: {result}")

    # Test 4: batch of 4 sequences
    print("\n=== Batch extract (4×32) ===")
    seqs = [list(range(i*100, i*100+32)) for i in range(4)]
    test_batch_extract(seqs)

    print("\nDone.")
