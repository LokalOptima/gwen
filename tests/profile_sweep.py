#!/usr/bin/env python3
"""Send batch_extract requests at various sizes to trigger server-side profiling."""

import struct
import urllib.request

SERVER = "http://127.0.0.1:8090"

def send_batch(B, L):
    N = B * L
    flat_tokens = [100 + (i % L) for i in range(N)]
    body = struct.pack(f"<II{N}i", B, L, *flat_tokens)
    req = urllib.request.Request(
        f"{SERVER}/batch_extract",
        data=body,
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(req) as resp:
        resp.read()

# Warmup
send_batch(1, 32)
send_batch(1, 32)

# Profiled runs at key sizes
configs = [
    (4, 128),    # 512 tokens, short L
    (1, 512),    # 512 tokens, long L
    (16, 512),   # 8K tokens
    (64, 128),   # 8K tokens, short L
    (64, 512),   # 32K tokens (training target)
]

for B, L in configs:
    print(f"--- B={B} L={L} ---")
    send_batch(B, L)

print("Done. Check server output for profiles.")
