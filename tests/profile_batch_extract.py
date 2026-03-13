#!/usr/bin/env python3
"""Send a single batch_extract request for nsys profiling."""

import struct
import urllib.request

SERVER = "http://127.0.0.1:8090"

B, L = 16, 512
N = B * L
flat_tokens = [100 + (i % L) for i in range(N)]
body = struct.pack(f"<II{N}i", B, L, *flat_tokens)

# Warmup
for _ in range(2):
    req = urllib.request.Request(
        f"{SERVER}/batch_extract",
        data=body,
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(req) as resp:
        resp.read()

# Profiled call
import time
req = urllib.request.Request(
    f"{SERVER}/batch_extract",
    data=body,
    headers={"Content-Type": "application/octet-stream"},
)
t0 = time.perf_counter()
with urllib.request.urlopen(req) as resp:
    data = resp.read()
t1 = time.perf_counter()
print(f"B={B} L={L} N={N}: {(t1-t0)*1000:.1f} ms")
