#!/usr/bin/env python3
"""Dump exact weight tensor sizes from the GGUF model file.
Used to compute precise theoretical bandwidth limits."""

import struct, sys, os

# GGML quant block sizes
QUANT_INFO = {
    # type_id: (name, block_size_elements, block_size_bytes)
    0:  ("F32",  1, 4),
    1:  ("F16",  1, 2),
    8:  ("Q8_0", 32, 34),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
}

def read_gguf_tensors(path):
    """Parse GGUF header to extract tensor info without loading data."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', f"Not a GGUF file: {magic}"
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        # Skip metadata key-value pairs
        def read_string():
            length = struct.unpack('<Q', f.read(8))[0]
            return f.read(length).decode('utf-8')

        def read_value(vtype):
            if vtype == 0:    return struct.unpack('<B', f.read(1))[0]  # uint8
            elif vtype == 1:  return struct.unpack('<b', f.read(1))[0]  # int8
            elif vtype == 2:  return struct.unpack('<H', f.read(2))[0]  # uint16
            elif vtype == 3:  return struct.unpack('<h', f.read(2))[0]  # int16
            elif vtype == 4:  return struct.unpack('<I', f.read(4))[0]  # uint32
            elif vtype == 5:  return struct.unpack('<i', f.read(4))[0]  # int32
            elif vtype == 6:  return struct.unpack('<f', f.read(4))[0]  # float32
            elif vtype == 7:  return struct.unpack('<?', f.read(1))[0]  # bool
            elif vtype == 8:  return read_string()                       # string
            elif vtype == 9:  # array
                atype = struct.unpack('<I', f.read(4))[0]
                alen = struct.unpack('<Q', f.read(8))[0]
                return [read_value(atype) for _ in range(alen)]
            elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]  # uint64
            elif vtype == 11: return struct.unpack('<q', f.read(8))[0]  # int64
            elif vtype == 12: return struct.unpack('<d', f.read(8))[0]  # float64
            else: raise ValueError(f"Unknown value type {vtype}")

        for _ in range(n_kv):
            key = read_string()
            vtype = struct.unpack('<I', f.read(4))[0]
            val = read_value(vtype)

        # Read tensor info
        tensors = []
        for _ in range(n_tensors):
            name = read_string()
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            type_id = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            n_elements = 1
            for d in dims:
                n_elements *= d
            if type_id in QUANT_INFO:
                qname, blk_elems, blk_bytes = QUANT_INFO[type_id]
                size_bytes = (n_elements // blk_elems) * blk_bytes
            else:
                qname = f"type_{type_id}"
                size_bytes = n_elements * 2  # assume FP16
            tensors.append({
                'name': name, 'dims': dims, 'type_id': type_id,
                'type_name': qname, 'n_elements': n_elements,
                'size_bytes': size_bytes
            })
        return tensors

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/Qwen3.5-9B-UD-Q4_K_XL.gguf")
    tensors = read_gguf_tensors(path)

    # Group by layer
    total_bytes = 0
    dn_total = 0
    fa_total = 0
    global_total = 0
    gemv_bytes = 0

    print(f"{'Tensor':<55} {'Shape':<20} {'Type':<6} {'Bytes':>12} {'MB':>8}")
    print("=" * 105)

    for t in sorted(tensors, key=lambda x: x['name']):
        shape_str = "x".join(str(d) for d in t['dims'])
        mb = t['size_bytes'] / 1024 / 1024
        print(f"{t['name']:<55} {shape_str:<20} {t['type_name']:<6} {t['size_bytes']:>12,} {mb:>8.2f}")
        total_bytes += t['size_bytes']

        name = t['name']
        if 'blk.' in name:
            layer_num = int(name.split('.')[1])
            is_fa = (layer_num + 1) % 4 == 0
            if is_fa:
                fa_total += t['size_bytes']
            else:
                dn_total += t['size_bytes']
        else:
            global_total += t['size_bytes']

        # Identify GEMV weights (read once per decode)
        is_gemv = any(k in name for k in [
            'attn_qkv', 'attn_gate.weight', 'ssm_out', 'ffn_gate', 'ffn_up', 'ffn_down',
            'attn_q_proj', 'attn_k_proj', 'attn_v_proj', 'attn_output',
        ])
        if name == 'token_embd.weight':
            is_gemv = True  # LM head
        if is_gemv:
            gemv_bytes += t['size_bytes']

    print("=" * 105)
    print(f"{'TOTAL':<55} {'':<20} {'':<6} {total_bytes:>12,} {total_bytes/1024/1024:>8.2f}")
    print(f"\n  DeltaNet layers (18): {dn_total:>12,} ({dn_total/1024/1024:.2f} MB)")
    print(f"  FullAttn layers (6):  {fa_total:>12,} ({fa_total/1024/1024:.2f} MB)")
    print(f"  Global:               {global_total:>12,} ({global_total/1024/1024:.2f} MB)")
    print(f"  GEMV weights:         {gemv_bytes:>12,} ({gemv_bytes/1024/1024:.2f} MB)")

    # Theoretical bandwidth limits
    bw = 896e9  # bytes/sec
    print(f"\n=== Theoretical Bandwidth Limits (RTX 5070 Ti, {bw/1e9:.0f} GB/s) ===")
    print(f"  Weights-only min time:   {total_bytes / bw * 1000:.3f} ms ({bw / total_bytes:.0f} tok/s)")

    # Q8_1 input vectors per decode (negligible but precise)
    # Per DeltaNet: 6 GEMVs read Q8_1 input, per FA: 7 GEMVs
    q8_per_dn = (1152 + 1152 + 2304 + 1152 + 1152 + 4032)  # qkv,gate,ssm_out,ffn_gate,ffn_up,ffn_down
    q8_per_fa = (1152 + 1152 + 1152 + 2304 + 1152 + 1152 + 4032)  # q,k,v,output,gate,up,down
    q8_lm = 1152
    q8_total = 18 * q8_per_dn + 6 * q8_per_fa + q8_lm
    # DeltaNet state: read+write S [16,128,128] FP32 per layer
    state_rw = 18 * 2 * 16 * 128 * 128 * 4
    # Conv state: read+write [3,6144] FP32 per layer
    conv_rw = 18 * 2 * 3 * 6144 * 4
    all_traffic = total_bytes + q8_total + state_rw + conv_rw
    print(f"  + Q8_1 inputs:           {q8_total:>10,} bytes")
    print(f"  + DeltaNet state R+W:    {state_rw:>10,} bytes ({state_rw/1024/1024:.1f} MB)")
    print(f"  + Conv state R+W:        {conv_rw:>10,} bytes ({conv_rw/1024/1024:.1f} MB)")
    print(f"  Total traffic:           {all_traffic:>10,} bytes ({all_traffic/1024/1024:.1f} MB)")
    print(f"  All-traffic min time:    {all_traffic / bw * 1000:.3f} ms ({bw / all_traffic:.0f} tok/s)")

if __name__ == '__main__':
    main()
