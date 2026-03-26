#!/bin/bash
# Profile individual GWEN kernels with Nsight Compute (ncu)
# Usage: ./scripts/ncu_profile.sh [kernel_filter] [output_name]
#
# Examples:
#   ./scripts/ncu_profile.sh                          # Profile all kernels (1 pass)
#   ./scripts/ncu_profile.sh gemv_q4_k_dp4a           # Profile only Q4_K dp4a GEMV
#   ./scripts/ncu_profile.sh kernel_deltanet_decode    # Profile DeltaNet recurrence
#   ./scripts/ncu_profile.sh "" full                   # Full metrics collection (slow)
#
# ncu captures detailed per-kernel metrics: achieved bandwidth, occupancy,
# warp stalls, instruction mix, memory throughput breakdown.

set -euo pipefail

MODEL="${GWEN_MODEL:-$HOME/models/Qwen3.5-9B-UD-Q4_K_XL.gguf}"
GWEN="./build/gwen"
KERNEL_FILTER="${1:-}"
NAME="${2:-gwen_kernels}"
OUTDIR="profiles"
mkdir -p "$OUTDIR"

echo "=== Nsight Compute Profile ==="

# Check ncu permissions
if ! ncu --launch-count 0 /bin/true &>/dev/null 2>&1; then
    echo "NOTE: ncu may need elevated permissions for GPU performance counters."
    echo "  Docker: run container with --cap-add SYS_ADMIN or --privileged"
    echo "  Host:   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-perf.conf && sudo modprobe -r nvidia && sudo modprobe nvidia"
    echo "  Or run: sudo ncu ..."
    echo ""
fi

# Lock GPU clocks for reproducibility
if nvidia-smi -lgc 3105,3105 &>/dev/null; then
    echo "GPU clocks locked to 3105 MHz"
    LOCKED=1
else
    echo "WARNING: Could not lock GPU clocks."
    LOCKED=0
fi

# Build ncu command
NCU_CMD="ncu"

# Kernel filter
if [ -n "$KERNEL_FILTER" ]; then
    NCU_CMD="$NCU_CMD --kernel-name $KERNEL_FILTER"
    echo "Filtering kernels: $KERNEL_FILTER"
fi

# Only profile decode (skip warmup) — skip first 15 kernel launches (prefill+warmup)
# Then capture up to 500 kernels (covers ~2 full decode steps)
NCU_CMD="$NCU_CMD --launch-skip 15 --launch-count 500"

# Metrics selection
if [ "$NAME" = "full" ]; then
    # Full metrics — very slow but comprehensive
    NCU_CMD="$NCU_CMD --set full"
    echo "Collecting FULL metrics (this will be slow)"
else
    # Targeted metrics for bandwidth-bound analysis
    NCU_CMD="$NCU_CMD \
        --metrics \
gpu__time_duration.avg,\
sm__throughput.avg_pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg_pct_of_peak_sustained_elapsed,\
dram__throughput.avg_pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
l2__throughput.avg_pct_of_peak_sustained_elapsed,\
sm__warps_active.avg_pct_of_peak_sustained_active,\
sm__inst_executed.avg_per_cycle_elapsed,\
launch__grid_size,\
launch__block_size,\
launch__registers_per_thread,\
launch__shared_mem_per_block_allocated"
fi

NCU_CMD="$NCU_CMD --csv --log-file $OUTDIR/${NAME}.csv"

echo "Running: $NCU_CMD $GWEN ..."
echo ""

# ncu requires running the app — use short generation (5 tokens) to keep profile fast
$NCU_CMD \
    "$GWEN" --model "$MODEL" \
    --greedy --max-predict 5 \
    "1 2 3 4 5 6 7 8" 2>/dev/null

echo ""
echo "Raw CSV: $OUTDIR/${NAME}.csv"
echo ""

# Parse and display results
echo "=== Per-Kernel Analysis ==="
python3 -c "
import csv, sys

with open('$OUTDIR/${NAME}.csv', 'r') as f:
    # Skip ncu header lines (start with ==)
    lines = [l for l in f if not l.startswith('==')]

reader = csv.DictReader(lines)
rows = list(reader)
if not rows:
    print('No kernel data captured')
    sys.exit(0)

# Group by kernel name
from collections import defaultdict
kernels = defaultdict(list)
for r in rows:
    name = r.get('Kernel Name', r.get('kernel_name', ''))
    kernels[name].append(r)

# Find metric columns
cols = list(rows[0].keys())
time_col = next((c for c in cols if 'time_duration' in c.lower() or 'duration' in c.lower()), None)
dram_r_col = next((c for c in cols if 'dram' in c.lower() and 'read' in c.lower() and 'bytes' in c.lower()), None)
dram_w_col = next((c for c in cols if 'dram' in c.lower() and 'write' in c.lower() and 'bytes' in c.lower()), None)
dram_pct_col = next((c for c in cols if 'dram' in c.lower() and 'throughput' in c.lower() and 'pct' in c.lower()), None)
sm_pct_col = next((c for c in cols if 'sm__throughput' in c.lower() and 'pct' in c.lower()), None)
occ_col = next((c for c in cols if 'warps_active' in c.lower() and 'pct' in c.lower()), None)

def safe_float(s):
    try: return float(s.replace(',',''))
    except: return 0.0

print(f\"{'Kernel':<55} {'N':>4} {'Avg(us)':>8} {'DRAM R(KB)':>10} {'DRAM W(KB)':>10} {'DRAM%':>6} {'SM%':>5} {'Occ%':>5}\")
print('=' * 107)

for kname, invocations in sorted(kernels.items(), key=lambda x: -sum(safe_float(r.get(time_col,'0')) for r in x[1])):
    n = len(invocations)
    avg_time_us = sum(safe_float(r.get(time_col,'0')) for r in invocations) / n / 1000  # ns->us
    avg_dram_r = sum(safe_float(r.get(dram_r_col,'0')) for r in invocations) / n / 1024 if dram_r_col else 0
    avg_dram_w = sum(safe_float(r.get(dram_w_col,'0')) for r in invocations) / n / 1024 if dram_w_col else 0
    avg_dram_pct = sum(safe_float(r.get(dram_pct_col,'0')) for r in invocations) / n if dram_pct_col else 0
    avg_sm_pct = sum(safe_float(r.get(sm_pct_col,'0')) for r in invocations) / n if sm_pct_col else 0
    avg_occ = sum(safe_float(r.get(occ_col,'0')) for r in invocations) / n if occ_col else 0
    short_name = kname.split('(')[0][-54:]  # truncate template args
    print(f'{short_name:<55} {n:>4} {avg_time_us:>8.2f} {avg_dram_r:>10.1f} {avg_dram_w:>10.1f} {avg_dram_pct:>5.1f}% {avg_sm_pct:>4.1f}% {avg_occ:>4.1f}%')

# Summary: total GPU time
if time_col:
    total_us = sum(safe_float(r.get(time_col,'0')) for r in rows) / 1000
    print(f\"{'TOTAL':<55} {len(rows):>4} {total_us:>8.1f} us total\")
" 2>/dev/null || echo "(parsing failed — check $OUTDIR/${NAME}.csv manually)"

# Unlock clocks
if [ "${LOCKED:-0}" = "1" ]; then
    nvidia-smi -rgc &>/dev/null || true
fi
