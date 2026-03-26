#!/bin/bash
# Profile GWEN decode with nsys (timeline + kernel stats)
# Usage: ./scripts/nsys_profile.sh [output_name]
#
# This captures a full timeline trace suitable for viewing in nsys-ui
# and also prints a kernel-level summary sorted by total GPU time.

set -euo pipefail

MODEL="${1:-$HOME/models/Qwen3.5-9B-UD-Q4_K_XL.gguf}"
GWEN="./build/gwen"
NAME="${2:-gwen_decode}"
OUTDIR="profiles"
mkdir -p "$OUTDIR"

echo "=== Nsight Systems Profile: $NAME ==="
echo ""

# Lock GPU clocks for reproducibility (requires root/admin)
if nvidia-smi -lgc 3105,3105 &>/dev/null; then
    echo "GPU clocks locked to 3105 MHz"
    LOCKED=1
else
    echo "WARNING: Could not lock GPU clocks. Results may vary."
    echo "  Run: sudo nvidia-smi -lgc 3105,3105"
    LOCKED=0
fi

# Capture trace — 100 token generation gives many decode iterations
# --trace=cuda captures kernel launches, memcpy, memset
# --cudabacktrace=kernel enables source-level correlation
# --gpu-metrics-device=all captures SM/memory utilization
nsys profile \
    --trace=cuda \
    --cuda-graph-trace=node \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output="$OUTDIR/$NAME" \
    "$GWEN" --model "$MODEL" \
    --greedy --max-predict 100 --benchmark \
    "1 2 3 4 5 6 7 8" 2>&1 | head -3

echo ""
echo "Trace saved to: $OUTDIR/${NAME}.nsys-rep"
echo ""

# Print kernel stats summary
echo "=== Kernel Statistics (sorted by total GPU time) ==="
nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    "$OUTDIR/${NAME}.nsys-rep" 2>/dev/null \
| python3 -c "
import sys, csv
reader = csv.DictReader(sys.stdin)
rows = list(reader)
# Find the time and count columns (names vary by nsys version)
time_col = next((c for c in rows[0].keys() if 'Total Time' in c or 'total_time' in c.lower()), None)
count_col = next((c for c in rows[0].keys() if 'Instances' in c or 'count' in c.lower()), None)
name_col = next((c for c in rows[0].keys() if 'Name' in c or 'name' in c.lower()), None)
avg_col = next((c for c in rows[0].keys() if 'Avg' in c or 'avg' in c.lower() or 'Average' in c), None)
if not all([time_col, count_col, name_col]):
    print('Could not parse nsys output columns:', list(rows[0].keys()))
    sys.exit(1)
# Sort by total time descending
rows.sort(key=lambda r: float(r[time_col].replace(',','')), reverse=True)
print(f\"{'Kernel':<60} {'Count':>6} {'Total (us)':>12} {'Avg (us)':>10} {'%':>6}\")
print('=' * 98)
total_us = sum(float(r[time_col].replace(',','')) / 1000 for r in rows)
for r in rows[:30]:
    t_us = float(r[time_col].replace(',','')) / 1000
    cnt = int(r[count_col].replace(',',''))
    avg_us = t_us / cnt if cnt > 0 else 0
    name = r[name_col][:59]
    pct = t_us / total_us * 100 if total_us > 0 else 0
    print(f'{name:<60} {cnt:>6} {t_us:>12.1f} {avg_us:>10.2f} {pct:>5.1f}%')
print(f\"{'TOTAL':<60} {'':<6} {total_us:>12.1f}\")
" 2>/dev/null || echo "(Install python3 to see formatted kernel stats)"

echo ""
echo "=== Launch Overhead Analysis ==="
nsys stats \
    --report cuda_api_sum \
    --format csv \
    "$OUTDIR/${NAME}.nsys-rep" 2>/dev/null \
| python3 -c "
import sys, csv
reader = csv.DictReader(sys.stdin)
rows = list(reader)
name_col = next((c for c in rows[0].keys() if 'Name' in c or 'name' in c.lower()), None)
time_col = next((c for c in rows[0].keys() if 'Total Time' in c or 'total_time' in c.lower()), None)
count_col = next((c for c in rows[0].keys() if 'Instances' in c or 'count' in c.lower()), None)
if not all([time_col, count_col, name_col]):
    sys.exit(0)
rows.sort(key=lambda r: float(r[time_col].replace(',','')), reverse=True)
print(f\"{'CUDA API':<40} {'Count':>8} {'Total (ms)':>12} {'Avg (us)':>10}\")
print('=' * 74)
for r in rows[:15]:
    t_ms = float(r[time_col].replace(',','')) / 1e6
    cnt = int(r[count_col].replace(',',''))
    avg_us = float(r[time_col].replace(',','')) / cnt / 1000 if cnt > 0 else 0
    print(f\"{r[name_col]:<40} {cnt:>8} {t_ms:>12.2f} {avg_us:>10.2f}\")
" 2>/dev/null || true

# Unlock clocks
if [ "${LOCKED:-0}" = "1" ]; then
    nvidia-smi -rgc &>/dev/null || true
    echo ""
    echo "GPU clocks unlocked."
fi

echo ""
echo "View full trace: nsys-ui $OUTDIR/${NAME}.nsys-rep"
