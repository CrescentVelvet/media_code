#!/usr/bin/env bash
set -uo pipefail
LOG=/mnt/c/code/media_code/compare_4methods/_probe_shfull.log
exec > "$LOG" 2>&1
O=/home/velvet/output/da3_gs_results

for d in 佳佳抱枕_shfull 佳佳抱枕_sh_ab 佳佳抱枕_ref0_sh; do
  echo "=== $d ==="
  ls -la "$O/$d/gs_ply_full/" 2>/dev/null | head -3
  ls -la "$O/$d/cameras.npz" 2>/dev/null
done
