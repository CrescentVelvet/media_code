#!/usr/bin/env bash
set -uo pipefail
LOG=/mnt/c/code/media_code/compare_4methods/_probe_ply_sizes.log
exec > "$LOG" 2>&1
O=/home/velvet/output/da3_gs_results

for d in 佳佳抱枕_shfull 佳佳抱枕_sh_ab 佳佳抱枕_ref0_sh 佳佳抱枕; do
  for sub in gs_ply_full gs_ply gs_ply_dc; do
    p="$O/$d/$sub/0000.ply"
    if [ -f "$p" ]; then
      sz=$(stat -c%s "$p")
      # 读 PLY 头里的元素数
      n=$(head -c 200000 "$p" | grep -m1 'element vertex' | awk '{print $3}')
      echo "$d/$sub  ${sz} bytes  ${n} verts"
    fi
  done
done
