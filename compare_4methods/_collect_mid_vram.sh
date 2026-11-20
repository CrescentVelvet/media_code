#!/bin/bash
# 中间档显存峰值（vram 文件每行一个 MiB 数字）+ 各步耗时线索
B=/home/velvet/output/batch_4methods
for m in is vggt da3 da3gs; do
  V="$B/mid_${m}_佳佳抱枕_mid.vram"
  P=$(awk 'BEGIN{m=0}{if($1+0>m)m=$1+0}END{printf "%.0f", m}' "$V" 2>/dev/null)
  N=$(wc -l < "$V" 2>/dev/null)
  echo "$m  峰值=${P}MiB ($(( ${P:-0} / 1024 ))GB)  采样${N}次"
done
