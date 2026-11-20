#!/bin/bash
# 收集中间档（1008）四方法批量结果：各步结局 + 显存峰值 + 耗时
B=/home/velvet/output/batch_4methods
echo "===== 各步日志尾（判断成败） ====="
for m in is vggt da3 da3gs; do
  L="$B/mid_${m}_佳佳抱枕_mid.log"
  echo "----- $m (size=$(stat -c%s "$L" 2>/dev/null)) -----"
  tail -6 "$L" 2>/dev/null
  echo ""
done
echo "===== 显存峰值（nvidia-smi 采样） ====="
for m in is vggt da3 da3gs; do
  V="$B/mid_${m}_佳佳抱枕_mid.vram"
  echo "--- $m 峰值 ---"
  # 取 MiB 列最大值
  awk 'NR>1 {for(i=1;i<=NF;i++) if($i ~ /MiB/) {gsub(/MiB/,"",$i); if($i+0>m) m=$i+0}} END {printf "  峰值 %.0f MiB\n", m}' "$V" 2>/dev/null || echo "  (无采样文件)"
done
