#!/bin/bash
# 中间档各步耗时提取（日志里的 ⏱️ 行）
B=/home/velvet/output/batch_4methods
for m in is vggt da3 da3gs; do
  L="$B/mid_${m}_佳佳抱枕_mid.log"
  echo "===== $m ====="
  grep -E "⏱️|端到端|耗时|前向|Done\." "$L" 2>/dev/null | tail -8
  echo ""
done
