#!/usr/bin/env bash
# 07_monitor_gpu.sh — WSL 侧训练监控（鉴别显存卡顿根因用）
# 每 INTERVAL 秒记录一行：时间戳 / GPU util+显存 / 显存进程列表 / 内存+swap / python 进程数。
# 用法：OUT=/mnt/d/output/xxx.log INTERVAL=10 bash 07_monitor_gpu.sh
set -o pipefail

OUT="${OUT:-/mnt/d/output/splannequin_monitor.log}"
INTERVAL="${INTERVAL:-10}"

echo "🚀 [07] GPU/内存监控启动 → $OUT（每 ${INTERVAL}s）"
while :; do
    {
        echo "=== $(date '+%m-%d %H:%M:%S') ==="
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
        echo "-- compute-apps (WSL 视角) --"
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || echo "(none)"
        echo "-- mem --"
        free -m | awk 'NR==2{printf "mem_used=%dM avail=%dM ", $3, $7} NR==3{printf "swap_used=%dM", $3} END{print ""}'
        echo "-- python procs --"
        ps -eo pid,etime,rss,cmd | grep -E "python.*train\.py" | grep -v grep || echo "(none)"
    } >> "$OUT" 2>&1
    sleep "$INTERVAL"
done
