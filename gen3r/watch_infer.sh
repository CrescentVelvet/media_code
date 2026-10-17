#!/usr/bin/env bash
# 观察 Gen3R 推理：日志字节增长 + 进度条 + GPU
LOG="${LOG:-/home/velvet/output/gen3r_1view_v3.log}"
A=$(wc -c < "$LOG")
sleep 40
B=$(wc -c < "$LOG")
echo "bytes: $A -> $B  (delta $((B-A)))"
echo "--- tail ---"
tail -c 200 "$LOG" | tr '\r' '\n' | tail -3
echo "--- gpu ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
