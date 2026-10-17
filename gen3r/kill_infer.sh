#!/usr/bin/env bash
# 停掉正在跑的 Gen3R 推理（按 PID 精确杀，避免 pkill 误伤本 shell）
for p in $(ps -eo pid,args | grep 'infer\.py --pretrained_model_name_or_path' | grep -v grep | awk '{print $1}'); do
    echo "kill $p"
    kill -9 "$p"
done
sleep 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader
