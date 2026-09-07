#!/usr/bin/env bash
# 训练剩余 6 个模型（p00 head 已完成 4000 iter 验证）
set -o pipefail
cd /mnt/c/code/media_code/vggt_human

for spec in "head 01" "head 02" "body 00" "body 01" "body 02" "scene"; do
    set -- $spec
    KIND=$1; PID=${2:-}
    echo ""
    echo "=================================================================="
    echo ">>> 训练 $KIND ${PID:+p$PID}"
    echo "=================================================================="
    if RESULTS_DIR=/mnt/d/output/vggt_human_ms MODEL_KIND=$KIND PID=${PID:-00} ITERATIONS=4000 \
       LR_SCALE=0.1 DENSIFY_UNTIL=0 bash 03i_train_three_models.sh; then
        echo "✅ $KIND ${PID:+p$PID} done"
    else
        echo "❌ $KIND ${PID:+p$PID} FAILED"
    fi
done
echo ""
echo "全部完成"
