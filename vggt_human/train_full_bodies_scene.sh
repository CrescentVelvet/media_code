#!/usr/bin/env bash
# body×3 + scene 完整训练：30k iter，densify 15k（区域外无梯度不 densify，已验证）
set -o pipefail
cd /mnt/c/code/media_code/vggt_human

for spec in "body 00" "body 01" "body 02" "scene"; do
    set -- $spec
    KIND=$1; PID=${2:-}
    echo ""
    echo "=================================================================="
    echo ">>> 完整训练 $KIND ${PID:+p$PID} (30k iter, densify 15k)"
    echo "=================================================================="
    if RESULTS_DIR=/mnt/d/output/vggt_human_ms MODEL_KIND=$KIND PID=${PID:-00} ITERATIONS=30000 \
       LR_SCALE=1.0 DENSIFY_UNTIL=15000 bash 03i_train_three_models.sh; then
        echo "✅ $KIND ${PID:+p$PID} done"
    else
        echo "❌ $KIND ${PID:+p$PID} FAILED"
    fi
done
echo ""
echo "全部完成"
