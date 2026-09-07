#!/usr/bin/env bash
# head×3 完整训练：10k iter（head 已验证 4k 收敛良好，10k 精修）
set -o pipefail
cd /mnt/c/code/media_code/vggt_human
for PID in 00 01 02; do
    echo ">>> head p$PID 10k iter"
    RESULTS_DIR=/mnt/d/output/vggt_human_ms MODEL_KIND=head PID=$PID ITERATIONS=10000 \
        LR_SCALE=1.0 DENSIFY_UNTIL=0 bash 03i_train_three_models.sh || echo "❌ head p$PID FAILED"
done
echo "heads done"
