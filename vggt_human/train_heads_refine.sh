#!/usr/bin/env bash
# head p01/p02 补 10k 精修（等待 body×3+scene 30k 完成后运行）
# head 无 densify，4k→10k 纯参数精修（SH3 评估下 p00: 23.99→24.72dB）
set -eo pipefail
cd /mnt/c/code/media_code/vggt_human

# 等待 train_full_bodies_scene.sh 结束（最多等 3 小时）
echo "等待 body/scene 训练进程结束..."
for i in $(seq 1 360); do
    if ! pgrep -f "train_full_bodies_scene.sh" > /dev/null 2>&1; then
        echo "body/scene 训练已结束"
        break
    fi
    if [ "$i" -eq 360 ]; then
        echo "⚠️ 等待超时（3h），仍有训练进程，退出"
        exit 1
    fi
    sleep 30
done

for pid in 01 02; do
    echo ""
    echo "=================================================================="
    echo ">>> head p$pid 10k 精修（从 4k checkpoint 续训）"
    echo "=================================================================="
    if RESULTS_DIR=/mnt/d/output/vggt_human_ms MODEL_KIND=head PID=$pid ITERATIONS=10000 \
       START_PLY=/mnt/d/output/vggt_human_ms/03i_head_p$pid/point_cloud/iteration_4000/point_cloud.ply \
       LR_SCALE=1.0 DENSIFY_UNTIL=0 \
       bash 03i_train_three_models.sh; then
        echo "✅ head p$pid 10k done"
    else
        echo "❌ head p$pid FAILED"
    fi
done
echo ""
echo "head 精修全部完成"
