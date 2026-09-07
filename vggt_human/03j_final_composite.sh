#!/usr/bin/env bash
# 03j_final_composite.sh — 三模型最终合成评估（head 10k + body/scene 30k）
#
# 前置：body×3 + scene 30k（train_full_bodies_scene.sh）、head×3 10k
#       （train_heads_refine.sh）全部完成
# 输出：合成 mosaic + 全帧/body区/head区 PSNR + 04b 基线同帧对比
set -eo pipefail
cd /mnt/c/code/media_code/vggt_human

RESULTS_DIR=/mnt/d/output/vggt_human_ms

# 检查全部 checkpoint 就绪
MISSING=0
for pid in 00 01 02; do
    for f in "03i_head_p${pid}/point_cloud/iteration_10000/point_cloud.ply" \
             "03i_body_p${pid}/point_cloud/iteration_30000/point_cloud.ply"; do
        if [ ! -f "$RESULTS_DIR/$f" ]; then
            echo "❌ 缺 $RESULTS_DIR/$f"
            MISSING=1
        fi
    done
done
if [ ! -f "$RESULTS_DIR/03i_scene/point_cloud/iteration_30000/point_cloud.ply" ]; then
    echo "❌ 缺 scene 30k"
    MISSING=1
fi
if [ "$MISSING" = "1" ]; then
    echo "先等训练完成（train_full_bodies_scene.sh / train_heads_refine.sh）"
    exit 1
fi

echo "=================================================================="
echo "三模型最终合成（head 10k + body 30k + scene 30k）vs 04b 基线"
echo "=================================================================="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vggt_human

OMP_NUM_THREADS=8 RESULTS_DIR=$RESULTS_DIR \
HEAD_ITERS=10000 BODY_ITERS=30000 SCENE_ITERS=30000 \
N_VIS=8 BASELINE=1 python render_composite.py 2>&1 | tail -25

echo ""
echo "=================================================================="
echo "各模型最终区域内 PSNR（SH3 评估）"
echo "=================================================================="
for pid in 00 01 02; do
    OMP_NUM_THREADS=8 RESULTS_DIR=$RESULTS_DIR KIND=head PID=$pid ITERS=10000 N_VIS=8 \
    python eval_region_psnr.py 2>&1 | grep -E '=== |med=' | tr '\n' ' '
    echo ""
    OMP_NUM_THREADS=8 RESULTS_DIR=$RESULTS_DIR KIND=body PID=$pid ITERS=30000 N_VIS=8 \
    python eval_region_psnr.py 2>&1 | grep -E '=== |med=' | tr '\n' ' '
    echo ""
done
OMP_NUM_THREADS=8 RESULTS_DIR=$RESULTS_DIR KIND=scene ITERS=30000 N_VIS=8 \
python eval_region_psnr.py 2>&1 | grep -E '=== |med=' | tr '\n' ' '
echo ""
echo ""
echo "🖼️ 合成图: $RESULTS_DIR/03i_composite_vis/composite_mosaic.png"
