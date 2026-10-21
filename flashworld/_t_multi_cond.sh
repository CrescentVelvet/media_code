#!/usr/bin/env bash
# 多图条件实验：打补丁 → 2cond / 3cond 各跑一次推理 → 渲染关键帧对比
SCRIPT_DIR=/mnt/c/code/media_code/flashworld
export SCRIPT_DIR
source "$SCRIPT_DIR/_env.sh"

echo "===== 0. 打多图条件补丁 ====="
python "$SCRIPT_DIR/04_patch_multi_image.py" || exit 1

for EXP in 2 3; do
    echo ""
    echo "===== 1. 推理：${EXP}cond ====="
    INPUT_DIR=/mnt/d/output/flashworld_input/json_multi${EXP} \
    OUT_DIR=/mnt/d/output/flashworld_results/06_multi_cond \
    VIDEO=0 PLY=1 SPZ=0 \
        bash "$SCRIPT_DIR/02_run_inference.sh" || exit 1

    echo ""
    echo "===== 2. 渲染：${EXP}cond（帧 0/6/11/17/23，不剪枝）====="
    FRAMES=0,6,11,17,23 OPACITY_MIN=0 \
        python "$SCRIPT_DIR/03_render_ply.py" \
        /mnt/d/output/flashworld_results/06_multi_cond/jiabao_${EXP}cond/gaussians.ply \
        /mnt/d/output/flashworld_input/json_multi${EXP}/jiabao_${EXP}cond.json \
        /mnt/d/output/flashworld_results/06_multi_cond/render_${EXP}cond || exit 1
done
echo ""
echo "ALL DONE"
