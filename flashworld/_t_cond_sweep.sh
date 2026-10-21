#!/usr/bin/env bash
# 条件帧数扫描：5/8/12/24cond 推理 + 全 24 帧渲染（2/3cond 也补全帧渲染）
SCRIPT_DIR=/mnt/c/code/media_code/flashworld
export SCRIPT_DIR
source "$SCRIPT_DIR/_env.sh"

for EXP in 5 8 12 24; do
    echo ""
    echo "########## 推理 ${EXP}cond ##########"
    INPUT_DIR=/mnt/d/output/flashworld_input/json_multi${EXP} \
    OUT_DIR=/mnt/d/output/flashworld_results/06_multi_cond \
    VIDEO=0 PLY=1 SPZ=0 \
        bash "$SCRIPT_DIR/02_run_inference.sh" || exit 1
done

for EXP in 2 3 5 8 12 24; do
    echo ""
    echo "########## 渲染 ${EXP}cond（全 24 帧）##########"
    OPACITY_MIN=0 \
        python "$SCRIPT_DIR/03_render_ply.py" \
        /mnt/d/output/flashworld_results/06_multi_cond/jiabao_${EXP}cond/gaussians.ply \
        /mnt/d/output/flashworld_input/json_multi${EXP}/jiabao_${EXP}cond.json \
        /mnt/d/output/flashworld_results/06_multi_cond/render_${EXP}cond || exit 1
done
echo ""
echo "ALL DONE"
