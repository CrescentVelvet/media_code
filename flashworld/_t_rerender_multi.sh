#!/usr/bin/env bash
# 重渲 2cond/3cond（修正 crop 内参后）
SCRIPT_DIR=/mnt/c/code/media_code/flashworld
export SCRIPT_DIR
source "$SCRIPT_DIR/_env.sh"

for EXP in 2 3; do
    echo "===== 重渲 ${EXP}cond ====="
    FRAMES=0,6,11,17,23 OPACITY_MIN=0 \
        python "$SCRIPT_DIR/03_render_ply.py" \
        /mnt/d/output/flashworld_results/06_multi_cond/jiabao_${EXP}cond/gaussians.ply \
        /mnt/d/output/flashworld_input/json_multi${EXP}/jiabao_${EXP}cond.json \
        /mnt/d/output/flashworld_results/06_multi_cond/render_${EXP}cond || exit 1
done
echo "ALL DONE"
