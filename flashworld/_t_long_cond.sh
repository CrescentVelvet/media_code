#!/usr/bin/env bash
# 长轨迹多条件扫描：traj41_cond12 / traj41_cond41 / traj82_cond24 / traj82_cond41 / traj82_cond82
# 5 组一次推理（cli.py 循环 json_lc 里 5 个 json），然后逐组全帧渲染。
SCRIPT_DIR=/mnt/c/code/media_code/flashworld
export SCRIPT_DIR
source "$SCRIPT_DIR/_env.sh"

echo "===== 0. 确认多图补丁已打（幂等） ====="
python "$SCRIPT_DIR/04_patch_multi_image.py" || exit 1

echo ""
echo "===== 1. 推理（5 组，模型只加载一次） ====="
INPUT_DIR=/mnt/d/output/flashworld_input/json_lc \
OUT_DIR=/mnt/d/output/flashworld_results/07_long_cond \
VIDEO=0 PLY=1 SPZ=0 \
    bash "$SCRIPT_DIR/02_run_inference.sh" || exit 1

for EXP in traj41_cond12 traj41_cond41 traj82_cond24 traj82_cond41 traj82_cond82; do
    PLY=/mnt/d/output/flashworld_results/07_long_cond/$EXP/gaussians.ply
    [ -f "$PLY" ] || { echo "⚠️ 缺 $PLY，跳过渲染"; continue; }
    echo ""
    echo "===== 2. 渲染：$EXP（全帧） ====="
    OPACITY_MIN=0 \
        python "$SCRIPT_DIR/03_render_ply.py" \
        "$PLY" \
        /mnt/d/output/flashworld_input/json_lc/$EXP.json \
        /mnt/d/output/flashworld_results/07_long_cond/render_$EXP || exit 1
done
echo ""
echo "ALL DONE"
