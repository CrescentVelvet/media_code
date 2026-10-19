#!/usr/bin/env bash
# opacity 剪枝去重影对照实验：同一 PLY、同一批帧，分别用 OPACITY_MIN=0/0.1/0.2/0.3 渲染
SCRIPT_DIR=/mnt/c/code/media_code/flashworld
export SCRIPT_DIR
source "$SCRIPT_DIR/_env.sh"

PLY=/mnt/d/output/flashworld_results/03_gaussians_narrow/jiabao_narrow24/gaussians.ply
JSON=/mnt/d/output/flashworld_input/json_narrow/jiabao_narrow24.json
export FRAMES=0,6,12,18

for T in 0.1 0.2 0.3; do
    OUT=/mnt/d/output/flashworld_results/04_render_prune/op$T
    echo "===== OPACITY_MIN=$T ====="
    OPACITY_MIN=$T python "$SCRIPT_DIR/03_render_ply.py" "$PLY" "$JSON" "$OUT"
done
echo "ALL DONE"
