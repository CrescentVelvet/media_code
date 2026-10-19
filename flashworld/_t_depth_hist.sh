#!/usr/bin/env bash
SCRIPT_DIR=/mnt/c/code/media_code/flashworld
export SCRIPT_DIR
source "$SCRIPT_DIR/_env.sh"
python "$SCRIPT_DIR/_t_depth_hist.py" \
    /mnt/d/output/flashworld_results/03_gaussians_narrow/jiabao_narrow24/gaussians.ply \
    /mnt/d/output/flashworld_input/json_narrow/jiabao_narrow24.json
