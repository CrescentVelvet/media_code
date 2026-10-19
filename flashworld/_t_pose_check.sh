#!/usr/bin/env bash
SCRIPT_DIR=/mnt/c/code/media_code/flashworld
export SCRIPT_DIR
source "$SCRIPT_DIR/_env.sh"
python "$SCRIPT_DIR/_t_pose_check.py" \
    "/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕" \
    /mnt/d/output/flashworld_input/jiabao_png \
    /mnt/d/output/flashworld_results/05_pose_check \
    0 12 23 41 80
