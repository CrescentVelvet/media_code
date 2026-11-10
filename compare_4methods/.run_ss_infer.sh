#!/bin/bash
# 重启 Scene Splatter 推理（佳佳抱枕，220° 轨迹，CAM_INTERVAL=3，约 1~1.5h）
IMAGE=/home/velvet/output/flashworld_results/_input/jiajia.png \
CAMERA_PATH=$HOME/output/scene_splatter_results/traj/jiajia_orbit_220_264.pkl.gz \
CAM_INTERVAL=3 \
bash /mnt/c/code/media_code/scene_splatter/02_run_inference.sh \
  > "$HOME/output/ss_02_jiajia.log" 2>&1
echo "SS_EXIT=$?" >> "$HOME/output/ss_02_jiajia.log"
