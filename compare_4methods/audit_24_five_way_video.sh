#!/usr/bin/env bash
# AUDIT-24 一键：五方法同一相机轨迹对比视频（252.7° 观测弧）
# 先把各方法 Umeyama 到 DA3-GS 的相机系（含尺度），再逐帧横排五联。
set -o pipefail
B=/home/velvet/output
OUT="${OUT:-$B/da3_gs_results/佳佳抱枕_p1m_off/orbit_video}"
CODE=/mnt/c/code/media_code
ENVS=/home/velvet/miniconda3/envs/querysplat

if [ -f /home/velvet/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1090
  source /home/velvet/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENVS"
fi
export CUDA_HOME="$ENVS"
export CPATH="$ENVS/targets/x86_64-linux/include"
export LIBRARY_PATH="$ENVS/targets/x86_64-linux/lib"

OUT="$OUT" NFRAMES="${NFRAMES:-240}" FPS="${FPS:-30}" CELLW="${CELLW:-372}" \
RADIUS_K="${RADIUS_K:-1.06}" ELEV="${ELEV:-}" BG="${BG:-white}" ONLY="${ONLY:-}" \
  exec python "$CODE/compare_4methods/audit_24_five_way_video.py"
