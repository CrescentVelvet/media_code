#!/usr/bin/env bash
# 13_run_nvs_first.sh — ref=first 口径 NVS 评估：32 视角高斯 → 渲染 125 视角，SH 不旋转
# 用法: bash 13_run_nvs_first.sh [PLY_DIR] [NPZ32] [EXTRA_SCALE]
set +u
PLY_DIR="${1:-/home/velvet/output/da3_human_results/bench_first_32v}"
NPZ32="${2:-$PLY_DIR/cameras.npz}"
EXTRA_SCALE="${3:-1.0}"
R=/home/velvet/output/da3_human_results
SCRIPT=/mnt/c/code/media_code/da3_human/12_eval_nvs_first.py

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vggt_human || { echo "conda activate 失败"; exit 1; }

PLY="$PLY_DIR/gs_max/0000.ply" \
NPZ32="$NPZ32" \
NPZ125="$R/bench_first_125v/cameras.npz" \
IMAGES_DIR=/home/velvet/output/vggt_human_results/source/images \
W=1440 H=1920 EXTRA_SCALE="$EXTRA_SCALE" \
python "$SCRIPT"
