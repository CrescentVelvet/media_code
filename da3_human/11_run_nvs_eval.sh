#!/usr/bin/env bash
# 11_run_nvs_eval.sh — 跑 NVS 口径评估（32 视角输入 → 125 视角渲染）
# 用法: wsl -d Ubuntu2404 -- bash /mnt/c/code/media_code/da3_human/11_run_nvs_eval.sh [PLY标签]
set +u
TAG="${1:-max}"
BASE=${BASE:-/home/velvet/output/da3_human_results/bench_first_32v_fast}
SCRIPT=/mnt/c/code/media_code/da3_human/10_eval_nvs_32to125.py

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vggt_human || { echo "❌ conda activate 失败"; exit 1; }

PLY="$BASE/gs_$TAG/0000.ply" \
D3NPZ="$BASE/cameras.npz" \
COLMAP_DIR=/home/velvet/output/vggt_human_results/source/sparse/0 \
IMAGES_DIR=/home/velvet/output/vggt_human_results/source/images \
N_INPUT=32 W=1440 H=1920 \
python "$SCRIPT"
