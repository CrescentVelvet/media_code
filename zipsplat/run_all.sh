#!/usr/bin/env bash
# run_all.sh — ZipSplat 佳佳抱枕：环境检查 → 权重 → 16v → 独立留出评测
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
RAW_DIR="${RAW_DIR:?❌ 请指定 RAW_DIR=/mnt/d/dataset/手部微动样例/<佳佳抱枕包>}"
INSTALL_DEPS="${INSTALL_DEPS:-0}" bash "$SCRIPT_DIR/00a_setup_env.sh" || exit 1
bash "$SCRIPT_DIR/01_download_models.sh" || exit 1
RAW_DIR="$RAW_DIR" N_VIEWS=16 bash "$SCRIPT_DIR/01b_prepare_jiajia.sh" || exit 1
RAW_DIR="$RAW_DIR" FRAMES=3,14,19,36,47,62,74,79 \
  CAP_DIR="$RESULTS_DIR/_captures/jiajia_holdout" bash "$SCRIPT_DIR/01b_prepare_jiajia.sh" || exit 1
IMAGES_DIR="$RESULTS_DIR/_captures/jiajia_baozhen/images" SCENE_NAME=jiajia_16v \
  MAX_VIEWS=16 BA=1 RENDER_VIDEO=1 bash "$SCRIPT_DIR/02_run_inference.sh" || exit 1
RESULT="$RESULTS_DIR/jiajia_16v" bash "$SCRIPT_DIR/03_evaluate_jiajia.sh" || exit 1
# 独立 BA=0 进程保留前馈相机；不能把 BA 后相机配前馈高斯叫纯前馈自洽。
IMAGES_DIR="$RESULTS_DIR/_captures/jiajia_baozhen/images" SCENE_NAME=jiajia_16v_feedforward \
  MAX_VIEWS=16 BA=0 RENDER_VIDEO=0 bash "$SCRIPT_DIR/02_run_inference.sh" || exit 1
RESULT="$RESULTS_DIR/jiajia_16v_feedforward" bash "$SCRIPT_DIR/03_evaluate_jiajia.sh" || exit 1
bash "$SCRIPT_DIR/08_move_output.sh" || exit 1
