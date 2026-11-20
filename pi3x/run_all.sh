#!/usr/bin/env bash
# run_all.sh — 一键：02 数据准备 → 03 纯图像推理 → 03 条件注入推理 → 04 两组 GT 评测。
#
# 用法:
#   GPU=0 SCENE=jiajia_baozhen \
#   RAW_DIR="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕" \
#     bash pi3x/run_all.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE="${SCENE:-jiajia_baozhen}"
RAW_DIR="${RAW_DIR:-}"

if [ -z "$RAW_DIR" ]; then
    echo "❌ ERROR: 必须指定 RAW_DIR（采集包路径）" >&2
    exit 1
fi

echo "🚀 [run_all] scene=$SCENE"

SCENE="$SCENE" RAW_DIR="$RAW_DIR" bash "$SCRIPT_DIR/02_prepare_images.sh" || exit 1

SCENE="$SCENE" USE_CONDITIONS=0 TAG=nocond bash "$SCRIPT_DIR/03_run_inference.sh" || exit 1
SCENE="$SCENE" TAG=nocond bash "$SCRIPT_DIR/04_eval_gt.sh" || exit 1

SCENE="$SCENE" USE_CONDITIONS=1 TAG=cond bash "$SCRIPT_DIR/03_run_inference.sh" || exit 1
SCENE="$SCENE" TAG=cond bash "$SCRIPT_DIR/04_eval_gt.sh" || exit 1

echo "🎉 [run_all] Done. 结果在 $RESULTS_DIR/$SCENE/{nocond,cond}/"
