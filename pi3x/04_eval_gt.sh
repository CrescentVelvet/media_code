#!/usr/bin/env bash
# 04_eval_gt.sh — Pi3X 预测 vs 采集包 GT（Sim(3) 对齐后 ATE/旋转/内参误差）。
#
# 用法:
#   SCENE=jiajia_baozhen TAG=nocond bash pi3x/04_eval_gt.sh
#   SCENE=jiajia_baozhen TAG=cond   bash pi3x/04_eval_gt.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE="${SCENE:-jiajia_baozhen}"
TAG="${TAG:-nocond}"
SCENE_DIR="${SCENE_DIR:-$RESULTS_DIR/$SCENE}"
PRED_DIR="${PRED_DIR:-$SCENE_DIR/$TAG}"

echo "🚀 [04] GT 评测  scene=$SCENE  tag=$TAG"

if [ ! -f "$PRED_DIR/predictions.npz" ]; then
    echo "❌ ERROR: 缺 predictions.npz（先跑 03）: $PRED_DIR" >&2
    exit 1
fi
if [ ! -f "$SCENE_DIR/conditions.npz" ]; then
    echo "❌ ERROR: 缺 conditions.npz（先跑 02）: $SCENE_DIR" >&2
    exit 1
fi

PRED_DIR="$PRED_DIR" SCENE_DIR="$SCENE_DIR" python "$SCRIPT_DIR/eval_gt.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 [04] Done."
