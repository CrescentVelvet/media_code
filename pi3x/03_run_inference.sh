#!/usr/bin/env bash
# 03_run_inference.sh — Pi3X 前馈推理。
#
# 用法（纯图像模式）:
#   GPU=0 SCENE=jiajia_baozhen bash pi3x/03_run_inference.sh
# 用法（注入 GT 内参+位姿条件，Pi3X 招牌能力）:
#   GPU=0 SCENE=jiajia_baozhen USE_CONDITIONS=1 bash pi3x/03_run_inference.sh
# 自定义:
#   PIXEL_LIMIT=700000 CONF_THRES=0.05 ... bash pi3x/03_run_inference.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE="${SCENE:-jiajia_baozhen}"
USE_CONDITIONS="${USE_CONDITIONS:-0}"
SCENE_DIR="${SCENE_DIR:-$RESULTS_DIR/$SCENE}"
IMAGES_DIR="${IMAGES_DIR:-$SCENE_DIR/images}"

if [ "$USE_CONDITIONS" = "1" ]; then
    TAG="${TAG:-cond}"
    export CONDITIONS_NPZ="${CONDITIONS_NPZ:-$SCENE_DIR/conditions.npz}"
    if [ ! -f "$CONDITIONS_NPZ" ]; then
        echo "❌ ERROR: conditions.npz 不存在（先跑 02）: $CONDITIONS_NPZ" >&2
        exit 1
    fi
else
    TAG="${TAG:-nocond}"
fi
SAVE_DIR="${SAVE_DIR:-$SCENE_DIR/$TAG}"

echo "🚀 [03] Pi3X 推理  scene=$SCENE  mode=$TAG"
echo "  🖼️  输入: $IMAGES_DIR"
echo "  💾 输出: $SAVE_DIR"

if [ ! -d "$IMAGES_DIR" ]; then
    echo "❌ ERROR: 图像目录不存在（先跑 02）: $IMAGES_DIR" >&2
    exit 1
fi
if [ ! -f "$PI3X_CKPT" ]; then
    echo "❌ ERROR: 权重不存在（先跑 01）: $PI3X_CKPT" >&2
    exit 1
fi

IMAGES_DIR="$IMAGES_DIR" SAVE_DIR="$SAVE_DIR" \
  python "$SCRIPT_DIR/pi3x_recon.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 [03] Done."
