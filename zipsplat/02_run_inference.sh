#!/usr/bin/env bash
# 02_run_inference.sh — 任意图片目录 → 官方 ZipSplat 前馈 + 可选 TokenBA
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

IMAGES_DIR="${IMAGES_DIR:?❌ 指定 IMAGES_DIR=/path/to/images}"
SCENE_NAME="${SCENE_NAME:-$(basename "$(dirname "$IMAGES_DIR")")}"
export OUT_DIR="${OUT_DIR:-$RESULTS_DIR/$SCENE_NAME}"
export IMAGES_DIR WEIGHT_PATH ZIPSPLAT_REPO
export MAX_VIEWS="${MAX_VIEWS:-16}"
export BA="${BA:-1}"
export SAVE_FEEDFORWARD="${SAVE_FEEDFORWARD:-1}"
export RENDER_VIDEO="${RENDER_VIDEO:-0}"

[ -d "$IMAGES_DIR" ] || { echo "❌ 图片目录不存在: $IMAGES_DIR" >&2; exit 1; }
[ -f "$WEIGHT_PATH" ] || { echo "❌ 缺权重 $WEIGHT_PATH，先跑 01_download_models.sh" >&2; exit 1; }
[ -f "$ZIPSPLAT_REPO/zipsplat/predictor.py" ] || { echo "❌ 缺官方仓" >&2; exit 1; }
mkdir -p "$OUT_DIR"
echo "🚀 [02] ZipSplat 重建"
echo "  📁 输入: $IMAGES_DIR (MAX_VIEWS=$MAX_VIEWS)"
echo "  🏋️ 权重: $WEIGHT_PATH"
echo "  💾 输出: $OUT_DIR"
echo "  🔧 TokenBA=$BA; 环绕视频=$RENDER_VIDEO"
python "$SCRIPT_DIR/run_inference.py"
RC=$?
if [ $RC -ne 0 ]; then echo "❌ 推理失败 exit=$RC" >&2; exit $RC; fi
echo "🎉 Done: $OUT_DIR"
