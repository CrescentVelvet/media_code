#!/usr/bin/env bash
# 02_detect_faces.sh — 阶段一：全图人脸检测（bbox + 关键点 + 置信度）。
#
# 全图直接检测，不做 mask 引导；归属交给 03_match_faces.sh 的三层判据。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [flame_human 02] 阶段一 人脸检测"
echo "  🖼️  输入: $SOURCE_DIR/images"
echo "  💾 输出: ${OUT_JSON:-$RESULTS_DIR/02_faces/face_boxes.json}"

if [ ! -d "$SOURCE_DIR/images" ]; then
    echo "❌ 图像目录不存在: $SOURCE_DIR/images"
    echo "   → 用 SOURCE_DIR= 指向 COLMAP 场景（含 images/ 与 sparse/）"
    exit 1
fi

OUT_JSON="${OUT_JSON:-$RESULTS_DIR/02_faces/face_boxes.json}" \
IMAGES_DIR="${IMAGES_DIR:-$SOURCE_DIR/images}" \
MIN_DET_CONF="${MIN_DET_CONF:-0.5}" \
MODEL_SELECTION="${MODEL_SELECTION:-1}" \
MAX_FACES="${MAX_FACES:-6}" \
    python "$SCRIPT_DIR/detect_faces.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: 人脸检测失败"
    exit 1
fi

echo "🎉 Done. 下一步：bash 03_match_faces.sh"
