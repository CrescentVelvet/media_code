#!/usr/bin/env bash
# 03e_detect_landmarks.sh — 多视角人脸 landmark 检测（MediaPipe FaceMesh 468）
#
# 头/身/景拆分 step 2：给 3DMM 拟合提供 2D 观测。
# 用 SAM3 face mask 定位人脸框 → 裁剪 → 在裁剪区里跑 FaceMesh，
# 多张脸时用 mask 质心挑最近的，landmark 天然带 pid 归属（不像整图检测会串人）。
#
# 输入: $RESULTS_DIR/03_source/images
#       $RESULTS_DIR/03_sam3_face_masks/<stem>.p<pid>.mask.png
# 输出: $RESULTS_DIR/03e_head_3dmm/face_landmarks.json
#
# Env (all optional, defaults shown):
#   RESULTS_DIR=      # 输出根（默认 $HOME/output/vggt_human_results）
#   IMAGES_DIR=       # 输入图像（默认 $RESULTS_DIR/03_source/images）
#   MASKS_DIR=        # SAM3 face mask（默认 $RESULTS_DIR/03_sam3_face_masks）
#   OUT_JSON=         # 输出 json（默认 $RESULTS_DIR/03e_head_3dmm/face_landmarks.json）
#   PAD_RATIO=0.6     # crop 外扩比例
#   MIN_MASK_PX=400   # mask 像素数下限
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

RESULTS_DIR="${RESULTS_DIR:-$HOME/output/vggt_human_results}"
IMAGES_DIR="${IMAGES_DIR:-$RESULTS_DIR/03_source/images}"
MASKS_DIR="${MASKS_DIR:-$RESULTS_DIR/03_sam3_face_masks}"
OUT_JSON="${OUT_JSON:-$RESULTS_DIR/03e_head_3dmm/face_landmarks.json}"

echo "🎯 [03e] 人脸 landmark 检测 (MediaPipe FaceMesh 468)"
echo "  📂 images: $IMAGES_DIR"
echo "  📂 masks:  $MASKS_DIR"
echo "  💾 out:    $OUT_JSON"
echo ""

if [ ! -d "$IMAGES_DIR" ]; then
    echo "❌ 图像目录不存在: $IMAGES_DIR" >&2
    exit 1
fi
if [ ! -d "$MASKS_DIR" ]; then
    echo "❌ mask 目录不存在: $MASKS_DIR" >&2
    echo "   先跑 SAM3 人脸分割 (03_sam3_face_masks)" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT_JSON")"

RESULTS_DIR="$RESULTS_DIR" \
IMAGES_DIR="$IMAGES_DIR" \
MASKS_DIR="$MASKS_DIR" \
OUT_JSON="$OUT_JSON" \
PAD_RATIO="${PAD_RATIO:-0.6}" \
MIN_MASK_PX="${MIN_MASK_PX:-400}" \
python "$SCRIPT_DIR/detect_face_landmarks.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi

echo ""
echo "✅ [03e] landmark 检测完成: $OUT_JSON"
