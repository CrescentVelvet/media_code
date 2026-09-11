#!/usr/bin/env bash
# 01d_enhance_faces.sh — HYPIR 增强 p0 人脸区域（提升 head 监督图上限）
#
# 前置：02（face_boxes.json）、03（face_person_match.json）。
# 产出：$OUT_DIR/images/*  —— 08 用 IMAGES_DIR 指向它即可（相机仍从 SOURCE_DIR 读）
# 冒烟：LIMIT=3 DEBUG_DIR=... bash 01d_enhance_faces.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$SCRIPT_DIR" source "$SCRIPT_DIR/_env.sh"

# HYPIR 依赖（diffusers/peft/accelerate）装在 vggt_human 环境，flame_human 没有
# → 本步切过去跑（其余步骤仍用 flame_human）
ENHANCE_ENV="${ENHANCE_ENV:-vggt_human}"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
if conda activate "$ENHANCE_ENV" 2>/dev/null; then
    echo "  🔁 已切换到 $ENHANCE_ENV 环境（HYPIR 依赖）"
else
    echo "  ⚠️ 无法激活 $ENHANCE_ENV，继续用当前环境" >&2
fi

MATCH_JSON="${MATCH_JSON:-$RESULTS_DIR/03_match/face_person_match.json}" \
SRC_IMAGES_DIR="${SRC_IMAGES_DIR:-$SOURCE_DIR/images}" \
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/01d_enhanced_faces}" \
HYPIR_DIR="${HYPIR_DIR:-$HYPIR_DIR}" \
HYPIR_BASE_MODEL="${HYPIR_BASE_MODEL:-$HYPIR_BASE_MODEL}" \
HYPIR_WEIGHT="${HYPIR_WEIGHT:-$HYPIR_WEIGHT}" \
FACE_PADDING="${FACE_PADDING:-0.25}" \
UPSCALE="${UPSCALE:-2}" \
PATCH_SIZE="${PATCH_SIZE:-512}" \
STRIDE="${STRIDE:-256}" \
LIMIT="${LIMIT:-0}" \
DEBUG_DIR="${DEBUG_DIR:-}" \
DEVICE="${DEVICE:-cuda}" \
    python "$SCRIPT_DIR/enhance_faces.py"
if [ $? -ne 0 ]; then
    echo "❌ [01d] HYPIR 人脸增强失败" >&2
    exit 1
fi
echo "🎉 [01d] Done. 下一步：IMAGES_DIR=\$OUT_DIR/images bash 08_train.sh"
