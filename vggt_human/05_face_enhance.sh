#!/usr/bin/env bash
# 05_face_enhance.sh — MediaPipe 人脸检测 + HYPIR 人脸增强 + 渐变融合.
#
# 对增强 COLMAP 场景中的每张图:
#   1. MediaPipe BlazeFace 检测人脸框
#   2. 框放大 20% (FACE_PADDING) 后裁剪
#   3. HYPIR (SD2Enhancer + LoRA, 用 beauty_ppr50k 训练的 checkpoint) 增强人脸
#   4. 二次衰减渐变 mask 把增强结果无缝融合回原图 (中心=增强, 边缘=原图)
#   5. COLMAP sparse/ 原样复制 (只增强图像, 不改相机)
#
# ⚠️ 本步用 hypir conda env (有 diffusers/transformers/peft), 不用 doll.
#    脚本通过 CONDA_ENV=hypir 自动切换. 需先做过 hypir 的首次准备 (hypir/00).
#    还需 mediapipe: conda activate hypir && pip install mediapipe
#
# Prerequisites: step 02 (source/) 或 step 04 (source_aug/) 已跑.
#
# Env (all optional, defaults shown):
#   RESULTS_DIR=             # 输出根
#   SOURCE_AUG_DIR=          # 输入 (step 04 输出, 默认 source_aug; 不存在则用 source)
#   SOURCE_FACE_DIR=         # 输出 (默认 source_aug_face)
#   HYPIR_WEIGHT=            # HYPIR LoRA checkpoint
#   HYPIR_BASE_MODEL=        # SD2 base model dir
#   FACE_PADDING=0.2         # 人脸框放大比例
#   UPSCALE=1                # HYPIR upscale (1=不超分, 只增强)
#   PATCH_SIZE=512            # HYPIR patch size
#   STRIDE=256               # HYPIR stride
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use hypir conda env (has diffusers/transformers/peft for HYPIR).
export CONDA_ENV="${CONDA_ENV:-hypir}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# Input: prefer source_aug (step 04), fall back to source (step 02)
INPUT_SOURCE_DIR="${SOURCE_AUG_DIR:-$RESULTS_DIR/source_aug}"
if [ ! -d "$INPUT_SOURCE_DIR/images" ]; then
    INPUT_SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/source}"
fi
SOURCE_FACE_DIR="${SOURCE_FACE_DIR:-$RESULTS_DIR/source_aug_face}"
# If input was source (not source_aug), output to source_face
if [ "$INPUT_SOURCE_DIR" = "${SOURCE_DIR:-$RESULTS_DIR/source}" ]; then
    SOURCE_FACE_DIR="${SOURCE_FACE_DIR:-$RESULTS_DIR/source_face}"
fi

echo "🚀 [05] 人脸增强 (MediaPipe + HYPIR + 渐变融合)"
echo "  🤖 HYPIR weight: $HYPIR_WEIGHT"
echo "  🏋️ base model:   $HYPIR_BASE_MODEL"
echo "  📂 input:        $INPUT_SOURCE_DIR/images"
echo "  💾 output:       $SOURCE_FACE_DIR/images"
echo "  📐 face_padding: $FACE_PADDING  upscale=$UPSCALE"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU:          physical $CUDA_VISIBLE_DEVICES"
fi
echo ""

# Sanity checks
if [ ! -d "$INPUT_SOURCE_DIR/images" ]; then
    echo "❌ ERROR: input images not found at $INPUT_SOURCE_DIR/images" >&2
    echo "       Run step 02 or 04 first." >&2
    exit 1
fi
if [ ! -f "$HYPIR_WEIGHT" ]; then
    echo "❌ ERROR: HYPIR weight not found: $HYPIR_WEIGHT" >&2
    echo "       Check HYPIR_WEIGHT env var or run hypir/00_setup_env.sh" >&2
    exit 1
fi
if [ ! -d "$HYPIR_BASE_MODEL" ]; then
    echo "❌ ERROR: HYPIR base model not found: $HYPIR_BASE_MODEL" >&2
    echo "       Run: bash hypir/01_download_models.sh" >&2
    exit 1
fi
if ! python -c "import mediapipe" 2>/dev/null; then
    echo "❌ ERROR: mediapipe not installed in env '$CONDA_ENV'." >&2
    echo "       Run: conda activate $CONDA_ENV && pip install mediapipe" >&2
    exit 1
fi
if [ ! -d "$HYPIR_DIR" ]; then
    echo "❌ ERROR: HYPIR code not found at $HYPIR_DIR" >&2
    echo "       Run: bash hypir/00_setup_env.sh" >&2
    exit 1
fi

export HYPIR_DIR HYPIR_BASE_MODEL HYPIR_WEIGHT
export INPUT_SOURCE_DIR SOURCE_FACE_DIR FACE_PADDING UPSCALE PATCH_SIZE STRIDE DEVICE
export LORA_RANK LORA_MODULES MODEL_T COEFF_T

python "$SCRIPT_DIR/face_enhance.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi

echo ""
echo "✅ [05] Done. Face-enhanced scene: $SOURCE_FACE_DIR"
echo "  Next: GPU=0 bash $SCRIPT_DIR/06_train_denoise.sh"
