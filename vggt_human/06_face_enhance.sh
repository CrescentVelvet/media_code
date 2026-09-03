#!/usr/bin/env bash
# 06_face_enhance.sh — 后处理: MediaPipe 人脸检测 + HYPIR 人脸增强 + 渐变融合.
#
# 对增强 COLMAP 场景中的每张图:
#   1. MediaPipe BlazeFace 检测人脸框
#   2. 框放大 20% (FACE_PADDING) 后裁剪
#   3. HYPIR (SD2Enhancer + LoRA, 用 beauty_ppr50k 训练的 checkpoint) 增强人脸
#   4. 二次衰减渐变 mask 把增强结果无缝融合回原图 (中心=增强, 边缘=原图)
#   5. COLMAP sparse/ 原样复制 (只增强图像, 不改相机)
#
# Prerequisites: INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh (建好 vggt_human env + HYPIR).
#               step 02 (03_source/) 或 step 04 (05_source_aug/) 已跑.
#
# Env (all optional, defaults shown):
#   RESULTS_DIR=             # 输出根
#   SOURCE_AUG_DIR=          # 输入 (step 05 输出, 默认 05_source_aug; 不存在则用 03_source)
#   SOURCE_FACE_DIR=         # 输出 (默认 06_source_aug_face)
#   HYPIR_WEIGHT=            # HYPIR LoRA checkpoint
#   HYPIR_BASE_MODEL=        # SD2 base model dir
#   FACE_PADDING=0.2         # 人脸框放大比例
#   UPSCALE=1                # HYPIR upscale (1=不超分, 只增强)
#   PATCH_SIZE=512            # HYPIR patch size
#   STRIDE=256               # HYPIR stride
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# Input: prefer 05_source_aug (step 05), fall back to 03_source (step 03)
INPUT_SOURCE_DIR="${SOURCE_AUG_DIR:-$RESULTS_DIR/05_source_aug}"
if [ ! -d "$INPUT_SOURCE_DIR/images" ]; then
    INPUT_SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/03_source}"
fi
SOURCE_FACE_DIR="${SOURCE_FACE_DIR:-$RESULTS_DIR/06_source_aug_face}"
# If input was 03_source (not 05_source_aug), output to 06b_source_face
if [ "$INPUT_SOURCE_DIR" = "${SOURCE_DIR:-$RESULTS_DIR/03_source}" ]; then
    SOURCE_FACE_DIR="${SOURCE_FACE_DIR:-$RESULTS_DIR/06b_source_face}"
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
    echo "       Check HYPIR_WEIGHT env var" >&2
    exit 1
fi
if [ ! -d "$HYPIR_BASE_MODEL" ]; then
    echo "❌ ERROR: HYPIR base model not found: $HYPIR_BASE_MODEL" >&2
    echo "       Run: INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh" >&2
    exit 1
fi
if ! python -c "import mediapipe" 2>/dev/null; then
    echo "❌ ERROR: mediapipe not installed in env '$CONDA_ENV'." >&2
    echo "       Run: INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh" >&2
    exit 1
fi
if [ ! -d "$HYPIR_DIR" ]; then
    echo "❌ ERROR: HYPIR code not found at $HYPIR_DIR" >&2
    echo "       Run: INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh" >&2
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
echo "  Next: GPU=0 bash $SCRIPT_DIR/07_train_denoise.sh"
