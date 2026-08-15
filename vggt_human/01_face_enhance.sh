#!/usr/bin/env bash
# 01_face_enhance.sh — 前处理: 对原始输入图像做人脸增强 (MediaPipe + HYPIR + 渐变融合).
#
# 与 06_face_enhance.sh (后处理) 调用同一个 face_enhance.py, 但:
#   - 06 对增强 COLMAP 场景中的图做后处理 (source_aug/images/ → source_aug_face/)
#   - 01 对原始输入图做前处理 (INPUT_DIR → input_face/images/)
#
# 前处理后的图像作为 step 02 (VGGT-Omega 推理) 的输入.
# face_enhance.py 自动适配: COLMAP场景(images/子夹) / test_task结构(image/子夹) / 散图夹.
#
# Prerequisites: INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh (建好 vggt_human env + HYPIR).
#
# Env (all optional, defaults shown):
#   INPUT_DIR=              # 原始图像文件夹 (散图 / image/子夹 / images/子夹)
#   RESULTS_DIR=            # 输出根
#   HYPIR_WEIGHT=           # HYPIR LoRA checkpoint
#   FACE_PADDING=0.2        # 人脸框放大比例
#   UPSCALE=1               # HYPIR upscale (1=不超分)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

INPUT_DIR="${INPUT_DIR:-$VGGT_DIR/examples}"
INPUT_FACE_DIR="${INPUT_FACE_DIR:-$RESULTS_DIR/input_face}"
FACE_PADDING="${FACE_PADDING:-0.2}"
UPSCALE="${UPSCALE:-1}"
DEVICE="${DEVICE:-cuda}"

echo "🚀 [01] 前处理人脸增强 (MediaPipe + HYPIR + 渐变融合)"
echo "  🤖 HYPIR weight: $HYPIR_WEIGHT"
echo "  🏋️ base model:   $HYPIR_BASE_MODEL"
echo "  📂 input:        $INPUT_DIR"
echo "  💾 output:       $INPUT_FACE_DIR/images"
echo "  📐 face_padding: $FACE_PADDING  upscale=$UPSCALE"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU:          physical $CUDA_VISIBLE_DEVICES"
fi
echo ""

# Sanity checks
if [ ! -e "$INPUT_DIR" ]; then
    echo "❌ ERROR: input not found: $INPUT_DIR" >&2
    exit 1
fi
if [ ! -f "$HYPIR_WEIGHT" ]; then
    echo "❌ ERROR: HYPIR weight not found: $HYPIR_WEIGHT" >&2
    echo "       Check HYPIR_WEIGHT env var or run INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh" >&2
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

# face_enhance.py reads INPUT_SOURCE_DIR (auto-detects images/ image/ or plain folder)
# and writes to SOURCE_FACE_DIR/images/
export HYPIR_DIR HYPIR_BASE_MODEL HYPIR_WEIGHT
export INPUT_SOURCE_DIR="$INPUT_DIR" SOURCE_FACE_DIR="$INPUT_FACE_DIR"
export FACE_PADDING UPSCALE PATCH_SIZE STRIDE DEVICE
export LORA_RANK LORA_MODULES MODEL_T COEFF_T

python "$SCRIPT_DIR/face_enhance.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi

echo ""
echo "✅ [01] Done. Face-enhanced images: $INPUT_FACE_DIR/images"
echo "  Next: GPU=0 INPUT_DIR=$INPUT_FACE_DIR bash $SCRIPT_DIR/02_run_inference.sh"
