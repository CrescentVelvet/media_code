#!/usr/bin/env bash
# 02_generate_video.sh — generate a 360-degree rotation video with Wan2.2-TI2V-5B
# + your trained LoRA, using the segmented image (white background) as I2V input.
#
# Runs in the wan22_rotate conda env (same as step 01). Directly calls
# wan22/run_inference.py — not wan22/02_run_inference.sh, to avoid its set -o
# pipefail swallowing errors before Python prints the traceback.
#
# REQUIRED:
#   WEIGHT_PATH=/path/to/epoch-N.safetensors   (trained LoRA)
#   SEGMENTED_IMAGE=/path/to/image.png         (from step 01, or your own)
#
# Common overrides (all optional):
#   PROMPT="人物360度旋转展示，高质量，细节清晰。"
#   HEIGHT=1248 WIDTH=704 NUM_FRAMES=121       (portrait; use 704 1248 for landscape)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# --- inputs ---
SEGMENTED_IMAGE="${SEGMENTED_IMAGE:-$RESULTS_DIR/segmented_image.png}"
WEIGHT_PATH="${WEIGHT_PATH:-}"
PROMPT="${PROMPT:-人物360度旋转展示，高质量，细节清晰。}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走}"

# --- video params (portrait by default; person is taller than wide) ---
HEIGHT="${HEIGHT:-1248}"
WIDTH="${WIDTH:-704}"
NUM_FRAMES="${NUM_FRAMES:-121}"
SEED="${SEED:-0}"
TILED="${TILED:-1}"
FPS="${FPS:-15}"
QUALITY="${QUALITY:-5}"
LOW_VRAM="${LOW_VRAM:-0}"
VRAM_LIMIT="${VRAM_LIMIT:-}"
OUTPUT_NAME="${OUTPUT_NAME:-rotate_360}"

# --- checks ---
if [ -z "$WEIGHT_PATH" ]; then
    echo "❌ ERROR: set WEIGHT_PATH=/path/to/epoch-N.safetensors (trained LoRA)" >&2
    exit 1
fi
if [ ! -f "$WEIGHT_PATH" ]; then
    echo "❌ ERROR: LoRA weight not found: $WEIGHT_PATH" >&2; exit 1
fi
if [ ! -f "$SEGMENTED_IMAGE" ]; then
    echo "❌ ERROR: segmented image not found: $SEGMENTED_IMAGE" >&2
    echo "       Run step 01 first, or set SEGMENTED_IMAGE=/path/to/your_image.png" >&2
    exit 1
fi
if [ ! -d "$DIFFSYNTH_DIR" ]; then
    echo "❌ ERROR: DiffSynth-Studio not found at $DIFFSYNTH_DIR." >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi
if ! python -c "import diffsynth" 2>/dev/null; then
    echo "❌ ERROR: diffsynth not importable in env '$CONDA_ENV'." >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

echo "🚀 [02] Wan2.2-TI2V-5B I2V with LoRA (360 rotation)"
echo "  🤖 模型:      Wan2.2-TI2V-5B ($WAN_MODEL_DIR/Wan2.2-TI2V-5B/)"
echo "  🏋️ LoRA:      $WEIGHT_PATH  (alpha=1)"
echo "  🖼️  输入图像:  $SEGMENTED_IMAGE"
echo "  📝 prompt:    $PROMPT"
echo "  📐 分辨率:    ${WIDTH}x${HEIGHT}  frames=$NUM_FRAMES  fps=$FPS"
echo "  💾 输出:      $RESULTS_DIR/${OUTPUT_NAME}.mp4"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU:       physical $CUDA_VISIBLE_DEVICES"
else
    echo "  🎮 GPU:       default cuda:0  [set GPU=N to pin]"
fi

mkdir -p "$RESULTS_DIR"

export PROMPT NEGATIVE_PROMPT
export INPUT_IMAGE="$SEGMENTED_IMAGE"
export WEIGHT_PATH
export OUTPUT_DIR="$RESULTS_DIR"
export OUTPUT_NAME
export HEIGHT WIDTH NUM_FRAMES SEED TILED FPS QUALITY
export DEVICE WAN_MODEL_DIR

# --- call generate_video.py (uses ModelConfig(path=...) to load model directly) ---
python "$SCRIPT_DIR/generate_video.py"

if [ $? -ne 0 ]; then
    echo "❌ [02] FAILED. Video not generated." >&2
    exit 1
fi

echo "🎉 [02] Done. Video: $RESULTS_DIR/${OUTPUT_NAME}.mp4"
