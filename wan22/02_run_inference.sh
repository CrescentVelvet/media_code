#!/usr/bin/env bash
# 02_run_inference.sh — Wan2.2-TI2V-5B video generation (T2V or I2V).
# Calls run_inference.py: loads the pipeline once, generates one video,
# and prints model-load + generation timing + the output path.
#
# Modes:
#   T2V (text-to-video):  set PROMPT="..."  (default)
#   I2V (image-to-video): set PROMPT="..." INPUT_IMAGE=/path/to/img.png
#
# Load a trained LoRA with WEIGHT_PATH=/path/to/epoch-N.safetensors.
# Low VRAM (disk offload, ~X GB) with LOW_VRAM=1.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# --- generation params (all overridable via env) ---
PROMPT="${PROMPT:-两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走}"
INPUT_IMAGE="${INPUT_IMAGE:-}"          # set to a path for I2V; empty = T2V
WEIGHT_PATH="${WEIGHT_PATH:-}"          # set to a trained LoRA .safetensors; empty = native model
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/../wan22_results}"
OUTPUT_NAME="${OUTPUT_NAME:-video}"

# video shape
HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1248}"
NUM_FRAMES="${NUM_FRAMES:-121}"
SEED="${SEED:-0}"
TILED="${TILED:-1}"                     # 1 = tiled VAE (avoids OOM on big frames)
FPS="${FPS:-15}"
QUALITY="${QUALITY:-5}"

# low-VRAM disk offload (much slower but fits small GPUs)
LOW_VRAM="${LOW_VRAM:-0}"

# VRAM limit (GB) for low-VRAM mode; default = free VRAM - 2 GB
VRAM_LIMIT="${VRAM_LIMIT:-}"

echo "=== [02] Wan2.2-TI2V-5B inference ==="
echo "  代码路径:   $DIFFSYNTH_DIR"
echo "  模型路径:   $MODEL_DIR  (DIFFSYNTH_SKIP_DOWNLOAD=$DIFFSYNTH_SKIP_DOWNLOAD)"
if [ -n "$WEIGHT_PATH" ]; then
    echo "  LoRA权重:   $WEIGHT_PATH  (alpha=1)"
else
    echo "  LoRA权重:   <none — native model>"
fi
if [ -n "$INPUT_IMAGE" ]; then
    echo "  模式:       I2V (image-to-video)"
    echo "  输入图像:   $INPUT_IMAGE"
else
    echo "  模式:       T2V (text-to-video)"
fi
echo "  prompt:     $PROMPT"
echo "  分辨率:     ${WIDTH}x${HEIGHT}  frames=$NUM_FRAMES  fps=$FPS"
echo "  seed:       $SEED  tiled=$TILED  low_vram=$LOW_VRAM"
echo "  输出:       $OUTPUT_DIR/${OUTPUT_NAME}.mp4"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:        physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change]"
else
    echo "  GPU:        default cuda:0 (= first visible)  [set GPU=N to pin a card]"
fi

# --- checks ---
if [ ! -d "$DIFFSYNTH_DIR" ]; then
    echo "ERROR: DiffSynth-Studio code dir not found at $DIFFSYNTH_DIR. Run run_all.sh or 00_setup_env.sh first." >&2; exit 1
fi
if ! python -c "import diffsynth" 2>/dev/null; then
    echo "ERROR: diffsynth not importable in env '$CONDA_ENV'. Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2; exit 1
fi
if [ -n "$INPUT_IMAGE" ] && [ ! -f "$INPUT_IMAGE" ]; then
    echo "ERROR: input image not found: $INPUT_IMAGE" >&2; exit 1
fi
if [ -n "$WEIGHT_PATH" ] && [ ! -f "$WEIGHT_PATH" ]; then
    echo "ERROR: LoRA weight not found: $WEIGHT_PATH" >&2; exit 1
fi

mkdir -p "$OUTPUT_DIR"

export DIFFSYNTH_DIR
export PROMPT NEGATIVE_PROMPT INPUT_IMAGE WEIGHT_PATH
export OUTPUT_DIR OUTPUT_NAME
export HEIGHT WIDTH NUM_FRAMES SEED TILED FPS QUALITY
export LOW_VRAM VRAM_LIMIT

python "$SCRIPT_DIR/run_inference.py"

echo "=== [02] Done. Video at: $OUTPUT_DIR/${OUTPUT_NAME}.mp4 ==="
