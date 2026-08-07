#!/usr/bin/env bash
# 02_generate_video.sh — generate a 360-degree rotation video with Wan2.2-TI2V-5B
# + your trained LoRA, using the segmented image (white background) as I2V input.
#
# Runs in the wan22_rotate conda env (same as step 01). Calls the existing
# wan22/02_run_inference.sh which loads the DiffSynth-Studio pipeline, applies
# the LoRA, and generates an mp4.
#
# REQUIRED:
#   WEIGHT_PATH=/path/to/epoch-N.safetensors   (trained LoRA)
#   SEGMENTED_IMAGE=/path/to/image.png         (from step 01, or your own)
#
# Common overrides (all optional):
#   PROMPT="人物360度旋转展示，高质量，细节清晰。"
#   HEIGHT=1248 WIDTH=704 NUM_FRAMES=121       (portrait; use 704 1248 for landscape)
set -euo pipefail
trap 'echo "ERROR: $SCRIPT_DIR/$(basename "$0") line $LINENO: $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# --- inputs ---
SEGMENTED_IMAGE="${SEGMENTED_IMAGE:-$RESULTS_DIR/segmented_image.png}"
WEIGHT_PATH="${WEIGHT_PATH:-}"
PROMPT="${PROMPT:-人物360度旋转展示，高质量，细节清晰。}"

# --- video params (portrait by default; person is taller than wide) ---
HEIGHT="${HEIGHT:-1248}"
WIDTH="${WIDTH:-704}"
NUM_FRAMES="${NUM_FRAMES:-121}"
FPS="${FPS:-15}"
OUTPUT_NAME="${OUTPUT_NAME:-rotate_360}"

# --- checks ---
if [ -z "$WEIGHT_PATH" ]; then
    echo "ERROR: set WEIGHT_PATH=/path/to/epoch-N.safetensors (trained LoRA)" >&2
    echo "  e.g. WEIGHT_PATH=../wan22_experiments/exp1/epoch-4.safetensors bash $0" >&2
    exit 1
fi
if [ ! -f "$WEIGHT_PATH" ]; then
    echo "ERROR: LoRA weight not found: $WEIGHT_PATH" >&2; exit 1
fi
if [ ! -f "$SEGMENTED_IMAGE" ]; then
    echo "ERROR: segmented image not found: $SEGMENTED_IMAGE" >&2
    echo "       Run step 01 first: INPUT_DIR=/path/to/subject bash $SCRIPT_DIR/01_pick_and_segment.sh" >&2
    echo "       Or set SEGMENTED_IMAGE=/path/to/your_image.png" >&2
    exit 1
fi
if [ ! -d "$DIFFSYNTH_DIR" ]; then
    echo "ERROR: DiffSynth-Studio not found at $DIFFSYNTH_DIR." >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

echo "=== [02] Wan2.2-TI2V-5B I2V with LoRA (360 rotation) ==="
echo "  LoRA:      $WEIGHT_PATH  (alpha=1)"
echo "  输入图像:  $SEGMENTED_IMAGE"
echo "  prompt:    $PROMPT"
echo "  分辨率:    ${WIDTH}x${HEIGHT}  frames=$NUM_FRAMES  fps=$FPS"
echo "  输出:      $RESULTS_DIR/${OUTPUT_NAME}.mp4"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES"
else
    echo "  GPU:       default cuda:0  [set GPU=N to pin]"
fi

# --- delegate to wan22's inference script ---
# wan22/_env.sh sees CONDA_ENV (exported by our _env.sh) and re-activates the
# same env (no-op). We just pass through the inference params.
export PROMPT
export INPUT_IMAGE="$SEGMENTED_IMAGE"
export WEIGHT_PATH
export OUTPUT_DIR="$RESULTS_DIR"
export OUTPUT_NAME
export HEIGHT WIDTH NUM_FRAMES FPS

bash "$REPO_DIR/wan22/02_run_inference.sh"

echo "=== [02] Done. Video: $RESULTS_DIR/${OUTPUT_NAME}.mp4 ==="
