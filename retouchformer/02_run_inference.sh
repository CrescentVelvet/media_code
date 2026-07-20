#!/usr/bin/env bash
# 02_run_inference.sh — batch RetouchFormer face retouching over a folder of
# images. Calls run_inference.py (a drop-in for the official img_retouching.py
# that loads the model ONCE, loops, preserves relative paths, and prints
# per-image timing + a summary).
#
# For each image in INPUT_DIR (walked recursively), produces:
#   $OUTPUT_DIR/result/<same-relative-path>.png   (retouched, 512x512)
#
# Differences from the official img_retouching.py:
#   - no hardcoded CUDA_VISIBLE_DEVICES="0" — use GPU=N to pick a card;
#   - output preserves the input's relative directory structure (flat
#     <name>_out.png dump in the official);
#   - RESIZE_MODE=square (default) center-crops to 512x512 so non-square wild
#     images don't crash the model (VRT hardcodes 512x512). On the FFHQR test
#     set (already 512x512) it is identical to the official. Set
#     RESIZE_MODE=smallest to reproduce wildDataset exactly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

RETOUCH_DIR="${RETOUCH_DIR:-$REPO_DIR/../RetouchFormer}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/RetouchFormer}"

# Checkpoint: official img_retouching.py defaults —ckpt release_model --epoch best
CKPT_DIR_NAME="${CKPT_DIR_NAME:-release_model}"
EPOCH="${EPOCH:-best}"
WEIGHT_FILE="${WEIGHT_FILE:-gen_${EPOCH}.pth}"
CKPT_DIR="${CKPT_DIR:-$MODEL_DIR/$CKPT_DIR_NAME}"
WEIGHT_PATH="${WEIGHT_PATH:-$CKPT_DIR/$WEIGHT_FILE}"

INPUT_DIR="${INPUT_DIR:-$RETOUCH_DIR/datasets/test}"
INPUT_NAME="$(basename "$INPUT_DIR")"
OUTPUT_DIR="${OUTPUT_DIR:-$RETOUCH_DIR/results/$INPUT_NAME}"

# --- inference params (all overridable via env) ---
MODEL_NAME="${MODEL_NAME:-RetouchFormer}"     # model.<NAME> -> InpaintGenerator()
RESIZE_MODE="${RESIZE_MODE:-square}"          # square | smallest
SIZE="${SIZE:-512}"                           # model is fixed to 512
DEVICE="${DEVICE:-cuda}"

echo "=== [02] RetouchFormer batch inference ==="
echo "  代码路径:      $RETOUCH_DIR"
echo "  模型:          model.$MODEL_NAME.InpaintGenerator"
echo "  权重:          $WEIGHT_PATH"
echo "  输入图像:      $INPUT_DIR"
echo "  输出目录:       $OUTPUT_DIR/result"
echo "  参数:          resize=$RESIZE_MODE size=$SIZE device=$DEVICE"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:           physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change]"
else
    echo "  GPU:           default cuda:0 (= first visible)  [set GPU=N to pin a card]"
fi

# --- checks ---
if [ ! -d "$RETOUCH_DIR" ]; then
    echo "ERROR: RetouchFormer code dir not found at $RETOUCH_DIR. Run run_all.sh first." >&2; exit 1
fi
if [ ! -f "$WEIGHT_PATH" ]; then
    echo "ERROR: checkpoint not found at $WEIGHT_PATH." >&2
    echo "       Run bash $SCRIPT_DIR/01_download_models.sh first (Baidu manual step), or set WEIGHT_PATH." >&2
    exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: input dir not found: $INPUT_DIR" >&2; exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Forward params to run_inference.py (reads env).
export RETOUCH_DIR
export WEIGHT_PATH MODEL_NAME INPUT_DIR OUTPUT_DIR RESIZE_MODE SIZE DEVICE

python "$SCRIPT_DIR/run_inference.py"

echo "=== [02] Done. Results in: $OUTPUT_DIR/result ==="
