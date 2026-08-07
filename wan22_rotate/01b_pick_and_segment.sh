#!/usr/bin/env bash
# 01b_pick_and_segment.sh — simplified: detect person -> SAM segment -> pick
# largest person area -> white background. No 3D body model, no global_rot.
#
# REQUIRED:  INPUT_DIR=/path/to/subject_folder  (must contain image/ subfolder)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# --- sam_3d_body detector + segmentor (no 3D body model) ---
export DETECTOR_NAME="${DETECTOR_NAME:-vitdet}"
export DETECTOR_PATH="${DETECTOR_PATH:-}"
export SEGMENTOR_NAME="${SEGMENTOR_NAME:-sam2}"
export SEGMENTOR_PATH="${SEGMENTOR_PATH:-}"
export DEVICE="${DEVICE:-cuda}"
export SAM3D_DIR

export BBOX_THRESH="${BBOX_THRESH:-0.8}"
export WHITE_BG="${WHITE_BG:-1}"
export PADDING="${PADDING:-0.1}"

# --- I/O ---
export INPUT_DIR="${INPUT_DIR:-}"
if [ -z "$INPUT_DIR" ]; then
    echo "ERROR: set INPUT_DIR=/path/to/subject_folder (must contain image/ subfolder)" >&2
    echo "  e.g. INPUT_DIR=/data/subject_001 bash $0" >&2
    exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: INPUT_DIR not found: $INPUT_DIR" >&2; exit 1
fi
if [ ! -d "$INPUT_DIR/image" ]; then
    echo "WARNING: no 'image' subfolder in $INPUT_DIR; will use $INPUT_DIR itself" >&2
fi

export OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR}"

# --- checks ---
if [ ! -d "$SAM3D_DIR" ]; then
    echo "ERROR: sam-3d-body code not found at $SAM3D_DIR." >&2
    exit 1
fi

echo "=== [01b] Pick largest person + segment (simplified) ==="
echo "  代码:      $SAM3D_DIR"
echo "  输入:      $INPUT_DIR/image"
echo "  输出:      $OUTPUT_DIR/segmented_image.png"
echo "  detector:  $DETECTOR_NAME"
echo "  segmentor: ${SEGMENTOR_NAME:-none} ${SEGMENTOR_PATH:+($SEGMENTOR_PATH)}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES"
else
    echo "  GPU:       default cuda:0  [set GPU=N to pin]"
fi

cd "$SAM3D_DIR"
python "$SCRIPT_DIR/pick_and_segment_simple.py"

echo "=== [01b] Done. Segmented image: $OUTPUT_DIR/segmented_image.png ==="
