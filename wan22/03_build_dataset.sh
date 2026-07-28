#!/usr/bin/env bash
# 03_build_dataset.sh — build a metadata.csv for Wan2.2-TI2V-5B LoRA training.
#
# Walks a folder of videos (mp4/mov/avi/webm/...), optionally reads per-video
# prompts from a matching .txt folder (same relative path, .txt extension), or
# uses a fixed PROMPT for all. Writes metadata.csv with columns: video,prompt.
#
# The video path in the CSV is RELATIVE to DATASET_BASE_PATH (so the training
# script can find files regardless of where it's launched).
#
# For TI2V training, the first frame of each training video is used as the image
# condition (--extra_inputs "input_image" in 04_train_lora.sh), so no separate
# image column is needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

DATA_DIR="${DATA_DIR:-}"
TXT_DIR="${TXT_DIR:-}"                 # matching prompt folder (.txt per video); empty = fixed prompt
PROMPT="${PROMPT:-}"                   # fixed prompt for all videos (when TXT_DIR is empty)
DATASET_BASE_PATH="${DATASET_BASE_PATH:-$DATA_DIR}"
METADATA_OUT="${METADATA_OUT:-$DATASET_BASE_PATH/metadata.csv}"

if [ -z "$DATA_DIR" ]; then
    echo "ERROR: set DATA_DIR=/path/to/videos" >&2
    echo "  e.g. DATA_DIR=/data_3d/w00xxxxxx/code/wan22_dataset bash $0" >&2
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR not found: $DATA_DIR" >&2; exit 1
fi

echo "=== [03] Build Wan2.2-TI2V-5B training dataset ==="
echo "  video dir:        $DATA_DIR"
echo "  prompt dir:       ${TXT_DIR:-<none — fixed prompt: '$PROMPT'>}"
echo "  dataset base:     $DATASET_BASE_PATH"
echo "  metadata output:  $METADATA_OUT"

export DATA_DIR TXT_DIR PROMPT
export DATASET_BASE_PATH METADATA_OUT

python "$SCRIPT_DIR/build_dataset.py"

echo "=== [03] Done. metadata.csv at: $METADATA_OUT ==="
echo "    Train with:"
echo "      DATASET_BASE_PATH=$DATASET_BASE_PATH bash $SCRIPT_DIR/04_train_lora.sh"
