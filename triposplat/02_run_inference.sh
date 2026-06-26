#!/usr/bin/env bash
# 02_run_inference.sh — batch TripoSplat inference over a folder of images.
# Loads the pipeline once (in run_batch.py) and loops; only NUM_GAUSSIANS density.
# Outputs are named after each input image (<stem>.ply / <stem>.splat).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

TRIPOSPLAT_DIR="${TRIPOSPLAT_DIR:-$REPO_DIR/../TripoSplat}"
INPUT_DIR="${INPUT_DIR:-$TRIPOSPLAT_DIR/static/example_inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-$TRIPOSPLAT_DIR/output}"
NUM_GAUSSIANS="${NUM_GAUSSIANS:-262144}"

echo "=== [02] Batch TripoSplat inference ==="
echo "  code dir:  $TRIPOSPLAT_DIR"
echo "  input:     $INPUT_DIR"
echo "  output:    $OUTPUT_DIR"
echo "  gaussians: $NUM_GAUSSIANS  (only this density)"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change]"
else
    echo "  GPU:       default cuda:0 (= first visible)  [set GPU=N to pin a card]"
fi

if [ ! -d "$TRIPOSPLAT_DIR" ]; then
    echo "ERROR: TripoSplat code dir not found at $TRIPOSPLAT_DIR." >&2
    exit 1
fi
if [ ! -e "$TRIPOSPLAT_DIR/ckpts" ]; then
    echo "ERROR: $TRIPOSPLAT_DIR/ckpts missing. Run 01_download_models.sh first." >&2
    exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: input dir not found: $INPUT_DIR" >&2
    exit 1
fi

export TRIPOSPLAT_DIR INPUT_DIR OUTPUT_DIR NUM_GAUSSIANS
python "$SCRIPT_DIR/run_batch.py"

echo "=== [02] Done. Outputs in: $OUTPUT_DIR ==="
