#!/usr/bin/env bash
# 02b_run_batch.sh — batch inference over PREP_DIR (objects with views.json).
# Shape (mv / single per object) for all objects, then PBR paint for all.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

export SHAPE_DIR="${SHAPE_DIR:-$MODEL_DIR/Hunyuan3D-2.1}"
export MV_DIR="${MV_DIR:-$MODEL_DIR/Hunyuan3D-2mv}"
export DINO_DIR="${DINO_DIR:-$MODEL_DIR/dinov2-giant}"
export PREP_DIR="${PREP_DIR:-$RESULTS_DIR/preprocessed}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/models}"

for f in "$SHAPE_DIR" "$MV_DIR" "$PREP_DIR"; do
    [ -d "$f" ] || { echo "❌ missing: $f (run 00/01/01b first)" >&2; exit 1; }
done

echo "🚀 [02b] batch inference"
echo "  📁 prep:   $PREP_DIR"
echo "  💾 output: $OUTPUT_DIR"
echo "  🎮 CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all>}"

python "$SCRIPT_DIR/run_batch.py"
rc=$?
[ $rc -ne 0 ] && { echo "❌ FAILED (rc=$rc)" >&2; exit $rc; }
echo "🎉 [02b] Done."
