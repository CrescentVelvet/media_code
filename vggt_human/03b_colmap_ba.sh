#!/usr/bin/env bash
# 03b_colmap_ba.sh — COLMAP Bundle Adjustment on step 03's output.
#
# Fixes intrinsics (fx, fy, cx, cy), only refines extrinsics (poses) + 3D points.
# Safe alternative to pose_refine (no rasterizer gradient needed).
#
# Env (all optional, defaults shown):
#   RESULTS_DIR=             # output root
#   SOURCE_DIR=              # input COLMAP sparse (default: $RESULTS_DIR/source)
#   OUTPUT_DIR=              # refined output (default: $RESULTS_DIR/source_ba)
#   BA_VERBOSE=0             # 1 = print Ceres progress
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

RESULTS_DIR="${RESULTS_DIR:-$RESULTS_ROOT}"
SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/source}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/source_ba}"
BA_VERBOSE="${BA_VERBOSE:-0}"

echo "🔧 [03b] COLMAP Bundle Adjustment (intrinsics FIXED)"
echo "  📂 input:  $SOURCE_DIR/sparse/0"
echo "  💾 output: $OUTPUT_DIR/sparse/0"
echo "  🔒 intrinsics: fx, fy, cx, cy FIXED"
echo "  🎯 refine: extrinsics (poses) + 3D points only"
echo ""

if [ ! -f "$SOURCE_DIR/sparse/0/cameras.txt" ]; then
    echo "❌ ERROR: $SOURCE_DIR/sparse/0/cameras.txt not found" >&2
    echo "   Run step 03 (npz_to_colmap) first." >&2
    exit 1
fi

export SOURCE_DIR="$SOURCE_DIR/sparse/0"
export OUTPUT_DIR="$OUTPUT_DIR/sparse/0"
export BA_VERBOSE

python "$SCRIPT_DIR/colmap_ba.py"

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo "❌ [03b] BA failed (exit=$EXIT)" >&2
    exit $EXIT
fi

echo ""
echo "✅ [03b] Done. Refined COLMAP model:"
echo "  $OUTPUT_DIR/cameras.txt"
echo "  $OUTPUT_DIR/images.txt"
echo "  $OUTPUT_DIR/points3D.txt"
echo ""
echo "  → Use with 04_train_3dgs.sh: SOURCE_DIR=$RESULTS_DIR/source_ba ..."
