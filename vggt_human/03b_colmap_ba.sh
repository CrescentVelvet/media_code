#!/usr/bin/env bash
# 03b_colmap_ba.sh — COLMAP Bundle Adjustment with VGGT poses (fixed intrinsics).
#
# VGGT reconstruction (03_source/sparse/0) has cameras (PINHOLE, VGGT intrinsics)
# + images (VGGT poses) + points3D (VGGT cloud), BUT no 2D-3D observations
# (num_points3D=0 per image). BA cannot optimize without observations.
#
# This script uses pycolmap.triangulate_points:
#   - Takes VGGT reconstruction (known poses) as input
#   - Runs SIFT feature extraction + exhaustive matching on images
#   - Triangulates 3D points from matched features using VGGT poses
#   - Creates 2D-3D observations (POINTS2D in images.txt)
#   - Runs BA to refine poses + 3D points (intrinsics FIXED)
#
# Result: VGGT poses refined by real feature observations, intrinsics unchanged.
#
# Env (all optional, defaults shown):
#   RESULTS_DIR=             # output root
#   SOURCE_DIR=              # VGGT COLMAP scene (default: $RESULTS_DIR/03_source)
#   IMAGES_DIR=              # original images (default: $SOURCE_DIR/images)
#   OUTPUT_DIR=              # refined output (default: $RESULTS_DIR/03b_source_ba)
#   BA_VERBOSE=0             # 1 = print progress
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

RESULTS_DIR="${RESULTS_DIR:-$RESULTS_ROOT}"
SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/03_source}"
IMAGES_DIR="${IMAGES_DIR:-$SOURCE_DIR/images}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/03b_source_ba}"
BA_VERBOSE="${BA_VERBOSE:-0}"

echo "🔧 [03b] COLMAP BA (VGGT poses → triangulate → refine, intrinsics FIXED)"
echo "  📂 source:   $SOURCE_DIR/sparse/0"
echo "  📂 images:   $IMAGES_DIR"
echo "  💾 output:   $OUTPUT_DIR"
echo "  🔒 intrinsics: FIXED (VGGT-predicted, no refinement)"
echo "  🎯 refine:   extrinsics (poses) + 3D points only"
echo ""

if [ ! -d "$SOURCE_DIR/sparse/0" ]; then
    echo "❌ ERROR: VGGT COLMAP scene not found: $SOURCE_DIR/sparse/0" >&2
    echo "       Run step 03 first: bash $SCRIPT_DIR/03_npz_to_colmap.sh" >&2
    exit 1
fi
if [ ! -d "$IMAGES_DIR" ]; then
    echo "❌ ERROR: images dir not found: $IMAGES_DIR" >&2
    exit 1
fi

export SOURCE_DIR IMAGES_DIR OUTPUT_DIR BA_VERBOSE

python "$SCRIPT_DIR/colmap_ba.py"

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo "❌ [03b] failed (exit=$EXIT)" >&2
    exit $EXIT
fi

echo ""
echo "✅ [03b] Done. BA-refined COLMAP model (VGGT poses refined):"
echo "  $OUTPUT_DIR/sparse/0_text/cameras.txt"
echo "  $OUTPUT_DIR/sparse/0_text/images.txt"
echo "  $OUTPUT_DIR/sparse/0_text/points3D.txt"
echo ""
echo "  → Use with 04_train_3dgs.sh: SOURCE_DIR=$RESULTS_DIR/03b_source_ba ..."
