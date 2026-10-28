#!/usr/bin/env bash
# 01b_preprocess_views.sh — prepare multi-view inputs for Hunyuan3D-2mv:
#   1. rembg every image (RGBA PNG; the shape pipeline only auto-masks alpha)
#   2. VGGT-Omega pose estimation over the ORIGINAL images (batch mode)
#   3. snap camera azimuths to canonical slots -> views.json per object
#
# INPUT_ROOT layout: <INPUT_ROOT>/<object>/<frame>.jpg (3 views per object).
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

INPUT_ROOT="${INPUT_ROOT:?❌ set INPUT_ROOT=<folder of object folders>}"
PREP_DIR="${PREP_DIR:-$RESULTS_DIR/preprocessed}"
VGGT_OUT="${VGGT_OUT:-$RESULTS_DIR/vggt}"
VGGT_CONDA_ENV="${VGGT_CONDA_ENV:-vggt_human}"
export PREP_DIR VGGT_OUT

echo "🚀 [01b] preprocess $INPUT_ROOT"
echo "  📁 rembg out:  $PREP_DIR"
echo "  📁 vggt out:   $VGGT_OUT"

# --- 1. rembg (hunyuan3d env) ---
echo "✂️ [01b] step 1/3: background removal"
python "$SCRIPT_DIR/preprocess_views.py" || { echo "❌ rembg failed" >&2; exit 1; }

# --- 2. VGGT-Omega poses (vggt env; reuses the sibling project's runner) ---
VGGT_RUNNER="$REPO_DIR/vggt-omega/02_run_inference.sh"
if [ ! -f "$VGGT_RUNNER" ]; then
    echo "❌ $VGGT_RUNNER not found" >&2; exit 1
fi
if [ ! -d "$VGGT_OUT" ] || [ -z "$(ls -A "$VGGT_OUT" 2>/dev/null)" ]; then
    echo "📐 [01b] step 2/3: VGGT-Omega pose estimation (env=$VGGT_CONDA_ENV)"
    CONDA_ENV="$VGGT_CONDA_ENV" \
    INPUT_DIR="$INPUT_ROOT" \
    OUTPUT_DIR="$VGGT_OUT" \
    MODEL_DIR="${VGGT_HUMAN_MODEL_DIR:-$REPO_DIR/../../model/VGGT-Omega}" \
        bash "$VGGT_RUNNER" || { echo "❌ VGGT failed" >&2; exit 1; }
else
    echo "⏭️ [01b] step 2/3: $VGGT_OUT exists, reuse (delete to redo)"
fi

# --- 3. azimuth -> canonical view slots ---
echo "🎯 [01b] step 3/3: assign view tags"
python "$SCRIPT_DIR/assign_views.py" || { echo "❌ assign_views failed" >&2; exit 1; }

echo "🎉 [01b] Done. views.json under $PREP_DIR/<object>/"
