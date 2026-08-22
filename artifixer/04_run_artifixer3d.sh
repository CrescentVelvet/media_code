#!/usr/bin/env bash
# 04_run_artifixer3d.sh — ArtiFixer3D: 3DGRUT distillation from ArtiFixer predicted frames.
#
# Trains a fresh 3DGRUT optimization on the union of real anchor views and
# ArtiFixer-generated target views. Then renders the updated reconstruction and
# prepares ArtiFixer3D+ inference metadata (split_artifixer3d_plus.json).
#
# This is the second stage of the full ArtiFixer pipeline:
#   ArtiFixer (03) → ArtiFixer3D (04) → ArtiFixer3D+ (05)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# --- params ---
SCENE_ROOT="${SCENE_ROOT:-$RESULTS_DIR/prep/my_scene}"
ARTIFIXER_SAVE_DIR="${ARTIFIXER_SAVE_DIR:-$RESULTS_DIR/corrected}"
SCENE_ID="${SCENE_ID:-$(basename "$SCENE_ROOT")}"

# Find ArtiFixer prediction frames dir
CKPT_STEM="${CKPT_STEM:-$(basename "$ARTIFIXER_CHECKPOINT" .pt)}"
# Output dir pattern from 03: $SAVE_DIR/<ckpt_stem>/<mode_name>
# mode_name = distilled_views_reconstructed_colmap_<views>_<selection>_sink<N>_<trajectory>
# Use find to locate the pred directory
ARTIFIXER_FRAMES_DIR="${ARTIFIXER_FRAMES_DIR:-}"

# 3DGRUT distillation params
ARTIFIXER3D_STEPS="${ARTIFIXER3D_STEPS:-30000}"
PHASES="${PHASES:-distill,render,prepare_artifixer3d_plus}"
USE_WANDB="${USE_WANDB:-0}"

echo "🚀 [04] ArtiFixer3D distillation"
echo "  📁 scene root:       $SCENE_ROOT"
echo "  📁 scene id:         $SCENE_ID"
echo "  ⚙️  distill steps:   $ARTIFIXER3D_STEPS"
echo "  📝 phases:           $PHASES"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU:              physical $CUDA_VISIBLE_DEVICES"
fi

# --- checks ---
if [ ! -d "$ARTIFIXER_DIR" ]; then
    echo "❌ ERROR: ArtiFixer code not found at $ARTIFIXER_DIR" >&2
    exit 1
fi
if [ ! -d "$SCENE_ROOT" ]; then
    echo "❌ ERROR: scene root not found: $SCENE_ROOT" >&2
    echo "   Run 02_prepare_data.sh first." >&2
    exit 1
fi
if ! python -c "import threedgrut" 2>/dev/null; then
    echo "❌ ERROR: threedgrut not importable. Run INSTALL_DEPS=1 bash 00_setup_env.sh" >&2
    exit 1
fi

# --- locate ArtiFixer predicted frames ---
if [ -z "$ARTIFIXER_FRAMES_DIR" ]; then
    # Search for the pred directory in the ArtiFixer output
    PRED_DIR=$(find "$ARTIFIXER_SAVE_DIR/$CKPT_STEM" -type d -name "pred" -path "*/frames/batch_0000/*" 2>/dev/null | head -1)
    if [ -n "$PRED_DIR" ] && [ -d "$PRED_DIR" ]; then
        ARTIFIXER_FRAMES_DIR="$PRED_DIR"
    fi
fi

if [ -z "$ARTIFIXER_FRAMES_DIR" ] || [ ! -d "$ARTIFIXER_FRAMES_DIR" ]; then
    echo "❌ ERROR: ArtiFixer prediction frames not found." >&2
    echo "   Expected: $ARTIFIXER_SAVE_DIR/$CKPT_STEM/*/frames/batch_0000/pred/" >&2
    echo "   Run 03_run_inference.sh first." >&2
    echo "   Or set ARTIFIXER_FRAMES_DIR=/path/to/pred explicitly." >&2
    exit 1
fi

echo "  🖼️  artifixer frames: $ARTIFIXER_FRAMES_DIR"

# --- build command ---
CMD=(python -m data_processing.run_artifixer3d
    --scene_root "$SCENE_ROOT"
    --artifixer_frames_dir "$ARTIFIXER_FRAMES_DIR"
    --artifixer3d_steps "$ARTIFIXER3D_STEPS"
    --phases "$PHASES"
)

if [ "$USE_WANDB" = "1" ]; then
    CMD+=(--use_wandb)
fi
if [ "${REPLACE_ARTIFIXER3D:-0}" = "1" ]; then
    CMD+=(--replace)
fi
if [ -n "${BASE_CHECKPOINT:-}" ]; then
    CMD+=(--base_checkpoint "$BASE_CHECKPOINT")
fi
if [ -n "${RENDER_TRAJECTORY_PATH:-}" ]; then
    CMD+=(--render_trajectory_path "$RENDER_TRAJECTORY_PATH")
fi

# --- run ---
cd "$ARTIFIXER_DIR"
export PYTHONPATH="$ARTIFIXER_DIR:${PYTHONPATH:-}"
echo "🔍 running ArtiFixer3D..."
"${CMD[@]}"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ArtiFixer3D" >&2
    exit 1
fi
cd "$SCRIPT_DIR"

# --- verify output ---
SPLIT_3D_PLUS="$SCENE_ROOT/split_artifixer3d_plus.json"
if [ -f "$SPLIT_3D_PLUS" ]; then
    echo "✅ ArtiFixer3D done."
    echo "  📁 3DGRUT distillation: $SCENE_ROOT/artifixer3d/"
    echo "  📁 recon results:       $SCENE_ROOT/artifixer3d/recon_results/"
    echo "  📁 split (3D+):         $SPLIT_3D_PLUS"
else
    echo "⚠️  split_artifixer3d_plus.json not found at $SPLIT_3D_PLUS" >&2
    echo "   Some phases may have been skipped." >&2
fi

echo "🎉 [04] Done."
echo "    Next: bash $SCRIPT_DIR/05_run_artifixer3d_plus.sh  (ArtiFixer3D+ inference)"
