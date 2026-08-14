#!/usr/bin/env bash
# 04_denoise_novel.sh — Render novel views → denoise → AdaIN → augmented COLMAP.
#
# Two-stage pipeline (separate Python processes to avoid GPU memory conflicts):
#   Stage 1 (render_novel.py): Load 3DGS model from step 03 → find angular gaps
#     in camera trajectory → insert NUM_NOVEL_VIEWS intermediate viewpoints →
#     render each (black+white bg for alpha) → save PNG + novel_poses.json.
#   Stage 2 (denoise_images.py): For each novel view with alpha < ALPHA_THRESH
#     (sparse region): denoise (DENOISER switchable) → AdaIN color-correct to
#     nearest training image → write augmented COLMAP scene (orig + novel cams).
#
# Prerequisites: step 01 + 02 + 03 already run (need 3DGS checkpoint + COLMAP scene).
#
# Env (all optional, defaults shown):
#   DENOISER=none           # diffbir | swinir | nafnet | none (identity)
#   NUM_NOVEL_VIEWS=10       # how many virtual cameras to insert
#   ALPHA_THRESH=0.3         # render alpha < this = sparse, needs denoising
#   ADAIN_REF=nearest        # nearest (closest train img) | mean (global avg color)
#   ITERATION=30000          # which 3DGS checkpoint iteration to load
#   RESULTS_DIR=             # output root
#   SOURCE_DIR=              # original COLMAP scene (default: $RESULTS_DIR/source)
#   GAUSSIAN_DIR=            # 3DGS model dir (default: $RESULTS_DIR/model_3dgs)
#   SOURCE_AUG_DIR=          # augmented COLMAP output (default: $RESULTS_DIR/source_aug)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

DENOISER="${DENOISER:-none}"
NUM_NOVEL_VIEWS="${NUM_NOVEL_VIEWS:-10}"
ALPHA_THRESH="${ALPHA_THRESH:-0.3}"
ADAIN_REF="${ADAIN_REF:-nearest}"
ITERATION="${ITERATION:-30000}"
GAUSSIAN_DIR="${GAUSSIAN_DIR:-$RESULTS_DIR/model_3dgs}"
SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/source}"
SOURCE_AUG_DIR="${SOURCE_AUG_DIR:-$RESULTS_DIR/source_aug}"
DEVICE="${DEVICE:-cuda}"

echo "🚀 [04] Render novel views → denoise → AdaIN → augmented COLMAP"
echo "  🤖 denoiser:     $DENOISER"
echo "  📐 novel views:  $NUM_NOVEL_VIEWS  alpha_thresh=$ALPHA_THRESH"
echo "  🎨 adain_ref:    $ADAIN_REF"
echo "  🏋️ 3DGS model:   $GAUSSIAN_DIR (iter=$ITERATION)"
echo "  📂 source:        $SOURCE_DIR"
echo "  💾 source_aug:    $SOURCE_AUG_DIR"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU:           physical $CUDA_VISIBLE_DEVICES"
fi
echo ""

# Sanity checks
if [ ! -f "$GAUSSIAN_DIR/point_cloud/iteration_$ITERATION/point_cloud.ply" ]; then
    echo "❌ ERROR: 3DGS checkpoint not found at $GAUSSIAN_DIR/point_cloud/iteration_$ITERATION/point_cloud.ply" >&2
    echo "       Run step 03 first: bash $SCRIPT_DIR/03_train_3dgs.sh" >&2
    exit 1
fi
if [ ! -d "$SOURCE_DIR/images" ] || [ ! -d "$SOURCE_DIR/sparse/0" ]; then
    echo "❌ ERROR: COLMAP scene not ready: $SOURCE_DIR" >&2
    echo "       Run step 02 first: bash $SCRIPT_DIR/02_npz_to_colmap.sh" >&2
    exit 1
fi

# ── Stage 1: Render novel views (GPU: 3DGS) ─────────────────────────────────
echo "━━ Stage 1: 3DGS rendering ━━"
export GS_DIR GAUSSIAN_DIR SOURCE_DIR RESULTS_DIR ITERATION NUM_NOVEL_VIEWS DEVICE
python "$SCRIPT_DIR/render_novel.py"
if [ $? -ne 0 ]; then
    echo "❌ Stage 1 (render) FAILED" >&2
    exit 1
fi

# Clear GPU memory between stages (3DGS model → denoiser model)
echo ""
echo "🧹 clearing GPU memory between stages..."
python -c "import torch; torch.cuda.empty_cache(); print('  ✅ cleared')" 2>/dev/null || true

# ── Stage 2: Denoise + AdaIN + augmented COLMAP (GPU: denoiser) ────────────
echo ""
echo "━━ Stage 2: denoise + AdaIN + COLMAP export ━━"
export DENOISER ALPHA_THRESH ADAIN_REF SOURCE_AUG_DIR DEVICE
python "$SCRIPT_DIR/denoise_images.py"
if [ $? -ne 0 ]; then
    echo "❌ Stage 2 (denoise) FAILED" >&2
    exit 1
fi

echo ""
echo "🎉 [04] Done. Augmented scene: $SOURCE_AUG_DIR"
echo "  Next: GPU=0 bash $SCRIPT_DIR/05_train_denoise.sh"
