#!/usr/bin/env bash
# 05_run_raw.sh — one-click: Pi3 + PDF-GS on ORIGINAL unsegmented images.
#
# Skips step 01 (segmentation) and trains WITHOUT --white_background.
# Why: white-background segmentation can cause 3DGS artifacts — radiating
# white "wings" at silhouette edges and overall blur — because the flat
# white background isn't real 3D geometry; Gaussians can't reconstruct it
# and the artifacts bleed into the person. Using original images with real
# backgrounds gives the model proper multi-view geometry everywhere, and
# PDF-GS's distractor filtering handles micro-motion regardless of background.
#
# Prerequisites:
#   - INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh (first time)
#   - Original images in a folder (INPUT_DIR)
#
# Env (all optional, defaults shown):
#   INPUT_DIR=               # original image folder (REQUIRED, no default)
#   GPU=0                    # physical GPU id
#   OUTPUT_NAME=orbit_raw    # base name (keeps raw output separate from segmented)
#   RESULTS_DIR=              # output root
#   FRAME_MAX=60             # max frames for Pi3
#   NUM_PHASES=4             # PDF-GS phases
#   ITER_PER_PHASE=10000     # iters per phase
#   WHITE_BG=0               # forced to 0 (no white background — that's the point)
#   SKIP_PI3=0               # 1 = reuse existing pi3/source/
#   SKIP_TRAIN=0             # 1 = reuse existing model_pdfgs/
#   SKIP_RENDER=0            # 1 = skip orbit video
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

INPUT_DIR="${INPUT_DIR:-}"
OUTPUT_NAME="${OUTPUT_NAME:-orbit_raw}"

if [ -z "$INPUT_DIR" ]; then
    echo "❌ ERROR: INPUT_DIR not set — original image folder." >&2
    echo "  Usage: INPUT_DIR=../your_images GPU=0 bash $SCRIPT_DIR/05_run_raw.sh" >&2
    exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ ERROR: INPUT_DIR not found: $INPUT_DIR" >&2
    exit 1
fi

echo "🚀 [05] Raw pipeline: Pi3 + PDF-GS on ORIGINAL (unsegmented) images"
echo "  📂 input:       $INPUT_DIR"
echo "  💾 output:      $RESULTS_DIR/$OUTPUT_NAME"
echo "  🎨 white_bg:    0 (no white background — using real scene)"
[ -n "${GPU:-}" ] && echo "  🎮 GPU:         $GPU"
echo ""

# ── 1. Pi3 + COLMAP (on original images, no segmentation) ──────────────
echo "━━━ [1/3] Pi3 pose + COLMAP ━━━"
INPUT="$INPUT_DIR" OUTPUT_NAME="$OUTPUT_NAME" \
    bash "$SCRIPT_DIR/02_pi3_colmap.sh"
if [ $? -ne 0 ]; then
    echo "❌ Step 02 (Pi3 + COLMAP) failed." >&2
    exit 1
fi
echo ""

# ── 2. PDF-GS training (without white background) ───────────────────────
echo "━━━ [2/3] PDF-GS training ━━━"
WHITE_BG=0 OUTPUT_NAME="$OUTPUT_NAME" \
    bash "$SCRIPT_DIR/03_train_pdfgs.sh"
if [ $? -ne 0 ]; then
    echo "❌ Step 03 (PDF-GS training) failed." >&2
    exit 1
fi
echo ""

# ── 3. Orbit render ────────────────────────────────────────────────────
echo "━━━ [3/3] Orbit render ━━━"
OUTPUT_NAME="$OUTPUT_NAME" \
    bash "$SCRIPT_DIR/04_render_orbit.sh"
if [ $? -ne 0 ]; then
    echo "❌ Step 04 (orbit render) failed." >&2
    exit 1
fi

echo ""
echo "🎉 [05] Raw pipeline done."
_NUM_PHASES="${NUM_PHASES:-4}"
_ITER="${ITER_PER_PHASE:-10000}"
echo "  🏋️ Gaussians:  $RESULTS_DIR/$OUTPUT_NAME/model_pdfgs/phase_$_NUM_PHASES/point_cloud/iteration_$_ITER/point_cloud.ply"
echo "  🎬 Orbit:      $RESULTS_DIR/$OUTPUT_NAME/model_pdfgs/orbit_render/${OUTPUT_NAME}_orbit.mp4"
