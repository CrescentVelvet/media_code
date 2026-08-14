#!/usr/bin/env bash
# 02_run_inference.sh — VGGT-Omega feed-forward reconstruction (poses + depth -> point cloud).
# Loads the model once, processes all scenes in INPUT_DIR. Each scene produces
# predictions.npz (raw model outputs: extrinsic, intrinsic, world_points_from_depth,
# depth_conf, images, pose_enc) + scene.ply + scene.glb + frames/.
# Step 02 reads predictions.npz to build the COLMAP scene for 3DGS.
#
# Prerequisites:
#   - INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh (first time)
#   - VGGT-Omega weights downloaded (gated HF; run vggt-omega/01_download_models.sh)
#
# Env (all optional, defaults shown):
#   INPUT_DIR=             # images folder / video / scene folders
#   MODEL_DIR=             # VGGT-Omega checkpoint (gated)
#   RESULTS_DIR=           # output root
#   VGGT_OUTPUT_DIR=       # step 01 output (default: $RESULTS_DIR/vggt)
#   VARIANT=1b_512         # checkpoint variant (must match downloaded weights)
#   RESOLUTION=512         # input resolution (1b_256_text -> 256)
#   MODE=balanced          # balanced | max_size
#   CONF_THRES=20          # depth-confidence percentile (0-100)
#   MAX_POINTS=2000000     # cap on scene.ply points (0=none)
#   VIDEO_FPS=1             # frame sampling fps when INPUT_DIR is a video
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

INPUT_DIR="${INPUT_DIR:-$VGGT_DIR/examples}"
VGGT_OUTPUT_DIR="${VGGT_OUTPUT_DIR:-$RESULTS_DIR/vggt}"
VARIANT="${VARIANT:-1b_512}"
RESOLUTION="${RESOLUTION:-512}"
MODE="${MODE:-balanced}"
CONF_THRES="${CONF_THRES:-20}"
MAX_POINTS="${MAX_POINTS:-2000000}"
VIDEO_FPS="${VIDEO_FPS:-1}"

echo "🚀 [01] VGGT-Omega feed-forward reconstruction"
echo "  🤖 model:       $MODEL_DIR  (variant=$VARIANT)"
echo "  📂 input:       $INPUT_DIR"
echo "  💾 output:      $VGGT_OUTPUT_DIR"
echo "  📐 resolution:  $RESOLUTION ($MODE), conf_thres=$CONF_THRES, max_points=$MAX_POINTS"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU:          physical $CUDA_VISIBLE_DEVICES"
else
    echo "  🎮 GPU:          default cuda:0"
fi
echo ""

# Sanity checks
if [ ! -d "$VGGT_DIR" ]; then
    echo "❌ ERROR: VGGT-Omega code not found at $VGGT_DIR" >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi
case "$VARIANT" in
    1b_512) CKPT_FILE="vggt_omega_1b_512.pt" ;;
    1b_256_text) CKPT_FILE="vggt_omega_1b_256_text.pt" ;;
    *) echo "❌ ERROR: VARIANT must be 1b_512 | 1b_256_text (got '$VARIANT')" >&2; exit 1 ;;
esac
if [ ! -f "$MODEL_DIR/$CKPT_FILE" ]; then
    echo "❌ ERROR: $MODEL_DIR/$CKPT_FILE missing." >&2
    echo "       Download: VARIANT=$VARIANT bash ../vggt-omega/01_download_models.sh" >&2
    exit 1
fi
if [ ! -e "$INPUT_DIR" ]; then
    echo "❌ ERROR: input not found: $INPUT_DIR" >&2
    exit 1
fi

export VGGT_DIR MODEL_DIR INPUT_DIR VARIANT RESOLUTION MODE CONF_THRES MAX_POINTS VIDEO_FPS
export OUTPUT_DIR="$VGGT_OUTPUT_DIR"

python "$SCRIPT_DIR/run_batch.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi

echo ""
echo "✅ [01] Done. Reconstructions in: $VGGT_OUTPUT_DIR"
echo "  Next: bash $SCRIPT_DIR/03_npz_to_colmap.sh"
