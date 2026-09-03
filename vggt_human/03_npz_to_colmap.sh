#!/usr/bin/env bash
# 03_npz_to_colmap.sh — Convert VGGT-Omega predictions.npz to COLMAP text format.
#
# Reads the npz from step 01 (predictions.npz: extrinsic w2c, intrinsic,
# world_points_from_depth, depth_conf, images) and produces a COLMAP scene:
#   source/images/<frame>.png
#   source/sparse/0/cameras.txt      # PINHOLE (actual VGGT-Omega intrinsics)
#   source/sparse/0/images.txt       # w2c quaternion + translation
#   source/sparse/0/points3D.txt     # adaptive-conf filtered + voxel-downsampled
#
# VGGT-Omega outputs w2c directly (OpenCV convention = COLMAP convention), so no
# c2w->w2c inversion is needed (unlike Pi3). Intrinsics are the model's actual
# predictions (not assumed fx=fy=max(W,H) like Pi3).
#
# Prerequisites: step 01 (VGGT-Omega inference) already run.
#
# Env (all optional, defaults shown):
#   RESULTS_DIR=             # output root
#   VGGT_OUTPUT_DIR=         # step 01 output (default: $RESULTS_DIR/vggt)
#   SCENE_NAME=              # scene subfolder (auto-detected if unset)
#   SOURCE_DIR=              # COLMAP output (default: $RESULTS_DIR/source)
#   TARGET_POINTS=200000     # voxel downsample target
#   ALIGN=1                  # 1 = center scene at origin
#   INTRINSIC_ZSCORE=3.0     # fx/fy z-score above this = anomaly (robust median+MAD)
#   INTRINSIC_REJECT=0       # 1 = remove anomalous frames; 0 = report only
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

VGGT_OUTPUT_DIR="${VGGT_OUTPUT_DIR:-$RESULTS_DIR/vggt}"
SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/source}"
TARGET_POINTS="${TARGET_POINTS:-200000}"
ALIGN="${ALIGN:-1}"
INTRINSIC_ZSCORE="${INTRINSIC_ZSCORE:-3.0}"
INTRINSIC_REJECT="${INTRINSIC_REJECT:-0}"

echo "✂️ [02] npz -> COLMAP conversion"
echo "  📂 VGGT output: $VGGT_OUTPUT_DIR"
echo "  💾 COLMAP src:  $SOURCE_DIR"
echo "  📐 target pts:  $TARGET_POINTS  align=$ALIGN"
echo "  📏 intrinsic:   zscore=$INTRINSIC_ZSCORE  reject=$INTRINSIC_REJECT"
echo ""

# Auto-detect scene subfolder if SCENE_NAME not set
if [ -z "${SCENE_NAME:-}" ]; then
    SCENE_NAME="$(ls -d "$VGGT_OUTPUT_DIR"/*/ 2>/dev/null | head -1 | xargs -I{} basename {} || true)"
    if [ -z "$SCENE_NAME" ]; then
        # Maybe the npz is directly in VGGT_OUTPUT_DIR (no scene subfolder)
        SCENE_NAME=""
    fi
fi

NPZ_PATH="$VGGT_OUTPUT_DIR/${SCENE_NAME:+$SCENE_NAME/}predictions.npz"
FRAMES_DIR="$VGGT_OUTPUT_DIR/${SCENE_NAME:+$SCENE_NAME/}frames"

echo "  📄 npz:          $NPZ_PATH"
echo "  🖼️ frames:       $FRAMES_DIR"
[ -n "$SCENE_NAME" ] && echo "  🎯 scene:        $SCENE_NAME"
echo ""

if [ ! -f "$NPZ_PATH" ]; then
    echo "❌ ERROR: predictions.npz not found at $NPZ_PATH" >&2
    echo "       Run step 02 first: INPUT_DIR=... bash $SCRIPT_DIR/02_run_inference.sh" >&2
    exit 1
fi

export NPZ_PATH FRAMES_DIR SOURCE_DIR TARGET_POINTS ALIGN INTRINSIC_ZSCORE INTRINSIC_REJECT
python "$SCRIPT_DIR/npz_to_colmap.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi

echo ""
echo "✅ [02] Done. COLMAP scene at: $SOURCE_DIR"
echo "  Next: GPU=0 bash $SCRIPT_DIR/04_train_3dgs.sh"
