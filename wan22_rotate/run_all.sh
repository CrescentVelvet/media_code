#!/usr/bin/env bash
# run_all.sh — one-click: pick front-facing image + segment person -> generate
# 360 rotation video with Wan2.2-TI2V-5B + your LoRA -> split video into JPGs
# -> Pi3 pose estimation (no 3D reconstruction).
#
# Steps 01-04 all run in the same conda env (wan22_rotate, cloned from doll;
# has sam_3d_body + diffsynth deps installed together — Pi3 deps overlap).
# Step 04 calls pi3_3dgs/pi3_recon.py with --no_colmap (skips COLMAP export
# + 2DGS training); drop SKIP_PI3=0 to run it.
#
# REQUIRED:
#   INPUT_DIR=/path/to/subject_folder     (contains image/ subfolder)
#   WEIGHT_PATH=/path/to/epoch-N.safetensors  (trained LoRA)
#
# Common overrides (all optional):
#   GPU=0 PROMPT="..." HEIGHT=1248 WIDTH=706 NUM_FRAMES=121
#   SKIP_EXTRACT=1                        (skip step 03 frame extraction)
#   SKIP_PI3=1                            (skip step 04 Pi3 pose estimation)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [run_all] wan22_rotate: pick+segment -> 360 video -> JPG frames -> Pi3 poses"
echo "  conda env: $CONDA_ENV"
echo ""

# --- step 01: pick + segment (01=full SAM 3D Body, 01b=simplified SAM only) ---
if [ "${SKIP_SEGMENT:-0}" = "1" ]; then
    echo "⏭️ [skip 01] SKIP_SEGMENT=1, using existing segmented image"
else
    PICK_SCRIPT="${PICK_SCRIPT:-01_pick_and_segment.sh}"
    bash "$SCRIPT_DIR/$PICK_SCRIPT"
    echo ""
fi

# --- step 02: generate 360 rotation video ---
bash "$SCRIPT_DIR/02_generate_video.sh"

# --- step 03: split video into JPGs (matches INPUT_DIR/image/ pattern) ---
if [ "${SKIP_EXTRACT:-0}" = "1" ]; then
    echo "⏭️ [skip 03] SKIP_EXTRACT=1"
else
    echo ""
    bash "$SCRIPT_DIR/03_extract_frames.sh"
fi

# --- step 04: Pi3 pose estimation (no 3D reconstruction) ---
if [ "${SKIP_PI3:-0}" = "1" ]; then
    echo "⏭️ [skip 04] SKIP_PI3=1"
else
    echo ""
    bash "$SCRIPT_DIR/04_pi3_pose.sh"
fi

echo ""
echo "🎉 [run_all] Done."
echo "  🖼️  Segmented image: $RESULTS_DIR/segmented_image.png"
echo "  🎬 Video:            $RESULTS_DIR/${OUTPUT_NAME:-rotate_360}.mp4"
echo "  📁 Frames:            $RESULTS_DIR/${OUTPUT_NAME:-rotate_360}/image/*.jpg"
echo "  📊 Pi3 poses:         $RESULTS_DIR/${OUTPUT_NAME:-rotate_360}/pi3/{predictions.npz,poses.json,dense_cloud.ply}"
