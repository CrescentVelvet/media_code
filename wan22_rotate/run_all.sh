#!/usr/bin/env bash
# run_all.sh — one-click: pick front-facing image + segment person -> generate
# 360 rotation video with Wan2.2-TI2V-5B + your LoRA.
#
# Step 01 runs in the sam_3d_body env (detection + 3D pose + segmentation).
# Step 02 runs in the wan22 env (DiffSynth-Studio video generation).
# They have conflicting pins so they MUST use separate conda envs.
#
# REQUIRED:
#   INPUT_DIR=/path/to/subject_folder     (contains image/ subfolder)
#   WEIGHT_PATH=/path/to/epoch-N.safetensors  (trained LoRA)
#
# Common overrides (all optional):
#   GPU=0 PROMPT="..." HEIGHT=1248 WIDTH=706 NUM_FRAMES=121
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "=== [run_all] wan22_rotate: pick+segment -> 360 video ==="
echo "  step 01 env: ${SAM3D_ENV:-sam_3d_body}"
echo "  step 02 env: ${WAN_ENV:-wan22}"
echo ""

# --- step 01: pick front-facing image + segment ---
if [ "${SKIP_SEGMENT:-0}" = "1" ]; then
    echo "--- [skip 01] SKIP_SEGMENT=1, using existing segmented image ---"
else
    bash "$SCRIPT_DIR/01_pick_and_segment.sh"
    echo ""
fi

# --- step 02: generate 360 rotation video ---
bash "$SCRIPT_DIR/02_generate_video.sh"

echo ""
echo "=== [run_all] Done. ==="
echo "  Segmented image: $RESULTS_DIR/segmented_image.png"
echo "  Video:           $RESULTS_DIR/${OUTPUT_NAME:-rotate_360}.mp4"
