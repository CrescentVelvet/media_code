#!/usr/bin/env bash
# 03_extract_frames.sh — split the rotate_360.mp4 into JPG frames under
# <video_name>/image/, matching the standard INPUT_DIR/image/ pattern that
# wan22_rotate step 01 / sam_3d_body / pi3_3dgs all accept as input.
#
# Output structure (alongside the source video):
#   $RESULTS_DIR/
#     rotate_360.mp4          # source video (unchanged, from step 02)
#     rotate_360/             # NEW: same-name folder
#       image/
#         00000.jpg
#         00001.jpg
#         ...
#
# So the resulting `rotate_360/` folder is itself a valid INPUT_DIR (image/
# subfolder pattern) — can be fed back to step 01 or to pi3_3dgs.
#
# Env (all optional):
#   VIDEO_PATH=             # default $RESULTS_DIR/${OUTPUT_NAME:-rotate_360}.mp4
#   FPS=0                   # 0 = every frame at source fps; N = sample at N fps
#   JPG_QUALITY=95          # JPG quality 1-100 (95 ≈ visually lossless)
#   START_FRAME=0           # start frame index (skip intro frames if any)
#   END_FRAME=-1            # end frame index, -1 = until end
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

OUTPUT_NAME="${OUTPUT_NAME:-rotate_360}"
VIDEO_PATH="${VIDEO_PATH:-$RESULTS_DIR/${OUTPUT_NAME}.mp4}"
FPS="${FPS:-0}"                # 0 = extract every frame at source fps
JPG_QUALITY="${JPG_QUALITY:-95}"
START_FRAME="${START_FRAME:-0}"
END_FRAME="${END_FRAME:--1}"

if [ ! -f "$VIDEO_PATH" ]; then
    echo "❌ ERROR: video not found: $VIDEO_PATH" >&2
    echo "       Run step 02 first, or set VIDEO_PATH=/path/to/video.mp4" >&2
    exit 1
fi

# cv2 is a step 01 dep (sam_3d_body uses it); should always be importable in
# the wan22_rotate env. If not, env is broken — point user at 00_setup_env.sh.
if ! python -c "import cv2" 2>/dev/null; then
    echo "❌ ERROR: cv2 not importable in env '$CONDA_ENV'." >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

echo "🚀 [03] extract video frames -> JPGs"
echo "  🎬 video:     $VIDEO_PATH"
echo "  📐 fps:       $FPS (0 = every frame at source fps)"
echo "  🖼️  quality:   $JPG_QUALITY"

export VIDEO_PATH FPS JPG_QUALITY
export START_FRAME END_FRAME

python "$SCRIPT_DIR/extract_frames.py"

if [ $? -ne 0 ]; then
    echo "❌ [03] FAILED. Frame extraction did not complete." >&2
    exit 1
fi

echo "🎉 [03] Done."
