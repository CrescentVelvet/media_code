#!/usr/bin/env bash
# 01a_video_to_frames.sh — Video → image folder (frames under image/), to feed into 01_face_enhance.sh.
#
# Preprocessing step before 01: extracts frames from a single video at VIDEO_FPS
# into <OUTPUT_DIR>/image/ (000000.png, 000001.png, ...). Output structure matches
# the test_task input format, so 01's INPUT_DIR points at <OUTPUT_DIR> and
# face_enhance.py auto-detects the image/ subfolder. Full pipeline works unchanged:
#   01a (video -> frames) -> 01 (face enhance) -> 02 (VGGT-Omega) -> 03 (COLMAP) -> 04 (3DGS)
#
# Env (all optional, defaults shown):
#   INPUT_DIR=             # single video file (.mp4/.mov/.avi/.mkv), required
#   RESULTS_DIR=           # output root
#   OUTPUT_DIR=             # frames output parent (frames go to <OUTPUT_DIR>/image/)
#   VIDEO_FPS=2            # frame sampling fps
#   BLUR_THRESHOLD=100     # Laplacian variance below this = skip (0=disable blur gate)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

INPUT_DIR="${INPUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/input_frames}"
VIDEO_FPS="${VIDEO_FPS:-2}"
BLUR_THRESHOLD="${BLUR_THRESHOLD:-100}"

echo "🎬 [01a] Video → image folder (frames)"
echo "  🎬 input video: $INPUT_DIR"
echo "  💾 output:      $OUTPUT_DIR"
echo "  📐 video_fps:   $VIDEO_FPS"
echo "  🔍 blur gate:   BLUR_THRESHOLD=$BLUR_THRESHOLD (0=off)"
echo ""

if [ -z "$INPUT_DIR" ]; then
    echo "❌ ERROR: INPUT_DIR not set (expect a video file path)" >&2
    exit 1
fi
if [ ! -f "$INPUT_DIR" ]; then
    echo "❌ ERROR: video not found: $INPUT_DIR" >&2
    exit 1
fi

export INPUT_DIR OUTPUT_DIR VIDEO_FPS BLUR_THRESHOLD
python "$SCRIPT_DIR/video_to_frames.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi

echo ""
echo "✅ [01a] Done. Frames in: $OUTPUT_DIR"
echo "  Next: GPU=0 INPUT_DIR=$OUTPUT_DIR bash $SCRIPT_DIR/01_face_enhance.sh"
