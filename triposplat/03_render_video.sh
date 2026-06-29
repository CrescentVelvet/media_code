#!/usr/bin/env bash
# 03_render_video.sh — render .ply to mp4 along a spiral trajectory (gsplat).
# Output path mirrors 02: VIDEOS_DIR/<input_folder_name>/<stem>.mp4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

TRIPOSPLAT_DIR="${TRIPOSPLAT_DIR:-$REPO_DIR/../TripoSplat}"
PLY_INPUT="${PLY_INPUT:-$TRIPOSPLAT_DIR/output}"
VIDEOS_DIR="${VIDEOS_DIR:-$TRIPOSPLAT_DIR/videos}"
WIDTH="${WIDTH:-1080}"
HEIGHT="${HEIGHT:-720}"
FRAMES="${FRAMES:-81}"
FPS="${FPS:-15}"
TURNS="${TURNS:-2}"
ELEV="${ELEV:-30}"
START_ANGLE="${START_ANGLE:-0}"   # starting left-right (azimuth) angle in degrees
FOV="${FOV:-60}"
UP_AXIS="${UP_AXIS:-z}"
UP_VEC="${UP_VEC:-0 -1 0}"   # object's up direction (overrides UP_AXIS); "0 -1 0" = -Y (upright for TripoSplat)
ROLL="${ROLL:-0}"

# Nest by input folder name (same logic as 02_run_inference.sh).
if [ -d "$PLY_INPUT" ]; then
    INPUT_NAME="$(basename "$PLY_INPUT")"
else
    INPUT_NAME="$(basename "$(dirname "$PLY_INPUT")")"
fi

echo "=== [03] Render .ply -> mp4 (spiral) ==="
echo "  ply input: $PLY_INPUT"
echo "  videos:    $VIDEOS_DIR/$INPUT_NAME"
echo "  spiral:    ${WIDTH}x${HEIGHT}, ${FRAMES}f@${FPS}fps, turns=$TURNS, elev=$ELEV°, start=$START_ANGLE°, fov=$FOV°, up=$UP_VEC, roll=$ROLL°"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change]"
else
    echo "  GPU:       default cuda:0 (= first visible)  [set GPU=N to pin a card]"
fi

if ! python -c "import gsplat, plyfile, imageio, imageio_ffmpeg" 2>/dev/null; then
    echo "ERROR: missing deps. Install with: pip install gsplat plyfile imageio imageio-ffmpeg" >&2
    exit 1
fi
if [ ! -e "$PLY_INPUT" ]; then
    echo "ERROR: PLY_INPUT not found: $PLY_INPUT" >&2
    echo "       Point it at a folder of .ply (e.g. ../TripoSplat/output/setA) or a single .ply." >&2
    exit 1
fi

export PLY_INPUT VIDEOS_DIR INPUT_NAME WIDTH HEIGHT FRAMES FPS TURNS ELEV START_ANGLE FOV UP_AXIS UP_VEC ROLL TRIPOSPLAT_DIR
python "$SCRIPT_DIR/render_video.py"

echo "=== [03] Done. Videos in: $VIDEOS_DIR/$INPUT_NAME ==="
