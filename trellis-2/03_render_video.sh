#!/usr/bin/env bash
# 03_render_video.sh — render TRELLIS.2 latents (.latent.npz from 02) to mp4.
# Loads the pipeline once (only the decoders are exercised), decodes each latent
# and renders a turntable video (PBR shaded, or the official composite grid).
# No 3-stage generation is repeated — this decodes cached latents, like the
# official app.py preview -> extract flow.
# Output path mirrors 02: VIDEOS_DIR/<input_folder_name>/<stem>.mp4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

TRELLIS_DIR="${TRELLIS_DIR:-$REPO_DIR/../TRELLIS.2}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/TRELLIS.2-4B}"
LATENT_INPUT="${LATENT_INPUT:-$TRELLIS_DIR/output}"
VIDEOS_DIR="${VIDEOS_DIR:-$TRELLIS_DIR/videos}"
RENDER_MODE="${RENDER_MODE:-shaded}"   # shaded (turntable) | pbr (composite grid, official example.py)
RENDER_RES="${RENDER_RES:-1024}"
NUM_FRAMES="${NUM_FRAMES:-120}"
FPS="${FPS:-15}"
R="${R:-2}"
FOV="${FOV:-40}"
ENVMAP="${ENVMAP:-forest}"   # forest / sunset / courtyard / none

# Nest by input folder name (same logic as 02_run_inference.sh).
if [ -d "$LATENT_INPUT" ]; then
    INPUT_NAME="$(basename "$LATENT_INPUT")"
else
    INPUT_NAME="$(basename "$(dirname "$LATENT_INPUT")")"
fi

echo "=== [03] Render TRELLIS.2 latents -> mp4 ==="
echo "  latents:   $LATENT_INPUT"
echo "  videos:    $VIDEOS_DIR/$INPUT_NAME"
echo "  mode:      $RENDER_MODE  (${NUM_FRAMES}f@${FPS}fps, r=$R, fov=$FOV, res=$RENDER_RES, envmap=$ENVMAP)"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change]"
else
    echo "  GPU:       default cuda:0 (= first visible)  [set GPU=N to pin a card]"
fi

if [ ! -f "$MODEL_DIR/pipeline.json" ]; then
    echo "ERROR: $MODEL_DIR/pipeline.json missing. Run 01_download_models.sh first." >&2
    exit 1
fi
if [ ! -e "$LATENT_INPUT" ]; then
    echo "ERROR: LATENT_INPUT not found: $LATENT_INPUT" >&2
    echo "       Point it at a folder of .latent.npz (from 02) or a single .latent.npz." >&2
    exit 1
fi

export TRELLIS_DIR MODEL_DIR LATENT_INPUT VIDEOS_DIR INPUT_NAME RENDER_MODE RENDER_RES NUM_FRAMES FPS R FOV ENVMAP
python "$SCRIPT_DIR/render_video.py"

echo "=== [03] Done. Videos in: $VIDEOS_DIR/$INPUT_NAME ==="
