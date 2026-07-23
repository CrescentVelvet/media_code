#!/usr/bin/env bash
# 02_run_inference.sh — batch TRELLIS.2 image-to-3D inference over a folder of
# images. Loads the pipeline once (in run_batch.py) and loops; for each image
# produces a GLB + a latent cache (.latent.npz) for 03_render_video.sh. Set
# RENDER_VIDEO=1 to also write a quick shaded turntable mp4 in the same pass.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

TRELLIS_DIR="${TRELLIS_DIR:-$REPO_DIR/../TRELLIS.2}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/TRELLIS.2-4B}"
INPUT_DIR="${INPUT_DIR:-$TRELLIS_DIR/assets/example_image}"
OUTPUT_DIR="${OUTPUT_DIR:-$TRELLIS_DIR/output}"
# Outputs go to OUTPUT_DIR/<input_folder_name>/ so multiple INPUT_DIR runs
# don't clobber each other (mirrors run_batch.py).
INPUT_NAME="$(basename "$INPUT_DIR")"

echo "=== [02] Batch TRELLIS.2 image-to-3D ==="
echo "  代码路径:  $TRELLIS_DIR"
echo "  模型路径:     $MODEL_DIR"
echo "  输入:     $INPUT_DIR"
echo "  输出:    $OUTPUT_DIR/$INPUT_NAME"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change]"
else
    echo "  GPU:       default cuda:0 (= first visible)  [set GPU=N to pin a card]"
fi

if [ ! -d "$TRELLIS_DIR" ]; then
    echo "ERROR: TRELLIS code dir not found at $TRELLIS_DIR." >&2
    echo "       Run run_all.sh first (it clones the official repo), or set TRELLIS_DIR." >&2
    exit 1
fi
if [ ! -f "$MODEL_DIR/pipeline.json" ]; then
    echo "ERROR: $MODEL_DIR/pipeline.json missing. Run 01_download_models.sh first." >&2
    exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: input dir not found: $INPUT_DIR" >&2
    exit 1
fi

export TRELLIS_DIR MODEL_DIR INPUT_DIR OUTPUT_DIR \
       SEED RESOLUTION DECIMATION_TARGET TEXTURE_SIZE \
       RENDER_VIDEO ENVMAP NUM_FRAMES FPS R FOV LOW_VRAM
python "$SCRIPT_DIR/run_batch.py"

echo "=== [02] Done. Outputs in: $OUTPUT_DIR/$INPUT_NAME ==="
