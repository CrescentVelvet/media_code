#!/usr/bin/env bash
# 02_run_inference.sh — batch VGGT-Omega reconstruction over scenes.
#
# INPUT_DIR is one of:
#   - a folder of images            -> one scene (named after the folder)
#   - a video file (.mp4/.mov/...)  -> one scene (frames extracted at VIDEO_FPS)
#   - a folder of scene folders     -> one reconstruction per subfolder (batch)
#   - a folder of videos            -> one reconstruction per video (batch)
# Each scene's images (copied/extracted) live under <out>/frames/.
#
# The model is loaded ONCE (run_batch.py) and reused across scenes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

VGGT_DIR="${VGGT_DIR:-$REPO_DIR/../vggt-omega}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/VGGT-Omega}"
INPUT_DIR="${INPUT_DIR:-$VGGT_DIR/examples}"
OUTPUT_DIR="${OUTPUT_DIR:-$VGGT_DIR/output}"
VARIANT="${VARIANT:-1b_512}"          # 1b_512 | 1b_256_text  (must match 01)
RESOLUTION="${RESOLUTION:-512}"       # 1b_512->512, 1b_256_text->256
MODE="${MODE:-balanced}"              # balanced | max_size  (see load_fn.py)
CONF_THRES="${CONF_THRES:-20}"        # depth-confidence percentile kept (0-100)
MAX_POINTS="${MAX_POINTS:-2000000}"   # cap on points saved to scene.ply (0=none)
VIDEO_FPS="${VIDEO_FPS:-1}"           # frame sampling fps when INPUT_DIR is a video

echo "=== [02] Batch VGGT-Omega reconstruction ==="
echo "  代码路径:  $VGGT_DIR"
echo "  模型路径:     $MODEL_DIR  (variant=$VARIANT)"
echo "  输入:     $INPUT_DIR"
echo "  输出:    $OUTPUT_DIR"
echo "  分辨率:       $RESOLUTION ($MODE), conf_thres=$CONF_THRES, max_points=$MAX_POINTS"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change]"
else
    echo "  GPU:       default cuda:0 (= first visible)  [set GPU=N to pin a card]"
fi

if [ ! -d "$VGGT_DIR" ]; then
    echo "ERROR: VGGT-Omega code dir not found at $VGGT_DIR." >&2
    exit 1
fi
case "$VARIANT" in
    1b_512) CKPT_FILE="vggt_omega_1b_512.pt" ;;
    1b_256_text) CKPT_FILE="vggt_omega_1b_256_text.pt" ;;
    *) echo "ERROR: VARIANT must be 1b_512 | 1b_256_text (got '$VARIANT')" >&2; exit 1 ;;
esac
if [ ! -f "$MODEL_DIR/$CKPT_FILE" ]; then
    echo "ERROR: $MODEL_DIR/$CKPT_FILE missing. Run 01_download_models.sh VARIANT=$VARIANT first." >&2
    exit 1
fi
if [ ! -e "$INPUT_DIR" ]; then
    echo "ERROR: input not found: $INPUT_DIR" >&2
    exit 1
fi

export VGGT_DIR MODEL_DIR INPUT_DIR OUTPUT_DIR VARIANT RESOLUTION MODE CONF_THRES MAX_POINTS VIDEO_FPS
python "$SCRIPT_DIR/run_batch.py"

echo "=== [02] Done. Reconstructions in: $OUTPUT_DIR ==="
