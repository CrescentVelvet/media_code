#!/usr/bin/env bash
# 01_pi3_recon.sh — extract frames, run Pi3 feed-forward pose+point inference,
# export to COLMAP text format (for 2DGS train.py).
#
# Input:  video file (.mp4/.mov) OR folder of images (env: INPUT)
# Output: $OUTPUT_DIR/{frames, predictions.npz, dense_cloud.ply, poses.json, source/}
#         source/ contains images/ + sparse/0/{cameras,images,points3D}.txt
#         (--no_colmap skips source/; wan22_rotate step 04 uses that mode)
#
# Env (all optional, defaults shown):
#   INPUT=/path/to/rotate_360.mp4   # required if --input not given
#   OUTPUT_DIR=../pi3_3dgs_results  # output root
#   FRAME_FPS=10                    # frame sampling fps when INPUT is a video
#   FRAME_MAX=60                    # max frames to keep (avoid OOM; Pi3 scales ~linearly)
#   CONF_THRES=0.1                  # sigmoid-conf threshold for init point filter
#   MAX_POINTS=100000               # cap on init COLMAP points (3DGS densifies from there)
#   FX/FY/CX/CY=                    # override assumed PINHOLE intrinsics (default: fx=max(W,H))
#   DEVICE=cuda                     # or cpu (slow)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

INPUT="${INPUT:?ERROR: INPUT env var required (video file or image folder)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR}"
export INPUT OUTPUT_DIR

# Sanity: Pi3 repo + ckpt must be present (run 00_setup_env.sh first).
if [ ! -d "$PI3_DIR" ]; then
    echo "ERROR: Pi3 repo not found at $PI3_DIR" >&2
    echo "       Run INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh first" >&2
    exit 1
fi
if [ ! -f "$PI3_CKPT" ]; then
    echo "ERROR: Pi3 checkpoint not found at $PI3_CKPT" >&2
    echo "       Download from https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors" >&2
    echo "       Place at: $PI3_CKPT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=== [01] Pi3 reconstruction (pose+point -> COLMAP) ==="
echo "  input:      $INPUT"
echo "  output_dir: $OUTPUT_DIR"
echo "  ckpt:       $PI3_CKPT"
echo "  frame_fps:  ${FRAME_FPS:-10}  frame_max: ${FRAME_MAX:-60}  conf_thres: ${CONF_THRES:-0.1}"
echo ""

python "$SCRIPT_DIR/pi3_recon.py" \
    --input "$INPUT" \
    --output_dir "$OUTPUT_DIR" \
    --ckpt "$PI3_CKPT" \
    --pi3_dir "$PI3_DIR" \
    --device "${DEVICE:-cuda}" \
    --frame_fps "${FRAME_FPS:-10}" \
    --frame_max "${FRAME_MAX:-60}" \
    --conf_thres "${CONF_THRES:-0.1}" \
    --max_points "${MAX_POINTS:-100000}" \
    ${FX:+--fx "$FX"} ${FY:+--fy "$FY"} ${CX:+--cx "$CX"} ${CY:+--cy "$CY"}

echo ""
echo "=== [01] Done. Pi3 reconstruction exported to COLMAP format. ==="
echo "    Source dir: $OUTPUT_DIR/source"
echo "    Next: SOURCE_DIR=$OUTPUT_DIR/source GPU=0 bash $SCRIPT_DIR/02_train_2dgs.sh"
