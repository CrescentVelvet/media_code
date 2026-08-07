#!/usr/bin/env bash
# 01_pick_and_segment.sh — pick the front-facing image from an orbit shoot
# and segment the person (background -> white).
#
# Runs in the wan22_rotate conda env (cloned from doll; has both sam_3d_body +
# diffsynth deps). Calls pick_and_segment.py which:
#   1. Loads SAM 3D Body (model + detector + segmentor + FOV estimator)
#   2. Processes every image in INPUT_DIR/image/ (recursive walk)
#   3. For each: 3D body estimation -> global_rot -> front-facing score
#   4. Picks the image with the best score
#   5. Segments the person (segmentor > 3D mesh silhouette > bbox fallback)
#   6. Applies mask: person kept, background -> white
#   7. Saves: $OUTPUT_DIR/segmented_image.png
#
# REQUIRED:  INPUT_DIR=/path/to/subject_folder  (must contain image/ subfolder)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# --- sam_3d_body model config ---
HF_REPO_ID="${HF_REPO_ID:-facebook/sam-3d-body-dinov3}"
CKPT_DIR="$SAM3D_MODEL_DIR/$(basename "$HF_REPO_ID")"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CKPT_DIR/model.ckpt}"
export MHR_PATH="${MHR_PATH:-$CKPT_DIR/assets/mhr_model.pt}"
export DEVICE="${DEVICE:-cuda}"
export DETECTOR_NAME="${DETECTOR_NAME:-vitdet}"
export DETECTOR_PATH="${DETECTOR_PATH:-}"
export SEGMENTOR_NAME="${SEGMENTOR_NAME:-sam2}"
export SEGMENTOR_PATH="${SEGMENTOR_PATH:-}"
export FOV_NAME="${FOV_NAME:-moge2}"
export FOV_PATH="${FOV_PATH:-$SAM3D_MODEL_DIR/moge-2-vitl-normal}"
export SAM3D_DIR

# --- pipeline params ---
export BBOX_THRESH="${BBOX_THRESH:-0.8}"
export INFERENCE_TYPE="${INFERENCE_TYPE:-body}"
export FRONTAL_SIGN="${FRONTAL_SIGN:-1}"
export WHITE_BG="${WHITE_BG:-1}"
export PADDING="${PADDING:-0.1}"

# --- I/O ---
export INPUT_DIR="${INPUT_DIR:-}"
if [ -z "$INPUT_DIR" ]; then
    echo "ERROR: set INPUT_DIR=/path/to/subject_folder (must contain image/ subfolder)" >&2
    echo "  e.g. INPUT_DIR=/data/subject_001 bash $0" >&2
    exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: INPUT_DIR not found: $INPUT_DIR" >&2; exit 1
fi
if [ ! -d "$INPUT_DIR/image" ]; then
    echo "WARNING: no 'image' subfolder in $INPUT_DIR; will use $INPUT_DIR itself" >&2
fi

export OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR}"

# --- checks ---
if [ ! -d "$SAM3D_DIR" ]; then
    echo "ERROR: sam-3d-body code not found at $SAM3D_DIR." >&2
    echo "       Run: INSTALL_DEPS=1 HF_TOKEN=hf_xxx bash $REPO_DIR/sam_3d_body/run_all.sh" >&2
    exit 1
fi
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: model.ckpt not found at $CHECKPOINT_PATH." >&2
    echo "       Run: HF_TOKEN=hf_xxx bash $REPO_DIR/sam_3d_body/01_download_models.sh" >&2
    exit 1
fi
if ! python -c "import sam_3d_body, cv2" 2>/dev/null; then
    echo "ERROR: sam_3d_body or cv2 not importable in env '$CONDA_ENV'." >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

echo "=== [01] Pick front-facing image + segment person ==="
echo "  代码:      $SAM3D_DIR"
echo "  ckpt:      $CHECKPOINT_PATH"
echo "  输入:      $INPUT_DIR/image"
echo "  输出:      $OUTPUT_DIR/segmented_image.png"
echo "  detector:  $DETECTOR_NAME"
echo "  segmentor: ${SEGMENTOR_NAME:-none} ${SEGMENTOR_PATH:+($SEGMENTOR_PATH)}"
echo "  fov:       ${FOV_NAME:-none}"
echo "  frontal_sign: $FRONTAL_SIGN  (set -1 if it picks the back-facing image)"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES"
else
    echo "  GPU:       default cuda:0  [set GPU=N to pin]"
fi

cd "$SAM3D_DIR"
python "$SCRIPT_DIR/pick_and_segment.py"

echo "=== [01] Done. Segmented image: $OUTPUT_DIR/segmented_image.png ==="
