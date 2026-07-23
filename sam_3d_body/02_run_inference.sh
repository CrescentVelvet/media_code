#!/usr/bin/env bash
# 02_run_inference.sh — batch SAM 3D Body human mesh recovery over a folder of
# images. Calls run_inference.py (a thin wrapper over SAM3DBodyEstimator that
# mirrors the official demo.py: ViTDet detector + MoGe2 FOV by default, SAM2
# segmentor only when a path is given) but loads the pipeline ONCE, walks the
# input folder recursively, and prints model-load + per-image timing + a
# summary. Also saves per-person PLY meshes and a per-image npz.
#
# For each image in INPUT_DIR (walked recursively), produces:
#   $OUTPUT_DIR/result/<rel>.jpg                 (rendered overlay)
#   $OUTPUT_DIR/mesh/<rel stem>_mesh_<pid>.ply   (per-person 3D mesh)
#   $OUTPUT_DIR/npz/<rel>.npz                    (per-person numeric outputs)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SAM3D_DIR="${SAM3D_DIR:-$REPO_DIR/../sam-3d-body}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/sam-3d-body}"

# Default to the DINOv3-H+ checkpoint (recommended). Override HF_REPO_ID /
# CKPT_DIR to use the ViT-H backbone (facebook/sam-3d-body-vith).
HF_REPO_ID="${HF_REPO_ID:-facebook/sam-3d-body-dinov3}"
CKPT_DIR="${CKPT_DIR:-$MODEL_DIR/$(basename "$HF_REPO_ID")}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CKPT_DIR/model.ckpt}"
MHR_PATH="${MHR_PATH:-$CKPT_DIR/assets/mhr_model.pt}"
FOV_PATH="${FOV_PATH:-$MODEL_DIR/moge-2-vitl-normal}"

INPUT_DIR="${INPUT_DIR:-$SAM3D_DIR/notebook/images}"
INPUT_NAME="$(basename "$INPUT_DIR")"
OUTPUT_DIR="${OUTPUT_DIR:-$SAM3D_DIR/results/$INPUT_NAME}"

# --- inference params (all overridable via env) ---
DEVICE="${DEVICE:-cuda}"
DETECTOR_NAME="${DETECTOR_NAME:-vitdet}"          # vitdet (default) | sam3 | "" to disable
DETECTOR_PATH="${DETECTOR_PATH:-}"                # ViTDet auto-downloads; set for offline
SEGMENTOR_NAME="${SEGMENTOR_NAME:-sam2}"          # sam2 (needs SEGMENTOR_PATH) | sam3 | "" to disable
SEGMENTOR_PATH="${SEGMENTOR_PATH:-}"              # sam2 repo dir w/ checkpoints/ + configs/
FOV_NAME="${FOV_NAME:-moge2}"                     # moge2 (default) | "" to disable (uses default FOV)
BBOX_THRESH="${BBOX_THRESH:-0.8}"
USE_MASK="${USE_MASK:-0}"                          # 1 = mask-conditioned (needs a segmentor)
INFERENCE_TYPE="${INFERENCE_TYPE:-full}"          # full | body | hand
SAVE_NPZ="${SAVE_NPZ:-1}"

echo "=== [02] SAM 3D Body batch inference ==="
echo "  代码路径:   $SAM3D_DIR"
echo "  模型ckpt:   $CHECKPOINT_PATH"
echo "  mhr模型:    $MHR_PATH"
echo "  输入图像:   $INPUT_DIR"
echo "  输出目录:   $OUTPUT_DIR  (result/ + mesh/ + npz/)"
echo "  组件:       detector=$DETECTOR_NAME segmentor=$SEGMENTOR_NAME fov=$FOV_NAME"
echo "  参数:       bbox_thr=$BBOX_THRESH use_mask=$USE_MASK inference=$INFERENCE_TYPE"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:        physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change]"
else
    echo "  GPU:        default cuda:0 (= first visible)  [set GPU=N to pin a card]"
fi

# --- checks ---
if [ ! -d "$SAM3D_DIR" ]; then
    echo "ERROR: SAM 3D Body code dir not found at $SAM3D_DIR. Run run_all.sh first." >&2; exit 1
fi
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: model.ckpt not found at $CHECKPOINT_PATH. Run 01_download_models.sh first." >&2; exit 1
fi
if [ ! -f "$MHR_PATH" ]; then
    echo "ERROR: assets/mhr_model.pt not found at $MHR_PATH. Run 01_download_models.sh first." >&2; exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: input image dir not found: $INPUT_DIR" >&2; exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Run from the official repo dir so any relative file lookups + pyrootutils
# resolve correctly (run_inference.py also puts SAM3D_DIR on sys.path).
export SAM3D_DIR CHECKPOINT_PATH MHR_PATH DEVICE
export DETECTOR_NAME DETECTOR_PATH SEGMENTOR_NAME SEGMENTOR_PATH
export FOV_NAME FOV_PATH INPUT_DIR OUTPUT_DIR
export BBOX_THRESH USE_MASK INFERENCE_TYPE SAVE_NPZ

cd "$SAM3D_DIR"
python "$SCRIPT_DIR/run_inference.py"

echo "=== [02] Done. Results in: $OUTPUT_DIR/result ==="
