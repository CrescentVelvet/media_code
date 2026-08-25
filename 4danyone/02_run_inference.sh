#!/usr/bin/env bash
# 02_run_inference.sh — run 4DAnyone to turn a monocular video into synchronized
# multi-view target videos (for downstream 4DGS reconstruction).
#
# ⚠️  Needs ~43 GiB peak VRAM (6-view minimum). Run on a >=48 GiB GPU on the
#    server (A6000/A100/H100/H200). RTX 3090 (24 GiB) is NOT enough.
#
# Usage:
#   GPU=0 VIDEO_PATH=../../4DAnyone/data/source/pexels/2785536-uhd_2160_3840_25fps.mp4 \
#     bash 4danyone/02_run_inference.sh
#
#   # 6-view (lowest VRAM, ~43 GiB) — default here; bump for denser coverage.
#   GPU=0 VIDEO_PATH=... VIEWS_PER_LAYER=6 bash 4danyone/02_run_inference.sh
#   # 24-view full orbit (for 4DGS reconstruction)
#   GPU=0 VIDEO_PATH=... VIEWS_PER_LAYER=24 bash 4danyone/02_run_inference.sh
#   # 48-view, three pitch layers (free-viewpoint 4DGS)
#   GPU=0 VIDEO_PATH=... VIEWS_PER_LAYER=16 LAYER_PITCHES='[-10,15,35]' \
#     bash 4danyone/02_run_inference.sh
#
# Env:
#   VIDEO_PATH       (required) input video; >=720p, 9:16, >=121 frames.
#   VIEWS_PER_LAYER  (default 6) views per pitch layer; divisible by 4 or 6.
#   LAYER_PITCHES    (default '[15]') pitch per layer; JSON list, e.g. '[-10,15,35]'.
#   START_YAW        (default 0) first yaw in degrees; 0 faces the person.
#   YAW_SPAN         (default 360) yaw range per layer in degrees.
#   DATA_DIR         (default $RESULTS_DIR) root for GVHMR motion + 4DAnyone output.
#   DEVICE           (default cuda:0) CUDA device (respecting CUDA_VISIBLE_DEVICES).
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [02] 4DAnyone inference"
echo "  🎬 input:   ${VIDEO_PATH:-(unset)}"
echo "  📐 views:   VIEWS_PER_LAYER=${VIEWS_PER_LAYER:-6}  LAYER_PITCHES=${LAYER_PITCHES:-'[15]'}"
echo "  💾 data_dir: ${DATA_DIR:-$RESULTS_DIR}"
echo "  🏋️ model:    $MODEL_DIR"
echo "  🎮 device:   ${DEVICE:-cuda:0}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
echo ""

# Pre-checks
if [ -z "${VIDEO_PATH:-}" ]; then
    echo "❌ ERROR: VIDEO_PATH not set" >&2
    echo "   Example: GPU=0 VIDEO_PATH=\$FDANYONE_DIR/data/source/pexels/2785536-uhd_2160_3840_25fps.mp4 bash $0" >&2
    exit 1
fi
if [ ! -f "$VIDEO_PATH" ]; then
    echo "❌ ERROR: input video not found: $VIDEO_PATH" >&2
    exit 1
fi
if [ ! -f "$FDANYONE_DIR/inference.py" ]; then
    echo "❌ ERROR: 4DAnyone repo missing at $FDANYONE_DIR" >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi
if [ ! -f "$MODEL_DIR/4danyone/model.safetensors" ]; then
    echo "❌ ERROR: 4DAnyone checkpoint missing at $MODEL_DIR/4danyone/model.safetensors" >&2
    echo "       Run: bash $SCRIPT_DIR/01_download_models.sh" >&2
    exit 1
fi
if [ ! -f "$SMPLX_PATH" ]; then
    echo "❌ ERROR: SMPL-X not installed at $SMPLX_PATH" >&2
    echo "       Register at https://smpl-x.is.tue.mpg.de/, download models_smplx_v1_1.zip, then:" >&2
    echo "         SMPLX_ARCHIVE=/path/to/models_smplx_v1_1.zip bash $SCRIPT_DIR/01_download_models.sh" >&2
    exit 1
fi

# expandable_segments reduces fragmentation on the 43 GiB allocation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$FDANYONE_DIR"
python inference.py \
    --video_path "$VIDEO_PATH" \
    --views_per_layer "${VIEWS_PER_LAYER:-6}" \
    --layer_pitches "${LAYER_PITCHES:-[15]}" \
    --start_yaw "${START_YAW:-0}" \
    --yaw_span "${YAW_SPAN:-360}" \
    --data_dir "${DATA_DIR:-$RESULTS_DIR}" \
    --model_dir "$MODEL_DIR" \
    --gvhmr_root "$GVHMR_DIR" \
    --device "${DEVICE:-cuda:0}"

if [ $? -ne 0 ]; then
    echo "❌ inference failed" >&2
    exit 1
fi

echo "🎬 [02] Done. Output layout (under \${DATA_DIR:-$RESULTS_DIR}):"
echo "    gvhmr/results/<clip>/          # reusable motion-recovery result"
echo "    fdanyone/<clip>/"
echo "      ├── metadata.json            # run settings, timings, resources"
echo "      ├── cameras.json             # the final N-camera rig"
echo "      ├── skeletons/00.mp4 ...     # per-view skeleton clips"
echo "      └── videos/"
echo "          ├── sparse/              # 6/24-view RCP proposals"
echo "          └── dense/00.mp4 ...     # generated target views"
echo ""
echo "🎉 4DAnyone inference complete. Feed dense/ + cameras.json to a 4DGS method."
