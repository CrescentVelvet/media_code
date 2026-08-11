#!/usr/bin/env bash
# 01c_pick_and_segment.sh — pick the front-facing image via MediaPipe Face Mesh
# and segment the person (background -> white).
#
# Lighter alternative to 01 (SAM 3D Body 3D pose) and 01b (ViTDet + largest area):
#   - Only needs MediaPipe (pip install mediapipe). No 1GB+ GATED weights.
#   - Runs on CPU by default (DEVICE=cpu); GPU not required for face mesh.
#   - Front-facing => nose centered between eyes + widest eye distance.
#   - Back-facing => no face detected (auto-excluded).
#
# Segmentation reuses 01b's ViTDet + SAM2 (optional). Skip it with
# SKIP_SEGMENTATION=1 if you only want to know which frame is frontal.
#
# REQUIRED:  INPUT_DIR=/path/to/subject_folder  (must contain image/ subfolder)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# --- MediaPipe face mesh config ---
export MP_MIN_CONFIDENCE="${MP_MIN_CONFIDENCE:-0.5}"
export MP_MODEL_PATH="${MP_MODEL_PATH:-$WAN_MODEL_DIR/mediapipe/face_landmarker.task}"
export DEVICE="${DEVICE:-cuda}"  # mediapipe always runs on CPU; this is for detector/segmentor

# --- optional person detector + segmentor (reused from 01b) ---
export DETECTOR_NAME="${DETECTOR_NAME:-vitdet}"
export DETECTOR_PATH="${DETECTOR_PATH:-$WAN_MODEL_DIR/ViTDet}"
export SEGMENTOR_NAME="${SEGMENTOR_NAME:-sam2}"
export SEGMENTOR_PATH="${SEGMENTOR_PATH:-}"
export SAM3D_DIR
export SKIP_SEGMENTATION="${SKIP_SEGMENTATION:-0}"

export BBOX_THRESH="${BBOX_THRESH:-0.8}"
export WHITE_BG="${WHITE_BG:-1}"
export PADDING="${PADDING:-0.1}"

# --- I/O ---
export INPUT_DIR="${INPUT_DIR:-}"
if [ -z "$INPUT_DIR" ]; then
    echo "❌ ERROR: set INPUT_DIR=/path/to/subject_folder (must contain image/ subfolder)" >&2
    echo "  e.g. INPUT_DIR=/data/subject_001 bash $0" >&2
    exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ ERROR: INPUT_DIR not found: $INPUT_DIR" >&2; exit 1
fi
if [ ! -d "$INPUT_DIR/image" ]; then
    echo "⚠️ WARNING: no 'image' subfolder in $INPUT_DIR; will use $INPUT_DIR itself" >&2
fi

export OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR}"

# --- checks ---
if ! python -c "import mediapipe" 2>/dev/null; then
    echo "❌ ERROR: mediapipe not importable in env '$CONDA_ENV'." >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

# sam_3d_body code is needed for the detector/segmentor (unless skipping segmentation).
if [ "$SKIP_SEGMENTATION" != "1" ]; then
    if [ ! -d "$SAM3D_DIR" ]; then
        echo "⚠️ WARNING: sam-3d-body code not found at $SAM3D_DIR." >&2
        echo "         Segmentation will fall back to bbox or be skipped." >&2
    fi
fi

echo "🚀 [01c] Pick front-facing image via MediaPipe Face Mesh"
echo "  📁 代码:      $SAM3D_DIR"
echo "  📂 输入:      $INPUT_DIR/image"
echo "  💾 输出:      $OUTPUT_DIR/segmented_image.png"
echo "  🤖 method:   mediapipe face mesh (CPU, no extra weights)"
echo "  🔍 detector:  ${DETECTOR_NAME:-none}"
echo "  ✂️  segmentor: ${SEGMENTOR_NAME:-none} ${SEGMENTOR_PATH:+($SEGMENTOR_PATH)}"
if [ "$SKIP_SEGMENTATION" = "1" ]; then
    echo "  ⏭️  SKIP_SEGMENTATION=1 (front-image pick only, no segmentation)"
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU:       physical $CUDA_VISIBLE_DEVICES"
else
    echo "  🎮 GPU:       default cuda:0  [set GPU=N to pin]"
fi

# Run from sam_3d_body dir so its tools.* imports resolve (matches 01/01b).
if [ -d "$SAM3D_DIR" ]; then
    cd "$SAM3D_DIR"
fi
python "$SCRIPT_DIR/pick_and_segment_mediapipe.py"

echo "🎉 [01c] Done. Segmented image: $OUTPUT_DIR/segmented_image.png"
