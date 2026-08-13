#!/usr/bin/env bash
# 01a_segment_all.sh — segment the person in EVERY image of an orbit shoot onto a
# white background, using wan22_rotate's ViTDet + SAM2 machinery (sam-3d-body).
#
# Higher-quality alternative to 01_segment_all.sh (which uses rembg + SAM2 in the
# pdfgs env). This one REUSES the wan22_rotate conda env — it already has
# detectron2 + sam-3d-body + SAM2 + ViTDet weights, so no separate env to build
# for step 01. Steps 02/03 still use the pdfgs env (Pi3 / PDF-GS / DINOv3); only
# step 01 borrows the wan22 env.
#
# It is a near-copy of wan22_rotate/01c_pick_and_segment.sh, but instead of
# picking ONE front-facing image it segments ALL images (PDF-GS needs the full
# multi-view set to triangulate the body). Done by setting SEGMENT_ALL=1 on the
# shared pick_and_segment_mediapipe.py (which then loops the same detect+segment
# code over every image instead of MediaPipe-scoring then segmenting one).
#
# Output: $SEGMENTED_DIR/<rel>.png (white-bg person, preserves input rel subpath).
#         Same layout as 01_segment_all.sh, so step 02 consumes either unchanged.
#
# REQUIRED:  INPUT_DIR=/path/to/subject_folder  (must contain image/ subfolder)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Reuse wan22_rotate's env: it activates the `wan22_rotate` conda env (detectron2
# + sam-3d-body + SAM2) and exports SAM3D_DIR / DETECTOR_PATH / SEGMENTOR_PATH /
# MP_MODEL_PATH / WAN_MODEL_DIR. pdfgs_human/_env.sh would activate the pdfgs env
# (no detectron2), so we source wan22's instead.
# Capture any user RESULTS_DIR before wan22 _env.sh defaults it to its own tree.
_PDFGS_RESULTS_DIR="${RESULTS_DIR:-}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../wan22_rotate/_env.sh"
# Point output at the pdfgs_human results tree (not wan22_rotate_results).
RESULTS_DIR="${_PDFGS_RESULTS_DIR:-$REPO_DIR/../pdfgs_human_results}"
export RESULTS_DIR

# --- sam_3d_body detector + segmentor (paths come from wan22 _env.sh) ---
export DETECTOR_NAME="${DETECTOR_NAME:-vitdet}"
export DETECTOR_PATH="${DETECTOR_PATH:-$WAN_MODEL_DIR/ViTDet}"
export SEGMENTOR_NAME="${SEGMENTOR_NAME:-sam2}"
export SEGMENTOR_PATH="${SEGMENTOR_PATH:-}"
export SAM3D_DIR
export DEVICE="${DEVICE:-cuda}"
export WHITE_BG="${WHITE_BG:-1}"
export PADDING="${PADDING:-0.1}"

# --- SEGMENT_ALL mode on the shared pick_and_segment_mediapipe.py ---
export SEGMENT_ALL=1
export SKIP_SEGMENTATION=0

# --- I/O ---
export INPUT_DIR="${INPUT_DIR:-}"
if [ -z "$INPUT_DIR" ]; then
    echo "❌ ERROR: set INPUT_DIR=/path/to/subject_folder (with image/ subfolder or raw images)" >&2
    echo "   e.g. INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_xxx \\" >&2
    echo "        bash pdfgs_human/01a_segment_all.sh" >&2
    exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ ERROR: INPUT_DIR not found: $INPUT_DIR" >&2; exit 1
fi
if [ ! -d "$INPUT_DIR/image" ]; then
    echo "⚠️ WARNING: no 'image' subfolder in $INPUT_DIR; will use $INPUT_DIR itself" >&2
fi

export SEGMENTED_DIR="${SEGMENTED_DIR:-$RESULTS_DIR/segmented_frames}"
export OUTPUT_DIR="$SEGMENTED_DIR"

# --- checks ---
if [ ! -d "$SAM3D_DIR" ]; then
    echo "❌ ERROR: sam-3d-body code not found at $SAM3D_DIR." >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/../wan22_rotate/00_setup_env.sh" >&2
    exit 1
fi
# ViTDet/detectron2 are the real deps here (MediaPipe is only for the frontal pick,
# which SEGMENT_ALL skips — so we don't require it).
if ! python -c "import torch, detectron2" 2>/dev/null; then
    echo "❌ ERROR: torch/detectron2 not importable in env '$CONDA_ENV'." >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/../wan22_rotate/00_setup_env.sh" >&2
    exit 1
fi

echo "🚀 [01a] segment ALL images (white-bg person, ViTDet + SAM2 — wan22 env)"
echo "  📁 代码:      $SAM3D_DIR"
echo "  🤖 env:       $CONDA_ENV (reused from wan22_rotate)"
echo "  📂 输入:      $INPUT_DIR/image"
echo "  💾 输出:      $SEGMENTED_DIR/<rel>.png"
echo "  🔍 detector:  $DETECTOR_NAME"
echo "  ✂️  segmentor: ${SEGMENTOR_NAME:-none} ${SEGMENTOR_PATH:+($SEGMENTOR_PATH)}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU:       physical $CUDA_VISIBLE_DEVICES"
else
    echo "  🎮 GPU:       default cuda:0  [set GPU=N to pin]"
fi

# Run from sam_3d_body dir so its tools.* imports resolve (matches 01c).
cd "$SAM3D_DIR"
python "$SCRIPT_DIR/../wan22_rotate/pick_and_segment_mediapipe.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED. Segmentation did not complete." >&2
    exit 1
fi

echo ""
echo "🎉 [01a] Done. Next:"
echo "  INPUT=$SEGMENTED_DIR bash $SCRIPT_DIR/02_pi3_colmap.sh"
