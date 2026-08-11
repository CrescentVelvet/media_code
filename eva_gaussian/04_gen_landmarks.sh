#!/usr/bin/env bash
# 04_gen_landmarks.sh — (optional) generate landmark.json for anchor loss.
#
# EVA-Gaussian supports an optional "anchor loss" that uses face + hand
# landmarks to regularize Gaussian positions. This requires extra packages
# (mmpose 0.x, mmdet 2.x, mmcv, xtcocotools, face_recognition, mediapipe)
# that are NOT installed by 00_setup_env.sh (they conflict with torch 2.5).
#
# If you don't need anchor loss, skip this step entirely — just run
# 02_pretrain_depth.sh + 03_train.sh with ANCHOR unset (default 0).
#
# Env:
#   DATA_ROOT=/path/to/dataset   (required)
#   GPU=0
#   INSTALL_LANDMARK_DEPS=1      (install mmpose/mmdet/etc first time)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

DATA_ROOT="${DATA_ROOT:?❌ DATA_ROOT env var required}"

if [ ! -f "$EVA_DIR/landmark_generation.py" ]; then
    echo "❌ ERROR: $EVA_DIR/landmark_generation.py not found." >&2
    echo "       Run INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh first" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Install landmark deps (mmpose 0.x + mmdet 2.x + mmcv + extras).
#    These are OLD APIs (init_pose_model, inference_top_down_pose_model) that
#    only exist in mmpose <1.0. They may not build against torch 2.5 — if so,
#    create a SEPARATE conda env with torch 1.13+cu117 just for this step.
# ---------------------------------------------------------------------------
if [ "${INSTALL_LANDMARK_DEPS:-0}" = "1" ]; then
    echo "📦 [04] installing landmark deps (mmpose 0.x, mmdet 2.x, mmcv, extras)"
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)
    _proxy="${https_proxy:-${http_proxy:-}}"
    [ -n "$_proxy" ] && PIP_FLAGS+=(--proxy "$_proxy")

    # mmcv-full (prebuilt for specific torch+cuda; may need --no-build-isolation).
    pip install "${PIP_FLAGS[@]}" mmcv-full==1.7.1 mmdet==2.28.0 mmpose==0.29.0 \
        xtcocotools face_recognition mediapipe
    if [ $? -ne 0 ]; then
        echo "❌ FAILED: landmark deps install" >&2
        echo "   These old mmpose/mmdet versions may not build against torch 2.5." >&2
        echo "   Try a separate env: conda create -n eva_landmark python=3.8 -y" >&2
        echo "     && conda activate eva_landmark" >&2
        echo "     && pip install torch==1.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117" >&2
        echo "     && pip install mmcv-full==1.7.1 mmdet==2.28.0 mmpose==0.29.0 xtcocotools face_recognition mediapipe" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 2. Patch landmark_generation.py: set dataset_path + fix the missing colon
#    on the for-loop line (official code has a syntax error).
# ---------------------------------------------------------------------------
echo "✂️ [04] patching landmark_generation.py (dataset_path + syntax fix)"
LANDMARK_SCRIPT="$EVA_DIR/landmark_generation.py"

# Set dataset_path (the line: dataset_path = 'path to your dataset').
sed -i "s|^dataset_path = .*|dataset_path = '$DATA_ROOT'|" "$LANDMARK_SCRIPT"

# Fix the missing colon: "for file in ['0.jpg', '1.jpg']" -> add ":".
sed -i "s|for file in \['0.jpg', '1.jpg'\]$|for file in ['0.jpg', '1.jpg']:|" "$LANDMARK_SCRIPT"

echo "  ✅ patched dataset_path -> $DATA_ROOT"

# ---------------------------------------------------------------------------
# 3. Run landmark_generation.py.
# ---------------------------------------------------------------------------
echo "🚀 [04] generating landmarks (face + hand detection)"
echo "  📁 DATA_ROOT: $DATA_ROOT"
echo "  💾 output:    $DATA_ROOT/landmark.json"
echo ""

cd "$EVA_DIR"
python landmark_generation.py
if [ $? -ne 0 ]; then
    echo "❌ FAILED: landmark_generation.py" >&2
    exit 1
fi

if [ -f "$DATA_ROOT/landmark.json" ]; then
    echo ""
    echo "🎉 [04] Done. landmark.json generated at $DATA_ROOT/landmark.json"
    echo "    Now run training with ANCHOR=1:"
    echo "      GPU=0 ANCHOR=1 DATA_ROOT=$DATA_ROOT bash $SCRIPT_DIR/run_all.sh"
else
    echo "❌ landmark.json not generated — check errors above" >&2
    exit 1
fi
