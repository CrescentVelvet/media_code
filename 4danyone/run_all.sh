#!/usr/bin/env bash
# run_all.sh — one-click 4DAnyone pipeline: setup env -> download models ->
# run inference (monocular video -> synchronized multi-view target videos).
#
# ⚠️  Inference needs ~43 GiB peak VRAM. Run on a >=48 GiB GPU on the server.
#
# Example (bundled pexels clip — fetch it first with EXAMPLE=1 in step 01):
#   GPU=0 VIDEO_PATH=../4DAnyone/data/source/pexels/2785536-uhd_2160_3840_25fps.mp4 \
#     bash 4danyone/run_all.sh
#
# Example (your own 9:16 portrait video, >=720p, >=121 frames):
#   GPU=0 VIDEO_PATH=/path/to/subject.mp4 VIEWS_PER_LAYER=24 \
#     bash 4danyone/run_all.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional proxy (must be set before git clone in 00).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

# Default input: the bundled pexels example (downloaded by 01 with EXAMPLE=1).
FDANYONE_DIR_DEFAULT="$REPO_DIR/../4DAnyone"
DEFAULT_VIDEO="$FDANYONE_DIR_DEFAULT/data/source/pexels/2785536-uhd_2160_3840_25fps.mp4"
VIDEO_PATH="${VIDEO_PATH:-$DEFAULT_VIDEO}"

echo "=== [run_all] 4DAnyone one-click pipeline ==="
echo "  conda env: ${CONDA_ENV:-4danyone}"
echo "  input:     $VIDEO_PATH"
echo "  views:     VIEWS_PER_LAYER=${VIEWS_PER_LAYER:-6}"
echo "  results:   ${RESULTS_DIR:-$REPO_DIR/../4danyone_results}"
echo ""

# 0. Setup: clone 4DAnyone + GVHMR submodule, create conda env, install deps.
echo "=== [0/2] setup env (clone + submodule + deps) ==="
INSTALL_DEPS="${INSTALL_DEPS:-1}" bash "$SCRIPT_DIR/00_setup_env.sh" || {
    echo "❌ 00_setup_env.sh failed; see log above." >&2
    exit 1
}

# 1. Download models from HF (+ SMPL-X if SMPLX_ARCHIVE is set).
echo ""
echo "=== [1/2] download models ==="
bash "$SCRIPT_DIR/01_download_models.sh" || {
    echo "❌ 01_download_models.sh failed; see log above." >&2
    exit 1
}

# 2. Inference.
echo ""
echo "=== [2/2] inference ==="
export VIDEO_PATH
bash "$SCRIPT_DIR/02_run_inference.sh" || {
    echo "❌ 02_run_inference.sh failed; see log above." >&2
    exit 1
}

echo ""
echo "=== [run_all] All steps finished. ==="
echo "  🎬 input:  $VIDEO_PATH"
echo "  💾 output: \${RESULTS_DIR:-$REPO_DIR/../4danyone_results}/fdanyone/<clip>/"
echo ""
echo "  Variants:"
echo "    6-view (lowest VRAM): VIEWS_PER_LAYER=6 bash $0  (default)"
echo "    24-view full orbit:  VIEWS_PER_LAYER=24 bash $0"
echo "    48-view 3 layers:    VIEWS_PER_LAYER=16 LAYER_PITCHES='[-10,15,35]' bash $0"
echo "    Frontal 180°:         VIEWS_PER_LAYER=8 START_YAW=-90 YAW_SPAN=180 bash $0"
