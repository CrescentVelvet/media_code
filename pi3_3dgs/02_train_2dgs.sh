#!/usr/bin/env bash
# 02_train_2dgs.sh — train 2D Gaussian Splatting (SIGGRAPH 2024) on the COLMAP
# scene produced by 01_pi3_recon.sh. The 2DGS train.py uses the same source
# path convention as vanilla 3DGS:
#   <source>/images/         (training images, copied by 01)
#   <source>/sparse/0/*.txt  (COLMAP cameras/images/points3D, written by 01)
#
# 2DGS initializes gaussians from COLMAP's points3D (here: Pi3's dense cloud,
# confidence-filtered + voxel-downsampled), then optimizes them against the
# training images. It adds two regularizations absent from vanilla 3DGS:
#   --lambda_normal       normal-consistency loss (geometry smoothness)
#   --lambda_distortion   depth-distortion loss (multi-view depth agreement)
# Both default to non-zero in 2DGS; tuning them helps on AI-generated video.
#
# Env (all optional, defaults shown):
#   SOURCE_DIR=             # required if --source not given (env: $RESULTS_DIR/source)
#   MODEL_DIR=              # trained-model output (env: $RESULTS_DIR/model)
#   ITERATIONS=30000        # 2DGS paper default; bump down to 7000 for a quick demo
#   PORT=6012               # viser viewer port (set to 0 to disable the viewer)
#   WHITE_BG=0              # 1 = use white bg (useful if your video has white bg
#                           #     from wan22_rotate segmentation); 0 = black bg
#   EXTRA_ARGS=""           # extra args forwarded to train.py
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/source}"
MODEL_DIR="${MODEL_DIR:-$RESULTS_DIR/model}"
ITERATIONS="${ITERATIONS:-30000}"
PORT="${PORT:-0}"   # 0 = no viewer (we redirect output to a log file)
WHITE_BG="${WHITE_BG:-0}"

if [ ! -d "$SOURCE_DIR/images" ] || [ ! -d "$SOURCE_DIR/sparse/0" ]; then
    echo "ERROR: source dir not ready: $SOURCE_DIR" >&2
    echo "       Expected: $SOURCE_DIR/images/ + $SOURCE_DIR/sparse/0/{cameras,images,points3D}.txt" >&2
    echo "       Run 01_pi3_recon.sh first:  INPUT=/path/to/video.mp4 bash $SCRIPT_DIR/01_pi3_recon.sh" >&2
    exit 1
fi
if [ ! -f "$GS2D_DIR/train.py" ]; then
    echo "ERROR: 2DGS train.py not found at $GS2D_DIR/train.py" >&2
    echo "       Run INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh first" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"

# Verify the two CUDA rasterizer exts are importable (they're required for train).
python -c "import simple_knn, diff_surfel_rasterization" 2>/dev/null || {
    echo "ERROR: 2DGS CUDA extensions not importable (simple_knn / diff_surfel_rasterization)." >&2
    echo "       Run: BUILD_CUDA=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
}

TRAIN_FLAGS=(
    -s "$SOURCE_DIR"
    -m "$MODEL_DIR"
    --iterations "$ITERATIONS"
    --port "$PORT"
)
# White-background switch (useful when input is wan22_rotate's white-bg segmented video).
if [ "$WHITE_BG" = "1" ]; then
    TRAIN_FLAGS+=(--white_background)
fi
# Forward extra args (e.g., --lambda_normal 0.05 --lambda_distortion 100).
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    TRAIN_FLAGS+=($EXTRA_ARGS)
fi

echo "=== [02] Train 2D Gaussian Splatting ==="
echo "  source:    $SOURCE_DIR"
echo "  model:     $MODEL_DIR"
echo "  iterations: $ITERATIONS"
echo "  white_bg:  $WHITE_BG"
echo "  extra:     ${EXTRA_ARGS:-(none)}"
echo "  train.py:  $GS2D_DIR/train.py"
echo "  cmd:       python train.py ${TRAIN_FLAGS[*]}"
echo ""

# Train inside the 2DGS repo so its relative imports (scene/, gaussian_renderer/,
# arguments/) resolve. All paths we pass are absolute.
( cd "$GS2D_DIR" && python train.py "${TRAIN_FLAGS[@]}" )

echo ""
echo "=== [02] Done. Trained 2DGS at $MODEL_DIR ==="
echo "    Gaussians:   $MODEL_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
echo "    cfg_args:    $MODEL_DIR/cfg_args"
echo "    Next:        MODEL_DIR=$MODEL_DIR SOURCE_DIR=$SOURCE_DIR bash $SCRIPT_DIR/03_render_2dgs.sh"
