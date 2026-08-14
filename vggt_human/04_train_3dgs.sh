#!/usr/bin/env bash
# 04_train_3dgs.sh — Original 3DGS (graphdeco-inria/gaussian-splatting) training.
#
# Consumes the COLMAP scene from step 02. Trains gaussians via gradient descent
# on L1+SSIM loss, with adaptive density control (split/clone/prune). Renders
# all training views for reconstruction-vs-GT comparison.
#
# No mesh: 3DGS ships train.py / render.py / metrics.py (no extract_mesh).
# Output is the gaussian point cloud + rendered novel views + (opt) metrics.
#
# Prerequisites:
#   - INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh (first time; compiles CUDA exts)
#   - step 01 (predictions.npz) + step 02 (COLMAP source/) already run
#
# Env (all optional, defaults shown):
#   RESULTS_DIR=             # output root
#   SOURCE_DIR=              # COLMAP scene (default: $RESULTS_DIR/source)
#   GAUSSIAN_DIR=            # gaussians output (default: $RESULTS_DIR/model_3dgs)
#   ITERATIONS=30000         # total training iterations
#   RES=                     # --resolution factor (UNSET = full-res)
#   WHITE_BG=0               # 1 = white rasterizer bg
#   SKIP_RENDER=0            # 1 = skip rendering
#   SKIP_METRICS=1           # 1 = skip PSNR/SSIM/LPIPS
#   TRAIN_EXTRA_ARGS=        # extra args passed verbatim to train.py
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/source}"
GAUSSIAN_DIR="${GAUSSIAN_DIR:-$RESULTS_DIR/model_3dgs}"
ITERATIONS="${ITERATIONS:-30000}"
RES="${RES:-}"
WHITE_BG="${WHITE_BG:-0}"
SKIP_RENDER="${SKIP_RENDER:-0}"
SKIP_METRICS="${SKIP_METRICS:-1}"

echo "🏋️ [03] 3DGS training (original gaussian-splatting)"
echo "  🤖 3DGS:        $GS_DIR"
echo "  📂 source:      $SOURCE_DIR"
echo "  💾 model:       $GAUSSIAN_DIR"
echo "  📐 iterations:  $ITERATIONS"
[ -n "$RES" ] && echo "  📐 resolution:  $RES"
echo "  🎨 white_bg:    $WHITE_BG"
echo ""

# Sanity checks
if [ ! -f "$GS_DIR/train.py" ]; then
    echo "❌ ERROR: gaussian-splatting not found at $GS_DIR (no train.py)" >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi
if [ ! -d "$SOURCE_DIR/images" ] || [ ! -d "$SOURCE_DIR/sparse/0" ]; then
    echo "❌ ERROR: COLMAP scene not ready: $SOURCE_DIR" >&2
    echo "       Run step 03 first: bash $SCRIPT_DIR/03_npz_to_colmap.sh" >&2
    exit 1
fi
if ! python -c "import diff_gaussian_rasterization, simple_knn" 2>/dev/null; then
    echo "❌ ERROR: 3DGS CUDA extensions not importable." >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh (needs nvcc)" >&2
    exit 1
fi

mkdir -p "$GAUSSIAN_DIR"

# ── 1. Training ────────────────────────────────────────────────────────────
echo "🏋️ 3DGS training ($ITERATIONS iterations)"
TRAIN_FLAGS=(
    -s "$SOURCE_DIR"
    -m "$GAUSSIAN_DIR"
    --iterations "$ITERATIONS"
    --port 0
    --disable_viewer
)
[ "$WHITE_BG" = "1" ] && TRAIN_FLAGS+=(--white_background)
[ -n "$RES" ] && TRAIN_FLAGS+=(--resolution "$RES")
if [ -n "${TRAIN_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    TRAIN_FLAGS+=($TRAIN_EXTRA_ARGS)
fi
( cd "$GS_DIR" && python train.py "${TRAIN_FLAGS[@]}" )
if [ $? -ne 0 ]; then
    echo "❌ FAILED. 3DGS training did not complete." >&2
    exit 1
fi
echo "✅ training done"
echo "  🏋️ gaussians: $GAUSSIAN_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
echo ""

# ── 2. Rendering ───────────────────────────────────────────────────────────
if [ "$SKIP_RENDER" = "1" ]; then
    echo "⏭️ skip rendering (SKIP_RENDER=1)"
else
    echo "🖼️ rendering (train views)"
    RENDER_FLAGS=(
        -s "$SOURCE_DIR"
        -m "$GAUSSIAN_DIR"
        --iteration "$ITERATIONS"
    )
    [ "$WHITE_BG" = "1" ] && RENDER_FLAGS+=(--white_background)
    ( cd "$GS_DIR" && python render.py "${RENDER_FLAGS[@]}" )
    if [ $? -ne 0 ]; then
        echo "⚠️ render.py failed (non-fatal)" >&2
    else
        echo "✅ rendering done"
        echo "  🖼️ renders: $GAUSSIAN_DIR/train/ours_$ITERATIONS/renders/*.png"
        echo "  🖼️ gt:      $GAUSSIAN_DIR/train/ours_$ITERATIONS/gt/*.png"
    fi
fi

# ── 3. Metrics (optional) ──────────────────────────────────────────────────
if [ "$SKIP_METRICS" = "1" ]; then
    echo "⏭️ skip metrics (SKIP_METRICS=1)"
else
    echo "📊 metrics (PSNR / SSIM / LPIPS)"
    ( cd "$GS_DIR" && python metrics.py -m "$GAUSSIAN_DIR" )
    if [ $? -ne 0 ]; then
        echo "⚠️ metrics.py failed (non-fatal)" >&2
    fi
fi

echo ""
echo "🎉 [03] Done. 3DGS reconstruction complete."
echo "  🏋️ Gaussians: $GAUSSIAN_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
echo ""
echo "  Inspect: https://playcanvas.com/supersplat/editor (drag .ply)"
echo "  Or MeshLab: meshlab $GAUSSIAN_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
