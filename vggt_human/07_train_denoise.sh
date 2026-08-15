#!/usr/bin/env bash
# 07_train_denoise.sh — 3DGS training on the face-enhanced + denoised augmented scene
# (original + denoised virtual cameras from step 04 + face-enhanced from step 05).
#
# Trains from scratch on the augmented COLMAP scene. The original training images
# provide the ground truth; the denoised novel-view images provide additional
# supervision in sparse regions (where 3DGS had artifacts before).
#
# ⚠️ Resuming from step 03's checkpoint: 3DGS saves chkpnt{N}.pth only when
#    --checkpoint_iterations is set. If you ran 03 with CHECKPOINT_ITERATIONS=30000,
#    you can resume here by pointing train.py at the old model dir:
#      MODEL_PATH=$RESULTS_DIR/model_3dgs LOADED_ITER=30000 bash 07_train_denoise.sh
#    Otherwise, training starts from scratch on the augmented scene (re-does
#    initial training, but the extra cameras usually improve the result).
#
# Prerequisites: step 04 (source_aug) + step 05 (source_aug_face) already run.
# If step 05 skipped, falls back to source_aug (step 04).
#
# Env (all optional, defaults shown):
#   RESULTS_DIR=                # output root
#   SOURCE_AUG_DIR=             # face-enhanced scene (default: source_aug_face; fallback: source_aug)
#   GAUSSIAN_DENOISE_DIR=        # model output (default: $RESULTS_DIR/model_3dgs_denoise)
#   ITERATIONS=30000             # training iterations
#   RES=                         # --resolution factor (UNSET = full-res)
#   WHITE_BG=0                   # 1 = white rasterizer bg
#   MODEL_PATH=                  # resume from this dir (empty = from scratch)
#   LOADED_ITER=                 # resume iteration (empty = from scratch)
#   SKIP_RENDER=0                # 1 = skip rendering
#   SKIP_METRICS=1               # 1 = skip PSNR/SSIM/LPIPS
#   TRAIN_EXTRA_ARGS=            # extra args for train.py
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SOURCE_AUG_DIR="${SOURCE_AUG_DIR:-$RESULTS_DIR/source_aug_face}"
# Fallback: if source_aug_face doesn't exist, use source_aug (step 04 only)
if [ ! -d "$SOURCE_AUG_DIR/images" ] && [ -d "$RESULTS_DIR/source_aug/images" ]; then
    SOURCE_AUG_DIR="$RESULTS_DIR/source_aug"
fi
GAUSSIAN_DENOISE_DIR="${GAUSSIAN_DENOISE_DIR:-$RESULTS_DIR/model_3dgs_denoise}"
ITERATIONS="${ITERATIONS:-30000}"
RES="${RES:-}"
WHITE_BG="${WHITE_BG:-0}"
SKIP_RENDER="${SKIP_RENDER:-0}"
SKIP_METRICS="${SKIP_METRICS:-1}"
MODEL_PATH="${MODEL_PATH:-}"
LOADED_ITER="${LOADED_ITER:-}"

echo "🏋️ [07] 3DGS training on face-enhanced + denoised scene"
echo "  🤖 3DGS:        $GS_DIR"
echo "  📂 source_aug:  $SOURCE_AUG_DIR"
echo "  💾 model:       $GAUSSIAN_DENOISE_DIR"
echo "  📐 iterations:  $ITERATIONS"
[ -n "$RES" ] && echo "  📐 resolution:  $RES"
echo "  🎨 white_bg:    $WHITE_BG"
if [ -n "$MODEL_PATH" ] && [ -n "$LOADED_ITER" ]; then
    echo "  🔄 resume:      from $MODEL_PATH (iter=$LOADED_ITER)"
fi
echo ""

# Sanity checks
if [ ! -f "$GS_DIR/train.py" ]; then
    echo "❌ ERROR: gaussian-splatting not found at $GS_DIR" >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi
if [ ! -d "$SOURCE_AUG_DIR/images" ] || [ ! -d "$SOURCE_AUG_DIR/sparse/0" ]; then
    echo "❌ ERROR: augmented COLMAP scene not ready: $SOURCE_AUG_DIR" >&2
    echo "       Run step 05+06 first" >&2
    exit 1
fi
if ! python -c "import diff_gaussian_rasterization, simple_knn" 2>/dev/null; then
    echo "❌ ERROR: 3DGS CUDA extensions not importable." >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

mkdir -p "$GAUSSIAN_DENOISE_DIR"

# ── Training ────────────────────────────────────────────────────────────────
echo "🏋️ 3DGS training ($ITERATIONS iterations on augmented scene)"
TRAIN_FLAGS=(
    -s "$SOURCE_AUG_DIR"
    -m "$GAUSSIAN_DENOISE_DIR"
    --iterations "$ITERATIONS"
    --port 0
    --disable_viewer
)
[ "$WHITE_BG" = "1" ] && TRAIN_FLAGS+=(--white_background)
[ -n "$RES" ] && TRAIN_FLAGS+=(--resolution "$RES")
# Resume from checkpoint (if specified)
if [ -n "$MODEL_PATH" ] && [ -n "$LOADED_ITER" ]; then
    TRAIN_FLAGS+=(--model_path "$MODEL_PATH" --iteration "$LOADED_ITER")
fi
if [ -n "${TRAIN_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    TRAIN_FLAGS+=($TRAIN_EXTRA_ARGS)
fi

# Use wrapper with pose optimization, or standard train.py
if [ "${POSE_ADJUST:-0}" = "1" ] || [ "${POSE_REFINE:-0}" = "1" ]; then
    echo "🏋️ using train_pose.py (POSE_ADJUST=$POSE_ADJUST POSE_REFINE=$POSE_REFINE)"
    export SOURCE_DIR="$SOURCE_AUG_DIR" GAUSSIAN_DIR="$GAUSSIAN_DENOISE_DIR" ITERATIONS SH_DEGREE WHITE_BG RES DEVICE
    export POSE_ADJUST POSE_REFINE REFINE_INTRINSIC POSE_REFINE_WEIGHT
    export POSE_REFINE_LR_Q POSE_REFINE_LR_T POSE_REFINE_LR_I GRAVITY_PRIOR
    ( cd "$GS_DIR" && python "$SCRIPT_DIR/train_pose.py" )
else
    ( cd "$GS_DIR" && python train.py "${TRAIN_FLAGS[@]}" )
fi
if [ $? -ne 0 ]; then
    echo "❌ FAILED. 3DGS training did not complete." >&2
    exit 1
fi
echo "✅ training done"
echo "  🏋️ gaussians: $GAUSSIAN_DENOISE_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
echo ""

# ── Rendering ───────────────────────────────────────────────────────────────
if [ "$SKIP_RENDER" = "1" ]; then
    echo "⏭️ skip rendering (SKIP_RENDER=1)"
else
    echo "🖼️ rendering (train views)"
    RENDER_FLAGS=(
        -s "$SOURCE_AUG_DIR"
        -m "$GAUSSIAN_DENOISE_DIR"
        --iteration "$ITERATIONS"
    )
    [ "$WHITE_BG" = "1" ] && RENDER_FLAGS+=(--white_background)
    ( cd "$GS_DIR" && python render.py "${RENDER_FLAGS[@]}" )
    if [ $? -ne 0 ]; then
        echo "⚠️ render.py failed (non-fatal)" >&2
    else
        echo "✅ rendering done"
        echo "  🖼️ renders: $GAUSSIAN_DENOISE_DIR/train/ours_$ITERATIONS/renders/*.png"
        echo "  🖼️ gt:      $GAUSSIAN_DENOISE_DIR/train/ours_$ITERATIONS/gt/*.png"
    fi
fi

# ── Metrics ─────────────────────────────────────────────────────────────────
if [ "$SKIP_METRICS" = "1" ]; then
    echo "⏭️ skip metrics (SKIP_METRICS=1)"
else
    echo "📊 metrics (PSNR / SSIM / LPIPS)"
    ( cd "$GS_DIR" && python metrics.py -m "$GAUSSIAN_DENOISE_DIR" )
    if [ $? -ne 0 ]; then
        echo "⚠️ metrics.py failed (non-fatal)" >&2
    fi
fi

echo ""
echo "🎉 [07] Done. 3DGS training complete."
echo "  🏋️ Gaussians: $GAUSSIAN_DENOISE_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
echo ""
echo "  Compare with step 03 (no denoise):"
echo "    $RESULTS_DIR/model_3dgs/point_cloud/iteration_30000/point_cloud.ply"
echo ""
echo "  Inspect: https://playcanvas.com/supersplat/editor (drag .ply)"
