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

# ── 0. 动态掩码 + 初始点云过滤（可选，03→04 之间） ────────────────────────────
# ENABLE_DYNAMIC_MASK=1   生成逐帧动态掩码（GroundingDINO+SAM2，缓存于 $MASKS_DIR）
# ENABLE_DYNAMIC_FILTER=1 多视角投票过滤 sparse/0/points3D.ply 中的动态点
# ⚠️ prompt 默认 "TV screen monitor"——刻意不含 person！本数据集主体是静态人物，
#    对 person 建掩码会把主体删掉（README_wsl.md「dynamic_mask 实测」）。
#    真动态场景请显式传 DYNAMIC_PROMPTS="person TV screen" 之类。
MASKS_DIR="${DYNAMIC_MASKS_DIR:-$RESULTS_DIR/dynamic_mask}"
if [ "${ENABLE_DYNAMIC_MASK:-0}" = "1" ]; then
    if [ -d "$MASKS_DIR" ] && [ "$(ls "$MASKS_DIR"/*.png 2>/dev/null | wc -l)" -gt 0 ] \
       && [ "${FORCE_DYNAMIC_MASK:-0}" != "1" ]; then
        echo "⏭️ dynamic masks already exist: $MASKS_DIR (FORCE_DYNAMIC_MASK=1 to regenerate)"
    else
        echo "🎭 generating dynamic masks (GroundingDINO + SAM2)"
        # shellcheck disable=SC2086
        python "$SCRIPT_DIR/dynamic_mask.py" \
            --images_dir "$SOURCE_DIR/images" \
            --output_dir "$RESULTS_DIR" \
            --prompts ${DYNAMIC_PROMPTS:-"TV screen monitor"}
        if [ $? -ne 0 ]; then
            echo "❌ FAILED. dynamic mask generation failed." >&2
            exit 1
        fi
    fi
fi
if [ "${ENABLE_DYNAMIC_FILTER:-0}" = "1" ]; then
    P3D="$SOURCE_DIR/sparse/0/points3D.ply"
    if [ ! -f "$P3D" ]; then
        echo "❌ ERROR: $P3D not found (step 03 must run first)" >&2
        exit 1
    fi
    if [ ! -f "$P3D.orig" ]; then
        cp "$P3D" "$P3D.orig"   # 首次过滤前备份原始点云
    fi
    echo "✂️ filtering dynamic points from points3D.ply (threshold=${DYNAMIC_THRESHOLD:-0.3}, dilate=${DYNAMIC_DILATE_PX:-5}px)"
    python "$SCRIPT_DIR/dynamic_filter.py" \
        --points "$P3D.orig" \
        --sparse_dir "$SOURCE_DIR/sparse/0" \
        --images_dir "$SOURCE_DIR/images" \
        --masks_dir "$MASKS_DIR" \
        --output "$P3D"
    if [ $? -ne 0 ]; then
        echo "❌ FAILED. dynamic point filtering failed." >&2
        exit 1
    fi
fi

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

# 默认走官方 3DGS train.py（干净基线）。增强逐个加回后在此加开关分支，
# 只有显式开启时才切到对应的 train_*.py（它们是官方 train.py 的副本 + 单一 hook）。
# ⚠️ 分支顺序=优先级。曾经这里先判 USE_DEPTH_NORMAL，而 _env.sh 把它的默认值
# 设成过 1，导致 `USE_NOISE_NEGATE=1` 时静默跑成了 depth-normal 训练。
# 现在：noise negating 优先，且所有增强开关默认都是 0（见 _env.sh）。
if [ "${USE_NOISE_NEGATE:-0}" = "1" ]; then
    echo "🎭 using train_noise_negate.py (DINOv2+MLP 动态区域抑制)"
    echo "   warmup=${NN_WARMUP_EPOCHS:-15} epochs, max_dynamic=${NN_MAX_DYNAMIC_RATIO:-0.2}"
    export USE_NOISE_NEGATE
    export DINO_MODEL_PATH
    export NN_FEATURE_SIZE NN_WARMUP_EPOCHS NN_MAX_DYNAMIC_RATIO NN_FIXED_THR \
           NN_DILATE_RADIUS NN_MLP_LR NN_SAMPLE_PIXELS
    ( cd "$GS_DIR" && python "$SCRIPT_DIR/train_noise_negate.py" "${TRAIN_FLAGS[@]}" )
elif [ "${USE_DEPTH_NORMAL:-0}" = "1" ]; then
    echo "🏋️ using train_depth_normal.py (depth-normal 约束, w=${DEPTH_NORMAL_WEIGHT:-0.05})"
    export USE_DEPTH_NORMAL
    export DEPTH_NORMAL_WEIGHT DEPTH_NORMAL_INTERVAL DEPTH_NORMAL_SAMPLE_POINTS
    ( cd "$GS_DIR" && python "$SCRIPT_DIR/train_depth_normal.py" "${TRAIN_FLAGS[@]}" )
elif [ "${USE_POSE_REFINE:-0}" = "1" ]; then
    echo "⚠️ train_pose_refine.py 当前不可用：diff_gaussian_rasterization 的 CUDA"
    echo "   rasterizer 不支持对 viewmatrix 求梯度，位姿参数拿不到 grad，精炼不生效。"
    echo "   改用官方 train.py（详见 README_wsl.md「pose_refine 现状」）"
    ( cd "$GS_DIR" && python train.py "${TRAIN_FLAGS[@]}" )
else
    echo "🏋️ using official train.py (baseline reconstruction)"
    ( cd "$GS_DIR" && python train.py "${TRAIN_FLAGS[@]}" )
fi
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
