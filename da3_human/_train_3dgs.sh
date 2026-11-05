# _train_3dgs.sh — 原版 3DGS 训练公共段（被 03/04 source，不直接执行）
# 前置环境变量：SOURCE_DIR（COLMAP 场景）、GAUSSIAN_DIR（输出）、ITERATIONS/RES/WHITE_BG/SKIP_RENDER/SKIP_METRICS
# 依赖 vggt_human env（gaussian-splatting 官方仓 + 已编译 CUDA 扩展，见 vggt_human/04_train_3dgs.sh）

set -o pipefail

ITERATIONS="${ITERATIONS:-30000}"
RES="${RES:-}"
WHITE_BG="${WHITE_BG:-0}"
SKIP_RENDER="${SKIP_RENDER:-0}"
SKIP_METRICS="${SKIP_METRICS:-0}"
GS_CONDA_ENV="${GS_CONDA_ENV:-vggt_human}"

echo "🏋️ 3DGS training (official gaussian-splatting, env=$GS_CONDA_ENV)"
echo "  🤖 GS_DIR:  $GS_DIR"
echo "  📂 source:  $SOURCE_DIR"
echo "  💾 output:  $GAUSSIAN_DIR"
echo "  📐 iters:   $ITERATIONS  res: ${RES:-full}  white_bg: $WHITE_BG"

if [ ! -f "$GS_DIR/train.py" ]; then
    echo "❌ ERROR: gaussian-splatting not found at $GS_DIR (no train.py)" >&2
    exit 1
fi
if [ ! -d "$SOURCE_DIR/images" ] || [ ! -d "$SOURCE_DIR/sparse/0" ]; then
    echo "❌ ERROR: COLMAP scene not ready: $SOURCE_DIR（需要 images/ + sparse/0/）" >&2
    exit 1
fi

# 切到 vggt_human env（3DGS CUDA 扩展编译在该 env 下；da3_human env 只管 DA3 推理）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$GS_CONDA_ENV"
if ! python -c "import diff_gaussian_rasterization, simple_knn" 2>/dev/null; then
    echo "❌ ERROR: 3DGS CUDA extensions not importable in env $GS_CONDA_ENV" >&2
    echo "       参照 vggt_human/00a_setup_env.sh 编译" >&2
    exit 1
fi

mkdir -p "$GAUSSIAN_DIR"

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
if [ $? -ne 0 ]; then echo "❌ 3DGS training FAILED" >&2; exit 1; fi
echo "✅ training done"

if [ "$SKIP_RENDER" = "1" ]; then
    echo "⏭️ skip rendering (SKIP_RENDER=1)"
else
    echo "🖼️ rendering (train views)"
    RENDER_FLAGS=( -s "$SOURCE_DIR" -m "$GAUSSIAN_DIR" --iteration "$ITERATIONS" )
    [ "$WHITE_BG" = "1" ] && RENDER_FLAGS+=(--white_background)
    ( cd "$GS_DIR" && python render.py "${RENDER_FLAGS[@]}" )
    if [ $? -ne 0 ]; then
        echo "⚠️ render.py failed (non-fatal)" >&2
    else
        echo "✅ renders: $GAUSSIAN_DIR/train/ours_$ITERATIONS/renders/"
    fi
fi

if [ "$SKIP_METRICS" = "1" ]; then
    echo "⏭️ skip metrics (SKIP_METRICS=1)"
else
    echo "📊 metrics (PSNR/SSIM/LPIPS, in-sample train views)"
    ( cd "$GS_DIR" && python metrics.py -m "$GAUSSIAN_DIR" )
    if [ $? -ne 0 ]; then echo "⚠️ metrics.py failed (non-fatal)" >&2; fi
fi
