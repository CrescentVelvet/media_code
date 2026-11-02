#!/usr/bin/env bash
# 02_run_inference.sh —— 单图/批量图 -> 3DGS PLY
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

IMAGE="${IMAGE:-}"
MODE="${MODE:-synthetic}"          # synthetic=渲染图/生成图(裁切1.2) | real=实拍照片(裁切1.7)
OUT="${OUT:-$RESULTS_DIR}"
SEED="${SEED:-42}"
STEPS="${STEPS:-50}"
CFG_STRENGTH="${CFG_STRENGTH:-3.0}"
RESCALE_T="${RESCALE_T:-3.0}"
NUM_GAUSSIANS="${NUM_GAUSSIANS:-262144}"
DINOV3_DTYPE="${DINOV3_DTYPE:-bf16}"
OPTIMIZE="${OPTIMIZE:-1}"          # 1=torch.compile+cudnn attn+TF32（首次生成会编译，之后快）

if [ -z "$IMAGE" ]; then
    echo "❌ ERROR: 必须指定 IMAGE（单张图片或图片目录）" >&2
    echo "   例: IMAGE=$IDOL_REPO/demo/examples/fur-egg-chair.webp bash $0" >&2
    exit 1
fi
if [ ! -e "$IMAGE" ]; then
    echo "❌ ERROR: 输入不存在: $IMAGE" >&2
    exit 1
fi
if [ ! -d "$IDOL_MODELS" ]; then
    echo "❌ ERROR: 权重目录不存在: $IDOL_MODELS（先跑 01_download_models.sh）" >&2
    exit 1
fi

mkdir -p "$OUT"
EXTRA=()
[ "$OPTIMIZE" = "0" ] && EXTRA+=(--no_optimize)

echo "🚀 [02] Idol 单图转 3DGS"
echo "  🖼️ 输入:   $IMAGE"
echo "  💾 输出:   $OUT"
echo "  🎛️ 模式:   $MODE (crop_margin=$([ "$MODE" = "real" ] && echo 1.7 || echo 1.2))"
echo "  🔢 高斯数: $NUM_GAUSSIANS | steps=$STEPS cfg=$CFG_STRENGTH rescale_t=$RESCALE_T seed=$SEED"
echo "  🏋️ 权重:   $IDOL_MODELS"
echo "  🎮 GPU:    ${CUDA_VISIBLE_DEVICES:-all}"

START=$(date +%s)
python -m idol.sample \
    --image "$IMAGE" \
    --out "$OUT" \
    --mode "$MODE" \
    --seed "$SEED" \
    --steps "$STEPS" \
    --cfg_strength "$CFG_STRENGTH" \
    --rescale_t "$RESCALE_T" \
    --num_gaussians "$NUM_GAUSSIANS" \
    --weights "$IDOL_MODELS" \
    --dinov3_dtype "$DINOV3_DTYPE" \
    "${EXTRA[@]}"
RC=$?
END=$(date +%s)

if [ $RC -ne 0 ]; then
    echo "❌ 推理失败 (exit $RC)" >&2
    exit 1
fi

echo "⏱️ 总耗时 $((END-START))s"
echo "📂 产物:"
ls -lh "$OUT"/*.ply 2>/dev/null | tail -20
echo "🎉 [02] Done。PLY 可直接用任意 Gaussian Splat viewer 打开（如 https://antimatter15.com/splat/）"
