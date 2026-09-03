#!/usr/bin/env bash
# 06b_face_finetune.sh — 方案一升级版：从 30k checkpoint/PLY 续训 finetune，
# 用 HYPIR 增强图做人脸区互补双监督（追加不替换，densification 冻结，LR×0.1）。
#
# 流程（对应设计定稿）：
#   前提: 04 训完 30k（$GAUSSIAN_DIR/point_cloud/iteration_30000/point_cloud.ply）
#         06_face_enhance.sh 已跑（$FACE_IMAGES_DIR，即 source_face/images）
#         人脸 mask 已生成（$FACE_MASKS_DIR；缺失时本脚本自动调 face_masks.py 补）
#   1. 校验 mask 与增强图一一对应（无 mask 的帧自动退化为纯官方监督）
#   2. train_face_finetune.py：ply 续训 + 互补 L1 + SSIM(原图)，--lr_scale 0.1
#   3. 产物: $GAUSSIAN_FACE_DIR/point_cloud/iteration_$FINETUNE_ITERATIONS/
#
# Env (all optional, defaults shown):
#   RESULTS_DIR=                 # 输出根
#   GAUSSIAN_DIR=                # 30k 模型目录 (默认 model_3dgs)
#   SOURCE_DIR=                  # COLMAP 场景 (默认 source；scenes 的 sparse/0 必须在)
#   FACE_IMAGES_DIR=             # HYPIR 增强图 (默认 source_face/images)
#   FACE_MASKS_DIR=              # 人脸 loss mask (默认 face_masks；无则自动生成)
#   FINETUNE_FROM=30000          # 起始迭代（ply 路径里的 iteration_N）
#   FINETUNE_ITERATIONS=35000    # finetune 终点迭代（默认 +5000）
#   GAUSSIAN_FACE_DIR=           # 输出模型目录 (默认 model_3dgs_face)
#   FACE_WEIGHT=0.5              # 人脸区增强监督权重 (1.0=complement, 0.5=dual)
#   FACE_SOFT=0                  # 1=用羽化 alpha 软权重, 0=二值 mask(阈值化+腐蚀)
#   LR_SCALE=0.1                 # 学习率缩放
#   WHITE_BG=0                   # 1=白底光栅化（须与 04 一致！）
#   RES=                         # --resolution（须与 04 一致！）
#   TEST_ITERATIONS="32000 35000"
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

GAUSSIAN_DIR="${GAUSSIAN_DIR:-$RESULTS_DIR/model_3dgs}"
SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/source}"
FACE_IMAGES_DIR="${FACE_IMAGES_DIR:-$RESULTS_DIR/source_face/images}"
FACE_MASKS_DIR="${FACE_MASKS_DIR:-$RESULTS_DIR/face_masks}"
FINETUNE_FROM="${FINETUNE_FROM:-30000}"
FINETUNE_ITERATIONS="${FINETUNE_ITERATIONS:-35000}"
GAUSSIAN_FACE_DIR="${GAUSSIAN_FACE_DIR:-$RESULTS_DIR/model_3dgs_face}"
FACE_WEIGHT="${FACE_WEIGHT:-0.5}"
FACE_SOFT="${FACE_SOFT:-0}"
LR_SCALE="${LR_SCALE:-0.1}"
WHITE_BG="${WHITE_BG:-0}"
RES="${RES:-}"
TEST_ITERATIONS="${TEST_ITERATIONS:-"32000 $FINETUNE_ITERATIONS"}"

START_PLY="$GAUSSIAN_DIR/point_cloud/iteration_$FINETUNE_FROM/point_cloud.ply"

echo "🧑 [06b] 人脸互补 finetune (30k ply → +$((FINETUNE_ITERATIONS-FINETUNE_FROM)) iters)"
echo "  📂 scene:        $SOURCE_DIR"
echo "  🔄 start ply:    $START_PLY"
echo "  🖼️ enhanced:     $FACE_IMAGES_DIR"
echo "  🎭 masks:        $FACE_MASKS_DIR (weight=$FACE_WEIGHT, soft=$FACE_SOFT)"
echo "  💾 model out:    $GAUSSIAN_FACE_DIR"
echo "  📐 $FINETUNE_FROM → $FINETUNE_ITERATIONS iters, lr×$LR_SCALE, densify OFF"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU:          physical $CUDA_VISIBLE_DEVICES"
fi
echo ""

# ── Sanity checks ─────────────────────────────────────────────────────────
if [ ! -f "$START_PLY" ]; then
    echo "❌ ERROR: start ply not found: $START_PLY" >&2
    echo "       Run 04_train_3dgs.sh to 30000 iterations first." >&2
    exit 1
fi
if [ ! -d "$SOURCE_DIR/sparse/0" ]; then
    echo "❌ ERROR: COLMAP scene not found: $SOURCE_DIR/sparse/0" >&2
    exit 1
fi
if [ ! -d "$FACE_IMAGES_DIR" ]; then
    echo "❌ ERROR: enhanced images not found: $FACE_IMAGES_DIR" >&2
    echo "       Run 06_face_enhance.sh first (HYPIR 人脸增强)." >&2
    exit 1
fi
if ! python -c "import diff_gaussian_rasterization, simple_knn" 2>/dev/null; then
    echo "❌ ERROR: 3DGS CUDA extensions not importable." >&2
    exit 1
fi

# ── 0. 人脸 loss mask（缺则自动生成；与 06 的 HYPIR 改动区域逐位对应）────────
if [ "$(ls "$FACE_MASKS_DIR"/*.mask.png 2>/dev/null | wc -l)" -eq 0 ]; then
    echo "🎭 face masks missing → generating (face_masks.py)"
    python "$SCRIPT_DIR/face_masks.py" \
        --images_dir "$SOURCE_DIR/images" \
        --output_dir "$FACE_MASKS_DIR" \
        --padding "${FACE_PADDING:-0.2}" --threshold 0.1 --erode_px 2
    if [ $? -ne 0 ]; then
        echo "❌ FAILED. face mask generation failed." >&2
        exit 1
    fi
fi

SOFT_FLAG=""
[ "$FACE_SOFT" = "1" ] && SOFT_FLAG="--face_soft"

TRAIN_FLAGS=(
    -s "$SOURCE_DIR"
    -m "$GAUSSIAN_FACE_DIR"
    --iterations "$FINETUNE_ITERATIONS"
    --start_ply "$START_PLY"
    --lr_scale "$LR_SCALE"
    --face_images_dir "$FACE_IMAGES_DIR"
    --face_masks_dir "$FACE_MASKS_DIR"
    --face_weight "$FACE_WEIGHT"
    $SOFT_FLAG
    --port 0
    --disable_viewer
    --test_iterations $TEST_ITERATIONS
    --save_iterations "$FINETUNE_ITERATIONS"
)
[ "$WHITE_BG" = "1" ] && TRAIN_FLAGS+=(--white_background)
[ -n "$RES" ] && TRAIN_FLAGS+=(--resolution "$RES")
[ -n "${TRAIN_EXTRA_ARGS:-}" ] && TRAIN_FLAGS+=($TRAIN_EXTRA_ARGS)

echo "🏋️ finetune training"
( cd "$GS_DIR" && python "$SCRIPT_DIR/train_face_finetune.py" "${TRAIN_FLAGS[@]}" )
if [ $? -ne 0 ]; then
    echo "❌ FAILED. face finetune did not complete." >&2
    exit 1
fi

echo ""
echo "✅ [06b] Done. face-finetuned gaussians:"
echo "  $GAUSSIAN_FACE_DIR/point_cloud/iteration_$FINETUNE_ITERATIONS/point_cloud.ply"
echo "  Next: 人脸区指标评估（LPIPS 对原图 + Laplacian 方差/梯度能量）"
