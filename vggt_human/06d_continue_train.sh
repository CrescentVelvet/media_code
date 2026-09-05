#!/usr/bin/env bash
# 06d_continue_train.sh — 简化续训配方：继续训练提升收敛（无需人脸监督）
#
# 背景（2026-09-04 消融实验结论，详见 AGENTS.md §12）：
#   对照实验证明 06c 人脸监督（masked-L1→HYPIR 增强图）本身无可测量贡献，
#   人脸区锐度提升（训练视角 +7.1%、常规新视角清晰度 +23%）全部来自
#   "30k 基线欠收敛 + lr_scale=0.2 继续训练"。本脚本即该配方的正式化。
#
# 前提: 04/04b 训完（point_cloud/iteration_$ITERATION/point_cloud.ply）
# Env (all optional, defaults shown):
#   RESULTS_DIR=                 # 输出根
#   GAUSSIAN_DIR=                # 基线模型 (默认 04b_model_3dgs_ba)
#   SOURCE_DIR=                  # COLMAP 场景 (默认 03b_source_ba)
#   ITERATION=30000              # 起始 checkpoint 迭代
#   EXTRA_ITERS=20000            # 续训步数 (终点 = ITERATION + EXTRA_ITERS)
#   LR_SCALE=0.2                 # 学习率缩放 (0.1 收敛偏慢, 0.2 实测最优)
#   FACE_WEIGHT=0                # 人脸监督权重 (0=关; 实验证明无贡献, 保留开关)
#   DENSIFY_UNTIL=0              # densify_until_iter (0=冻结续训, 15000=开启加密, 消融用)
#   GAUSSIAN_CONTINUE_DIR=       # 输出模型 (默认 06d_model_3dgs_continue)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

RESULTS_DIR="${RESULTS_DIR:-$RESULTS_ROOT}"
GAUSSIAN_DIR="${GAUSSIAN_DIR:-$RESULTS_DIR/04b_model_3dgs_ba}"
SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/03b_source_ba}"
ITERATION="${ITERATION:-30000}"
EXTRA_ITERS="${EXTRA_ITERS:-20000}"
LR_SCALE="${LR_SCALE:-0.2}"
FACE_WEIGHT="${FACE_WEIGHT:-0}"
DENSIFY_UNTIL="${DENSIFY_UNTIL:-0}"
GAUSSIAN_CONTINUE_DIR="${GAUSSIAN_CONTINUE_DIR:-$RESULTS_DIR/06d_model_3dgs_continue}"
FINETUNE_ITERATIONS=$((ITERATION + EXTRA_ITERS))

START_PLY="$GAUSSIAN_DIR/point_cloud/iteration_$ITERATION/point_cloud.ply"
[ -f "$START_PLY" ] || { echo "❌ start ply not found: $START_PLY" >&2; exit 1; }

echo "🏋️ [06d] 续训收敛: iter $ITERATION -> $FINETUNE_ITERATIONS (lr_scale=$LR_SCALE, face_weight=$FACE_WEIGHT)"
echo "  📂 start ply:  $START_PLY"
echo "  📂 scene:      $SOURCE_DIR"
echo "  📂 out:        $GAUSSIAN_CONTINUE_DIR"
echo ""

# face_weight=0 时不需要人脸数据目录; >0 时必须给平铺的图片目录(非其父目录)
TRAIN_FLAGS=(
    -s "$SOURCE_DIR"
    -m "$GAUSSIAN_CONTINUE_DIR"
    --iterations "$FINETUNE_ITERATIONS"
    --start_ply "$START_PLY"
    --lr_scale "$LR_SCALE"
    --face_weight "$FACE_WEIGHT"
    --densify_until "$DENSIFY_UNTIL"
    --port 0
    --disable_viewer
    --test_iterations "$FINETUNE_ITERATIONS"
    --save_iterations "$FINETUNE_ITERATIONS"
)
if [ "$FACE_WEIGHT" != "0" ]; then
    MERGED_FACE_IMAGES="${MERGED_FACE_IMAGES:-$RESULTS_DIR/06c_merged_face_images/images}"
    MERGED_MASKS="${MERGED_MASKS:-$RESULTS_DIR/06c_merged_face_masks}"
    TRAIN_FLAGS+=(--face_images_dir "$MERGED_FACE_IMAGES" --face_masks_dir "$MERGED_MASKS")
fi
[ "${WHITE_BG:-0}" = "1" ] && TRAIN_FLAGS+=(--white_background)
[ -n "${RES:-}" ] && TRAIN_FLAGS+=(--resolution "$RES")

mkdir -p "$GAUSSIAN_CONTINUE_DIR"
( cd "$GS_DIR" && python "$SCRIPT_DIR/train_face_finetune.py" "${TRAIN_FLAGS[@]}" )
if [ $? -ne 0 ]; then
    echo "❌ continue-train failed" >&2
    exit 1
fi

echo ""
echo "✅ [06d] Done. continued gaussians:"
echo "  $GAUSSIAN_CONTINUE_DIR/point_cloud/iteration_$FINETUNE_ITERATIONS/point_cloud.ply"
