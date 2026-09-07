#!/usr/bin/env bash
# 03i_train_three_models.sh — 三模型独立训练（head / body / scene，区域 mask 监督）
#
# 架构（来自 09-05 grill-me 用户拍板）：高斯初始化即拆三个独立模型。
#   - head : 3DMM 模板 + 多视角拟合（03e_head_3dmm/head_gs_p{pid}.ply）
#   - body : SAM3 person mask 内、SfM 投票通过、head 邻域外的点（03h）
#   - scene: 剩余 SfM 点（03h/scene_gs.ply）
#
# 每个模型只在自己的区域 mask 内被监督（区域外用 gt 填充 → loss/梯度=0；
# 区域外 densify 自然不发生）—— 这样三个模型真正独立，不会因 densify/prune
# 混合成一个大模型。
#
# 前置：03g（head_gs）+ 03h（body_gs/scene_gs）+ 03i_region_masks
#
# Env:
#   RESULTS_DIR     输出根
#   SOURCE_DIR      COLMAP 场景（默认 $RESULTS_DIR/03b_source_ba）
#   HEAD_GS_DIR     head_gs 目录（默认 $RESULTS_DIR/03e_head_3dmm）
#   BODY_GS_DIR     body_gs 目录（默认 $RESULTS_DIR/03h_person_scene_split）
#   REGION_MASKS_DIR 区域 mask（默认 $RESULTS_DIR/03i_region_masks）
#   MODEL_KIND      head | body | scene
#   PID             00 | 01 | 02（head/body 需要；scene 忽略）
#   ITERATIONS      训练轮数（默认 1000=验证；生产用 30000）
#   LR_SCALE        起始 LR 缩放（head 1.0, body/scene 0.1）
#   DENSIFY_UNTIL   densify 终止 iter（默认 0=冻结）
#   START_PLY       覆盖默认初始化 ply（checkpoint 续训用）
#   START_ITER      续训起始 iter（train_face_finetune 从目录名解析）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

RESULTS_DIR="${RESULTS_DIR:-$RESULTS_ROOT}"
SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/03b_source_ba}"
HEAD_GS_DIR="${HEAD_GS_DIR:-$RESULTS_DIR/03e_head_3dmm}"
BODY_GS_DIR="${BODY_GS_DIR:-$RESULTS_DIR/03h_person_scene_split}"
REGION_MASKS_DIR="${REGION_MASKS_DIR:-$RESULTS_DIR/03i_region_masks}"
MODEL_KIND="${MODEL_KIND:-head}"
PID="${PID:-00}"
ITERATIONS="${ITERATIONS:-1000}"
LR_SCALE="${LR_SCALE:-1.0}"
DENSIFY_UNTIL="${DENSIFY_UNTIL:-0}"

# 选 start_ply（START_PLY 环境变量可覆盖，用于从 checkpoint 续训）
case "$MODEL_KIND" in
    head)
        START_PLY="${START_PLY:-$HEAD_GS_DIR/head_gs_p${PID}.ply}"
        [ ! -f "$START_PLY" ] && { echo "❌ 缺 $START_PLY"; exit 1; }
        OUT_DIR="$RESULTS_DIR/03i_${MODEL_KIND}_p${PID}"
        ;;
    body)
        START_PLY="${START_PLY:-$BODY_GS_DIR/body_gs_p${PID}.ply}"
        [ ! -f "$START_PLY" ] && { echo "❌ 缺 $START_PLY"; exit 1; }
        OUT_DIR="$RESULTS_DIR/03i_${MODEL_KIND}_p${PID}"
        ;;
    scene)
        START_PLY="${START_PLY:-$BODY_GS_DIR/scene_gs.ply}"
        [ ! -f "$START_PLY" ] && { echo "❌ 缺 $START_PLY"; exit 1; }
        OUT_DIR="$RESULTS_DIR/03i_scene"
        PID=""  # scene 不需要
        ;;
    *)
        echo "❌ MODEL_KIND must be head|body|scene, got: $MODEL_KIND" >&2
        exit 1
        ;;
esac

[ ! -d "$REGION_MASKS_DIR" ] && { echo "❌ 缺 $REGION_MASKS_DIR (先跑 build_region_masks.py)"; exit 1; }

echo "🧩 [03i] 三模型独立训练: $MODEL_KIND${PID:+ p$PID}"
echo "  📂 scene:      $SOURCE_DIR"
echo "  🧱 start_ply:  $START_PLY"
echo "  🎭 region mask: $REGION_MASKS_DIR ($MODEL_KIND${PID:+ p$PID})"
echo "  💾 out:         $OUT_DIR"
echo "  ⚙️ iter=$ITERATIONS  lr_scale=$LR_SCALE  densify_until=$DENSIFY_UNTIL"

TRAIN_FLAGS=(
    -s "$SOURCE_DIR"
    -m "$OUT_DIR"
    --iterations "$ITERATIONS"
    --start_ply "$START_PLY"
    --region_masks_dir "$REGION_MASKS_DIR"
    --model_kind "$MODEL_KIND"
    ${PID:+--pid "$PID"}
    --lr_scale "$LR_SCALE"
    --densify_until "$DENSIFY_UNTIL"
    --port 0
    --disable_viewer
    --test_iterations "$ITERATIONS"
    --save_iterations "$ITERATIONS"
)
# 🧩 head 模型：传 head_fit.json 启用逐帧可微 warp（canonical 高斯 + 每帧刚性变形）
if [ "$MODEL_KIND" = "head" ] && [ -n "${PID}" ]; then
    export HEAD_WARP_JSON="$HEAD_GS_DIR/head_fit.json"
fi
[ "${WHITE_BG:-0}" = "1" ] && TRAIN_FLAGS+=(--white_background)
[ -n "${RES:-}" ] && TRAIN_FLAGS+=(--resolution "$RES")

mkdir -p "$OUT_DIR"
( cd "$GS_DIR" && python "$SCRIPT_DIR/train_face_finetune.py" "${TRAIN_FLAGS[@]}" )
if [ $? -ne 0 ]; then
    echo "❌ 三模型训练失败" >&2
    exit 1
fi

echo ""
echo "✅ [03i] $MODEL_KIND${PID:+ p$PID} 训练完成"
echo "  $OUT_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
