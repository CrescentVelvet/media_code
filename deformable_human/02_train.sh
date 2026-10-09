#!/usr/bin/env bash
# 02_train.sh — vanilla Deformable-3D-Gaussians 训练（canonical 高斯 + 形变 MLP）。
#
# 这是「去重影」的 baseline：不叠加任何 MHR 锚定，先量化 vanilla 能把
# 单目动态人体的重影消到几成，再决定 Phase 2 锚定要加多重。
# 逻辑镜像 deformable_gaussians/04_train_real.sh（NeRF-DS 模式，20000 步），
# 差异仅在默认路径与 env（本目录 env 是 torch 2.5.1 clone，不是 py3.7/torch1.13）。
#
# 输入：SOURCE_PATH/{images/, sparse/0/*.bin}（01 的输出）
# 输出：MODEL_PATH/
#         point_cloud/iteration_<N>/point_cloud.ply   canonical 高斯
#         deform/                                     形变 MLP 权重
#         pose_refine.json                            USE_POSE_REFINE=1 时的精炼位姿
#         cfg_args / cameras.json / input.ply / TensorBoard events
#
# Env (all optional, defaults shown):
#   SCENE_NAME=human_seq
#   SOURCE_PATH=           # 默认 $RESULTS_DIR/$SCENE_NAME/colmap_scene
#   MODEL_PATH=            # 默认 $RESULTS_DIR/$SCENE_NAME/model
#   ITERATIONS=20000       # NeRF-DS 真实序列标配（D-NeRF 才用 40000）
#   IS_6DOF=0              # 1=6DoF 变体（指标略高、更慢）
#   WHITE_BG=0             # 1=白底（输入做了分割抠图时开）
#   EVAL=1                 # 1=划分 train/test（llffhold=8；要指标必须开）
#   USE_POSE_REFINE=0      # 1=训练中联合精炼位姿（可学四元数+平移，见
#                          #   train_pose_refine.py；内参不学——stock 光栅化器
#                          #   对 projmatrix 无梯度，见 README_wsl.md）
#   POSE_REFINE_WEIGHT=0.01  POSE_REFINE_LR_Q=1e-3  POSE_REFINE_LR_T=1e-3
#   TEST_ITERATIONS=       # 覆盖评测步（默认 train.py 自带）
#   SAVE_ITERATIONS=       # 覆盖存盘步
#   SKIP_VERIFY=0          # 1=跳过 CUDA 扩展 import 校验
#   EXTRA_TRAIN_ARGS=      # 透传（如 --sh_degree 2）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE_NAME="${SCENE_NAME:-human_seq}"
SOURCE_PATH="${SOURCE_PATH:-$RESULTS_DIR/$SCENE_NAME/colmap_scene}"
MODEL_PATH="${MODEL_PATH:-$RESULTS_DIR/$SCENE_NAME/model}"
ITERATIONS="${ITERATIONS:-20000}"
IS_6DOF="${IS_6DOF:-0}"
WHITE_BG="${WHITE_BG:-0}"
EVAL="${EVAL:-1}"
USE_POSE_REFINE="${USE_POSE_REFINE:-0}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

echo "🚀 [02] 训练 vanilla Deformable-GS（canonical + 形变 MLP）"
echo "  🤖 代码:       $DG_DIR"
echo "  📂 数据源:     $SOURCE_PATH"
echo "  💾 输出:       $MODEL_PATH"
echo "  📐 iterations: $ITERATIONS  is_6dof: $IS_6DOF  white_bg: $WHITE_BG  eval: $EVAL"
echo ""

# ── 前置检查 ───────────────────────────────────────────────────────────────
if [ ! -f "$DG_DIR/train.py" ]; then
    echo "❌ ERROR: 官方仓未就绪: $DG_DIR/train.py（先跑 00a_setup_env.sh）" >&2
    exit 1
fi
if [ ! -d "$SOURCE_PATH/sparse/0" ] || [ ! -d "$SOURCE_PATH/images" ]; then
    echo "❌ ERROR: COLMAP 场景不完整: $SOURCE_PATH（先跑 01_prepare_data.sh）" >&2
    exit 1
fi
if [ "$SKIP_VERIFY" != "1" ]; then
    if ! python -c "import diff_gaussian_rasterization, simple_knn" 2>/dev/null; then
        echo "❌ ERROR: CUDA 扩展不可 import。BUILD_CUDA=1 bash $SCRIPT_DIR/00a_setup_env.sh" >&2
        exit 1
    fi
fi

# ── 组装参数并训练 ──────────────────────────────────────────────────────────
mkdir -p "$MODEL_PATH"
TRAIN_FLAGS=(
    -s "$SOURCE_PATH"
    -m "$MODEL_PATH"
    --iterations "$ITERATIONS"
    --port 0            # 关掉 GUI server（默认会卡住等连接）
)
[ "$EVAL" = "1" ] && TRAIN_FLAGS+=(--eval)
[ "$IS_6DOF" = "1" ] && TRAIN_FLAGS+=(--is_6dof)
[ "$WHITE_BG" = "1" ] && TRAIN_FLAGS+=(--white_background)
# shellcheck disable=SC2206
[ -n "${TEST_ITERATIONS:-}" ] && TRAIN_FLAGS+=(--test_iterations $TEST_ITERATIONS)
# shellcheck disable=SC2206
[ -n "${SAVE_ITERATIONS:-}" ] && TRAIN_FLAGS+=(--save_iterations $SAVE_ITERATIONS)
# shellcheck disable=SC2086
[ -n "$EXTRA_TRAIN_ARGS" ] && TRAIN_FLAGS+=($EXTRA_TRAIN_ARGS)

echo "🏋️ train.py ${TRAIN_FLAGS[*]}"
echo "    (warm_up 前 3000 步变形量=0；前 15000 步致密化；每 3000 步 opacity 重置)"
echo ""

# train.py 用相对 import → 必须在 $DG_DIR 里跑
if [ "$USE_POSE_REFINE" = "1" ]; then
    # 可学位姿（场景侧 delta 注入，梯度真实有效；原理与限制见 README_wsl.md）
    export USE_POSE_REFINE POSE_REFINE_WEIGHT POSE_REFINE_LR_Q POSE_REFINE_LR_T DG_DIR
    echo "🧭 using train_pose_refine.py（可学四元数+平移，"
    echo "    w=${POSE_REFINE_WEIGHT:-0.01} lr_q=${POSE_REFINE_LR_Q:-1e-3} lr_t=${POSE_REFINE_LR_T:-1e-3}）"
    ( cd "$DG_DIR" && python "$SCRIPT_DIR/train_pose_refine.py" "${TRAIN_FLAGS[@]}" )
else
    ( cd "$DG_DIR" && python train.py "${TRAIN_FLAGS[@]}" )
fi
if [ $? -ne 0 ]; then
    echo "❌ FAILED: train.py 没跑完。常见原因:" >&2
    echo "    - OOM: 减 ITERATIONS / 01 里 NUM_IMAGES_MAX 截帧" >&2
    echo "    - fid ValueError: 图像名不是纯数字（01 的 COLMAP 步应已重命名）" >&2
    echo "    - import 错: fork 光栅化器没编译过（回 00a BUILD_CUDA=1）" >&2
    exit 1
fi

LAST_ITER="$(ls -d "$MODEL_PATH/point_cloud/iteration_"* 2>/dev/null | sort -V | tail -1 | sed 's/.*iteration_//')"
echo ""
echo "🎉 [02] Done."
echo "  🏋️ 高斯点云:  $MODEL_PATH/point_cloud/iteration_${LAST_ITER:-$ITERATIONS}/point_cloud.ply"
echo "  🎭 形变 MLP:   $MODEL_PATH/deform/"
echo "  📊 TensorBoard: tensorboard --logdir $MODEL_PATH --port 6006"
echo "  → 渲染评测: GPU=0 MODEL_PATH=$MODEL_PATH bash deformable_human/03_render.sh"
