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
#   ITERATIONS=10000       # 快速迭代默认；最终出片用 20000。
#                          # 消融（hand_motion/vggt_source）：7k 掉 1.5~2dB
#                          # （形变 MLP 未收敛，收益被吃光），10k 只掉 0.7~1dB
#   WARM_UP=1000           # 前 N 步形变量=0（纯静态热身，官方默认 3000；
#                          # 配合 10k 短训等比缩短，让形变 MLP 更早开工）
#   IS_6DOF=0              # 1=6DoF 变体（指标略高、更慢）
#   WHITE_BG=0             # 1=白底（输入做了分割抠图时开）
#   EVAL=1                 # 1=划分 train/test（llffhold=8；要指标必须开）
#   USE_POSE_REFINE=0      # 1=训练中联合精炼位姿（可学四元数+平移，见
#                          #   train_pose_refine.py；内参不学——stock 光栅化器
#                          #   对 projmatrix 无梯度，见 README_wsl.md）
#   POSE_REFINE_START=0    # >0 时延迟到第 N 步才启用位姿精炼（BARF 式课程，
#                          #   几何成型后启动防垃圾梯度拖走初值；坏初值场景
#                          #   建议 WARM_UP+2000，见 README 实验记录）
#   POSE_REFINE_LR_WARMUP=2000      # B v2：启动后 lr 从 0 线性爬到满值的步数
#   POSE_REFINE_ROT_ONLY_STEPS=1000 # B v2：启动后只放开旋转的步数，之后才放开平移
#   USE_MASK_LOSS=0        # 1=前景 mask 加权 L1（train_mask.py；先跑 01d）。
#                          #   前景权重向人倾斜，稠密化随梯度自然向人集中。
#   MASK_DIR=              # 默认 $SOURCE_PATH/masks（01d 的输出）
#   FG_WEIGHT=1.0  BG_WEIGHT=0.2
#   BG_DEFORM_LAMBDA=0     # >0 时启用背景点变形量趋零软正则（warm_up 后生效，
#                          #   建议 0.1；梯度只作用 deform MLP，见 train_mask.py）
#   C2F=0                  # 1=coarse-to-fine 分辨率调度（train_mask.py）：
#                          #   前 30% 步 1/4 分辨率 → 30-70% 1/2 → 末段全分辨率，
#                          #   评测仍按原生分辨率。⚠️ 仅 USE_MASK_LOSS=1 分支可用
#   C2F_SCHEDULE=          # 显式档位覆盖默认，如 "0:4,3000:2,7000:1"
#   USE_DENSIFY_SCHED=0    # 1=方案 D 稠密化自动调度（train_mask.py，经 shim 复用
#                          #   som_human/densify_sched.py）：cap>0 显式上限，cap=0
#                          #   momentum 自适应；topk 截断用 absgrad 均值分数
#   MAX_GAUSSIANS=         # 点数硬护栏（空/0=关）。调度器开时兼作显式 cap
#   DENSIFY_MAX_RATE=1.0   # 相邻周期 next_n 相对增长上限（服务器同名常量）
#   DENSIFY_CNT_CLAMP=3.0  # 分数分母 clamp（服务器 /cnt.clamp_min(3)）
#   DENSIFY_PRUNE_AFTER=1  # 窗口结束后继续按期 prune（只剪不增）
#   ※ USE_MASK_LOSS 与 USE_POSE_REFINE 互斥（两个独立 fork，需要时再合并）
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
ITERATIONS="${ITERATIONS:-10000}"
WARM_UP="${WARM_UP:-1000}"
IS_6DOF="${IS_6DOF:-0}"
WHITE_BG="${WHITE_BG:-0}"
EVAL="${EVAL:-1}"
USE_POSE_REFINE="${USE_POSE_REFINE:-0}"
POSE_REFINE_START="${POSE_REFINE_START:-0}"
POSE_REFINE_LR_WARMUP="${POSE_REFINE_LR_WARMUP:-2000}"
POSE_REFINE_ROT_ONLY_STEPS="${POSE_REFINE_ROT_ONLY_STEPS:-1000}"
USE_MASK_LOSS="${USE_MASK_LOSS:-0}"
MASK_DIR="${MASK_DIR:-$SOURCE_PATH/masks}"
FG_WEIGHT="${FG_WEIGHT:-1.0}"
BG_WEIGHT="${BG_WEIGHT:-0.2}"
BG_DEFORM_LAMBDA="${BG_DEFORM_LAMBDA:-0}"
C2F="${C2F:-0}"
C2F_SCHEDULE="${C2F_SCHEDULE:-}"
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
    --warm_up "$WARM_UP"
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
echo "    (warm_up 前 $WARM_UP 步变形量=0；致密化/opacity 重置节奏由 EXTRA_TRAIN_ARGS 或官方默认决定)"
echo ""

# train.py 用相对 import → 必须在 $DG_DIR 里跑
if [ "$USE_MASK_LOSS" = "1" ] && [ "$USE_POSE_REFINE" = "1" ]; then
    echo "❌ ERROR: USE_MASK_LOSS 与 USE_POSE_REFINE 互斥（两个独立 fork，需要时再合并）" >&2
    exit 1
fi
if [ "$C2F" = "1" ] && [ "$USE_MASK_LOSS" != "1" ]; then
    echo "❌ ERROR: C2F 分辨率调度挂在 train_mask.py fork 里（官方 train.py 不改）。" >&2
    echo "       请叠加 USE_MASK_LOSS=1；若想要无 mask 的 C2F，设 FG_WEIGHT=1 BG_WEIGHT=1" >&2
    echo "       （此时加权 L1 数学上退化为普通 L1）。" >&2
    exit 1
fi
if [ "$USE_MASK_LOSS" = "1" ]; then
    [ -d "$MASK_DIR" ] || { echo "❌ ERROR: mask 目录不存在: $MASK_DIR（先跑 01d_fg_masks.sh）" >&2; exit 1; }
    export DG_DIR
    echo "🎭 using train_mask.py（前景加权 L1，fg=$FG_WEIGHT bg=$BG_WEIGHT，mask: $MASK_DIR）"
    [ "$BG_DEFORM_LAMBDA" != "0" ] && echo "   + 背景变形软正则 λ=$BG_DEFORM_LAMBDA（warm_up 后生效）"
    C2F_FLAGS=()
    if [ "$C2F" = "1" ]; then
        C2F_FLAGS+=(--c2f)
        [ -n "$C2F_SCHEDULE" ] && C2F_FLAGS+=(--res_schedule "$C2F_SCHEDULE")
        echo "   🔽 C2F 分辨率调度启用${C2F_SCHEDULE:+（档位: $C2F_SCHEDULE）}（默认 0-30% 1/4 → 30-70% 1/2 → 末段全分辨率）"
    fi
    # 方案 D：稠密化调度器 / 硬护栏（实现本体在 som_human/densify_sched.py，
    # 经 deformable_human/densify_sched.py shim 引入）
    DENSIFY_FLAGS=()
    if [ "$USE_DENSIFY_SCHED" = "1" ]; then
        DENSIFY_FLAGS+=(--densify_sched 1)
        DENSIFY_FLAGS+=(--max_gaussians "${MAX_GAUSSIANS:-0}")
        DENSIFY_FLAGS+=(--densify_max_rate "${DENSIFY_MAX_RATE:-1.0}")
        DENSIFY_FLAGS+=(--densify_cnt_clamp "${DENSIFY_CNT_CLAMP:-3.0}")
        DENSIFY_FLAGS+=(--densify_prune_after "${DENSIFY_PRUNE_AFTER:-1}")
        echo "   📈 方案 D 调度器启用（cap=${MAX_GAUSSIANS:-0=关/momentum}，rate=${DENSIFY_MAX_RATE:-1.0}，clamp=${DENSIFY_CNT_CLAMP:-3.0}，prune_after=${DENSIFY_PRUNE_AFTER:-1}）"
    elif [ -n "$MAX_GAUSSIANS" ] && [ "$MAX_GAUSSIANS" != "0" ]; then
        DENSIFY_FLAGS+=(--max_gaussians "$MAX_GAUSSIANS")
        echo "   🛡️ 硬护栏 MAX_GAUSSIANS=$MAX_GAUSSIANS（到顶只 prune 不 densify）"
    fi
    ( cd "$DG_DIR" && python "$SCRIPT_DIR/train_mask.py" "${TRAIN_FLAGS[@]}" \
        --mask_dir "$MASK_DIR" --fg_weight "$FG_WEIGHT" --bg_weight "$BG_WEIGHT" \
        --bg_deform_lambda "$BG_DEFORM_LAMBDA" "${C2F_FLAGS[@]}" "${DENSIFY_FLAGS[@]}" )
elif [ "$USE_POSE_REFINE" = "1" ]; then
    # 可学位姿（场景侧 delta 注入，梯度真实有效；原理与限制见 README_wsl.md）
    export USE_POSE_REFINE POSE_REFINE_WEIGHT POSE_REFINE_LR_Q POSE_REFINE_LR_T POSE_REFINE_START \
           POSE_REFINE_LR_WARMUP POSE_REFINE_ROT_ONLY_STEPS DG_DIR
    echo "🧭 using train_pose_refine.py（可学四元数+平移，"
    echo "    w=${POSE_REFINE_WEIGHT:-0.01} lr_q=${POSE_REFINE_LR_Q:-1e-3} lr_t=${POSE_REFINE_LR_T:-1e-3} start=${POSE_REFINE_START}"
    echo "    课程: lr_warmup=${POSE_REFINE_LR_WARMUP} rot_only=${POSE_REFINE_ROT_ONLY_STEPS}）"
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
