#!/usr/bin/env bash
# 02_train.sh — SoM 式运动基训练（canonical 高斯 + 逐帧 SE(3) 基凸组合）。
#
# 这是「手部微动方案」Phase 1 的核心验证：在 deformable_human 框架内把逐帧
# 形变 MLP 换成低秩运动基，检验「持久身份 + 跨帧低秩约束」能否消除手部
# 重影/撕裂（成因③）。数据准备完全复用 deformable_human/01_prepare_data.sh
# 的产物（COLMAP 场景），本目录不重复建设。
#
# 输入：SOURCE_PATH/{images/, sparse/0/*.bin}（deformable_human 01 的输出）
# 输出：MODEL_PATH/
#         point_cloud/iteration_<N>/point_cloud.ply   canonical 高斯
#         som_motion/iteration_<N>/som.pth            运动基 + 系数场权重
#         cfg_args / cameras.json / TensorBoard events
#
# Env (all optional, defaults shown):
#   SCENE_NAME=human_seq
#   SOURCE_PATH=           # 默认 $DEFORMABLE_HUMAN_RESULTS_DIR/$SCENE_NAME/colmap_scene
#                          # （回退 $HOME/output/deformable_human_results/$SCENE_NAME/colmap_scene）
#   MODEL_PATH=            # 默认 $RESULTS_DIR/$SCENE_NAME/model
#   ITERATIONS=10000       # 对齐 deformable_human 快速迭代口径；最终出片 20000
#   WARM_UP=1000           # 前 N 步纯静态热身（基此时是 identity，天然等价）
#   NUM_BASES=10           # SoM 全局基数量（Phase 1 扫 {8,12,16} 前先跑默认）
#   SMOOTH_WEIGHT=0.1      # 基加速度时间平滑权重（SoM 0.1）
#   BASIS_LR=1.6e-4        # 基 rots/transls lr（SoM configs.py:23-26）
#   COEF_LR=1e-2           # 系数场 lr（SoM configs.py:5-11）
#   WHITE_BG=0  EVAL=1
#   RES_SCHEDULE=""           # 分阶段分辨率。⚠️ 2026-10-13 A/B/C 实测：本链路
#                              # 瓶颈是随高斯数增长的非渲染开销（SoM 系数场/Adam/稠密化
#                              # 记账均 ∝N），1/4 分辨率渲染省的时间≈0（6:48 vs 6:45），
#                              # 阈值补偿反而多 29% 高斯拖慢 34%。默认关闭，代码保留备查。
#   RES_COMPENSATE_DENSIFY=0   # 配合上条，实测有害（README 负结果节有数据）
#   EXTRA_TRAIN_ARGS=      # 透传。⚠️ 复杂背景必须带降载参数（deformable_human 实测）：
#                          #   '-r 2 --densify_grad_threshold 0.0004 --densify_until_iter 2000'
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE_NAME="${SCENE_NAME:-human_seq}"
_DH_RESULTS="${DEFORMABLE_HUMAN_RESULTS_DIR:-$HOME/output/deformable_human_results}"
SOURCE_PATH="${SOURCE_PATH:-$_DH_RESULTS/$SCENE_NAME/colmap_scene}"
MODEL_PATH="${MODEL_PATH:-$RESULTS_DIR/$SCENE_NAME/model}"
ITERATIONS="${ITERATIONS:-10000}"
WARM_UP="${WARM_UP:-1000}"
WHITE_BG="${WHITE_BG:-0}"
EVAL="${EVAL:-1}"
NUM_BASES="${NUM_BASES:-10}"
SMOOTH_WEIGHT="${SMOOTH_WEIGHT:-0.1}"
BASIS_LR="${BASIS_LR:-1.6e-4}"
COEF_LR="${COEF_LR:-1e-2}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
RES_SCHEDULE="${RES_SCHEDULE-}"             # 注意 ${VAR-}（无冒号）：默认关闭（实测零加速）
RES_COMPENSATE_DENSIFY="${RES_COMPENSATE_DENSIFY:-0}"
MAX_GAUSSIANS="${MAX_GAUSSIANS:-600000}"    # 高斯数上限护栏（v2 爆炸事故后引入；
                                            # 2026-10-15 cap600k A/B：@10k 掉 0.09dB<
                                            # 0.3dB 判定线、@20k 21.81 反超、2.4× 省点、
                                            # ~1.7× 加速 → 默认值 150万 改 60万）
USE_DENSIFY_SCHED="${USE_DENSIFY_SCHED:-0}"  # 方案 D 自动调度（1=开）：显式 cap 为主，
                                             # 候选按 absgrad 分数 topk 截断，窗口后只剪不增
DENSIFY_MOMENTUM_MULT="${DENSIFY_MOMENTUM_MULT:-auto}"  # auto: init>=10万×0.5 否则×4
DENSIFY_MAX_RATE="${DENSIFY_MAX_RATE:-1.0}"  # 相邻周期 next_n 相对增长上限（服务器同名常量）
DENSIFY_CNT_CLAMP="${DENSIFY_CNT_CLAMP:-3.0}"  # 分数分母 clamp（服务器 /cnt.clamp_min(3)）
DENSIFY_PRUNE_AFTER="${DENSIFY_PRUNE_AFTER:-1}"  # densify 窗口结束后继续按期 prune（只剪不增）
USE_EXPOSURE="${USE_EXPOSURE:-0}"            # 方案 A 曝光仿射补偿（1=开）：
                                             # 每相机可学 4×3 仿射，门控开启瞬间 lstsq
                                             # 闭式初始化，test 帧按 fid 插值
EXPOSURE_FROM_ITER="${EXPOSURE_FROM_ITER:--1}"  # -1 = WARM_UP（几何成形后再放曝光）
EXPOSURE_LR="${EXPOSURE_LR:-1e-3}"           # 仿射独立小 lr
EXPOSURE_REG="${EXPOSURE_REG:-0.01}"         # 拉回恒等正则（防与 SH 互相吸收）
DEPTH_LAMBDA="${DEPTH_LAMBDA:-0}"            # 方案 E 深度先验权重（0=关；0.05 起步）。
                                             # ⚠️ 点级锚定（非渲染深度 loss，本仓光栅化器
                                             # depth 无梯度回路）；需先跑 01f_depth_maps.sh
                                             # 产出 $SOURCE_PATH/depths/*.npz
DEPTH_FROM_ITER="${DEPTH_FROM_ITER:--1}"     # -1 = WARM_UP 后启用（且强制 ≥ warm_up）
DEPTH_USE_MASK="${DEPTH_USE_MASK:-1}"        # 1=有 01d 前景 mask 就用（<scene>/masks/）
DEPTH_MIN_POINTS="${DEPTH_MIN_POINTS:-2048}" # 帧内投影点不足则该步跳过深度损失
# ── 单人前景监督（per-person 独立重建，2026-10-17）─────────────────────────
# 01g 出 SAM3 多人 mask、01h 拆出每人场景后，把该人的 masks/ 指过来：
#   FG_MASK_DIR=<scene>/persons/p00/masks
# loss 只在该 mask 内计算 + COLMAP 初始点按多视角投票过滤，产出**只有这个人**
# 的高斯（不含背景/他人）。多人场景的每个人各训一个 MODEL_PATH。
FG_MASK_DIR="${FG_MASK_DIR:-}"               # 空=全图监督（历史行为）
FG_INIT_FILTER="${FG_INIT_FILTER:-1}"        # 1=按 mask 过滤初始点（强烈建议）
FG_INIT_MIN_VIEWS="${FG_INIT_MIN_VIEWS:-3}"  # 至少落入多少帧的 mask 才保留
FG_INIT_MIN_RATIO="${FG_INIT_MIN_RATIO:-0.3}" # 落入比例下限（投票/可见）
BG_RANDOM="${BG_RANDOM:-1}"                  # 1=训练时背景色每步随机
# 手部区域 loss 加权（档 0 判决实验）：需 FG_MASK_DIR 已启用；
# hand mask 用 01g（SAM3_PROMPT=hand）+ 01i 合并产出。W=1 即关闭。
HAND_MASK_DIR="${HAND_MASK_DIR:-}"           # 例 <scene>/hands/merged
HAND_WEIGHT="${HAND_WEIGHT:-1.0}"            # 手部像素相对权重（L1 项）

echo "🚀 [02] 训练 SoM 运动基（canonical + ${NUM_BASES} 条 SE(3) 基凸组合）"
echo "  🤖 代码:       $DG_DIR（官方仓）+ $SCRIPT_DIR/{som_motion,train_som}.py"
echo "  📂 数据源:     $SOURCE_PATH"
echo "  💾 输出:       $MODEL_PATH"
echo "  📐 iterations: $ITERATIONS  white_bg: $WHITE_BG  eval: $EVAL"
echo "  🧬 bases:      B=$NUM_BASES  smooth_w=$SMOOTH_WEIGHT  basis_lr=$BASIS_LR  coef_lr=$COEF_LR"
echo "  🚀 res_sched:  ${RES_SCHEDULE:-（关闭，全程全分辨率）}  compensate_densify=$RES_COMPENSATE_DENSIFY"
echo "  🛡️ max_gauss:  $MAX_GAUSSIANS（超过即停止 densify 只 prune）"
echo "  📈 dens_sched: $USE_DENSIFY_SCHED（mult=$DENSIFY_MOMENTUM_MULT, rate=$DENSIFY_MAX_RATE, clamp=$DENSIFY_CNT_CLAMP, prune_after=$DENSIFY_PRUNE_AFTER；0 时上面那就是硬护栏）"
echo "  💡 exposure:   $USE_EXPOSURE（from=${EXPOSURE_FROM_ITER:-warm_up}, lr=$EXPOSURE_LR, reg=$EXPOSURE_REG）"
echo "  🌊 depth:      λ=$DEPTH_LAMBDA（点级锚定, mask=$DEPTH_USE_MASK, from=${DEPTH_FROM_ITER:-warm_up}）"
echo "  🎭 fg_mask:    ${FG_MASK_DIR:-（关闭，全图监督）}（init_filter=$FG_INIT_FILTER, bg_random=$BG_RANDOM）"
echo "  ✋ hand_mask:  ${HAND_MASK_DIR:-（关闭）}（weight=$HAND_WEIGHT）"
echo ""

# ── 前置检查 ───────────────────────────────────────────────────────────────
if [ ! -f "$DG_DIR/train.py" ]; then
    echo "❌ ERROR: 官方仓未就绪: $DG_DIR/train.py（先跑 00a_setup_env.sh）" >&2
    exit 1
fi
if [ ! -d "$SOURCE_PATH/sparse/0" ] || [ ! -d "$SOURCE_PATH/images" ]; then
    echo "❌ ERROR: COLMAP 场景不完整: $SOURCE_PATH" >&2
    echo "       先跑 deformable_human/01_prepare_data.sh 产出场景（本目录复用其产物）" >&2
    exit 1
fi
if [ "$SKIP_VERIFY" != "1" ]; then
    if ! python -c "import diff_gaussian_rasterization, simple_knn" 2>/dev/null; then
        echo "❌ ERROR: CUDA 扩展不可 import。bash som_human/00a_setup_env.sh 体检" >&2
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
[ "$WHITE_BG" = "1" ] && TRAIN_FLAGS+=(--white_background)
# shellcheck disable=SC2206
[ -n "${TEST_ITERATIONS:-}" ] && TRAIN_FLAGS+=(--test_iterations $TEST_ITERATIONS)
# shellcheck disable=SC2206
[ -n "${SAVE_ITERATIONS:-}" ] && TRAIN_FLAGS+=($SAVE_ITERATIONS)
# shellcheck disable=SC2086
[ -n "$EXTRA_TRAIN_ARGS" ] && TRAIN_FLAGS+=($EXTRA_TRAIN_ARGS)

echo "🏋️ train_som.py ${TRAIN_FLAGS[*]}"
echo ""

export DG_DIR SOM_NUM_BASES="$NUM_BASES" SOM_SMOOTH_WEIGHT="$SMOOTH_WEIGHT" \
       SOM_BASIS_LR="$BASIS_LR" SOM_COEF_LR="$COEF_LR" \
       SOM_RES_SCHEDULE="$RES_SCHEDULE" SOM_RES_COMPENSATE_DENSIFY="$RES_COMPENSATE_DENSIFY" \
       SOM_MAX_GAUSSIANS="$MAX_GAUSSIANS" \
       SOM_DENSIFY_SCHED="$USE_DENSIFY_SCHED" SOM_DENSIFY_MOMENTUM_MULT="$DENSIFY_MOMENTUM_MULT" \
       SOM_DENSIFY_MAX_RATE="$DENSIFY_MAX_RATE" SOM_DENSIFY_CNT_CLAMP="$DENSIFY_CNT_CLAMP" \
       SOM_DENSIFY_PRUNE_AFTER="$DENSIFY_PRUNE_AFTER" \
       SOM_EXPOSURE="$USE_EXPOSURE" SOM_EXPOSURE_FROM_ITER="$EXPOSURE_FROM_ITER" \
       SOM_EXPOSURE_LR="$EXPOSURE_LR" SOM_EXPOSURE_REG="$EXPOSURE_REG" \
       SOM_DEPTH_LAMBDA="$DEPTH_LAMBDA" SOM_DEPTH_FROM_ITER="$DEPTH_FROM_ITER" \
       SOM_DEPTH_USE_MASK="$DEPTH_USE_MASK" SOM_DEPTH_MIN_POINTS="$DEPTH_MIN_POINTS" \
       SOM_FG_MASK_DIR="$FG_MASK_DIR" SOM_FG_INIT_FILTER="$FG_INIT_FILTER" \
       SOM_FG_INIT_MIN_VIEWS="$FG_INIT_MIN_VIEWS" SOM_FG_INIT_MIN_RATIO="$FG_INIT_MIN_RATIO" \
       SOM_BG_RANDOM="$BG_RANDOM" \
       SOM_HAND_MASK_DIR="$HAND_MASK_DIR" SOM_HAND_WEIGHT="$HAND_WEIGHT"
# train_som.py 用官方仓的相对 import → 必须在 $DG_DIR 里跑
( cd "$DG_DIR" && python "$SCRIPT_DIR/train_som.py" "${TRAIN_FLAGS[@]}" )
if [ $? -ne 0 ]; then
    echo "❌ FAILED: train_som.py 没跑完。常见原因:" >&2
    echo "    - OOM: 减 ITERATIONS / 01 里 NUM_IMAGES_MAX 截帧 / 带降载 EXTRA_TRAIN_ARGS" >&2
    echo "    - fid ValueError: 图像名不是纯数字（deformable_human 01 应已重命名）" >&2
    echo "    - num_frames 断言: 渲染/训练场景相机数不一致（见 som_motion.load_weights）" >&2
    exit 1
fi

LAST_ITER="$(ls -d "$MODEL_PATH/point_cloud/iteration_"* 2>/dev/null | sort -V | tail -1 | sed 's/.*iteration_//')"
echo ""
echo "🎉 [02] Done."
echo "  🏋️ 高斯点云:  $MODEL_PATH/point_cloud/iteration_${LAST_ITER:-$ITERATIONS}/point_cloud.ply"
echo "  🧬 运动基:     $MODEL_PATH/som_motion/iteration_${LAST_ITER:-$ITERATIONS}/som.pth"
echo "  📊 TensorBoard: tensorboard --logdir $MODEL_PATH --port 6006"
echo "  → 渲染评测: GPU=0 SCENE_NAME=$SCENE_NAME bash som_human/03_render.sh"
