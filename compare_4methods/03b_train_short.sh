#!/usr/bin/env bash
# 03b_train_short.sh — 短精调：前馈高斯 init（或 COLMAP 点云 init）+ N 步训练
#
# 为什么需要这个脚本（而不是直接跑 vggt_human/04）：
#   原版 30k 配方的两个调度是按 30k 步设计的。只把 --iterations 砍到 5k 而不动调度，
#   位置学习率会一直停在高位（衰减曲线还没走完），结果既不能代表"该方法 5k 步的收敛
#   上限"，也会把前馈给的好初值抖坏。所以这里默认把整条调度等比压缩
#   （COMPRESS_SCHEDULE=1），另一组 COMPRESS_SCHEDULE=0 作对照，专门隔离这件事。
#
# Env:
#   SCENE_DIR       COLMAP 场景目录（必须含 images/ 与 sparse/0/）
#   OUT_DIR         训练输出目录
#   INIT_CKPT       可选，3DGS 格式 .pth（98b 产出）。不给则走 COLMAP 点云初始化
#   ITERATIONS      默认 5000
#   RES             分辨率因子，默认 2（与 30k 基线一致：1439x1919 → 720x960）
#   COMPRESS_SCHEDULE 默认 1：把 30k 的 lr/密度调度等比压到 ITERATIONS
#   POSITION_LR_SCALE 默认 1：位置 lr 的整体系数。前馈 init 的位置已经很准
#                   （DA3 的 scale 中位数只有 0.8 mm），官方 1.6e-4×extent 的位置 lr
#                   会在几百步内把细节抖散 —— 精调要先把它按 0.1 / 0.01 降下来试。
#   LR_MAX_STEPS    可选，显式指定位置 lr 的衰减步数（不给则按 COMPRESS_SCHEDULE 推）
#   DENSIFY         默认 0：关掉自适应密度控制（前馈 init 已经很密）。
#                   1 = 开启并等比压缩，2 = 开启且用原 30k 区间
#   RENDER          默认 1：训练后渲染训练视角
#   GS_DIR          默认 ~/repos/gaussian-splatting
#   CONDA_ENV       默认 vggt_human
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GS_DIR="${GS_DIR:-$HOME/repos/gaussian-splatting}"
CONDA_ENV="${CONDA_ENV:-vggt_human}"
SCENE_DIR="${SCENE_DIR:?❌ 需要 SCENE_DIR=<COLMAP 场景>}"
OUT_DIR="${OUT_DIR:?❌ 需要 OUT_DIR=<输出目录>}"
ITERATIONS="${ITERATIONS:-5000}"
RES="${RES:-2}"
INIT_CKPT="${INIT_CKPT:-}"
COMPRESS_SCHEDULE="${COMPRESS_SCHEDULE:-1}"
DENSIFY="${DENSIFY:-0}"
RENDER="${RENDER:-1}"

if [ ! -f "$GS_DIR/train.py" ]; then
    echo "❌ 找不到 $GS_DIR/train.py" >&2; exit 1
fi
if [ ! -d "$SCENE_DIR/images" ] || [ ! -d "$SCENE_DIR/sparse/0" ]; then
    echo "❌ COLMAP 场景不完整: $SCENE_DIR（需 images/ + sparse/0/）" >&2; exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    # 非交互 shell 里 conda 通常不在 PATH 上，按常见安装位置兜底
    for _cb in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -f "$_cb/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$_cb/etc/profile.d/conda.sh"
            break
        fi
    done
    unset _cb
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "❌ 找不到 conda（需要 env '$CONDA_ENV'）" >&2; exit 1
fi
conda activate "$CONDA_ENV" || { echo "❌ conda env $CONDA_ENV 不存在" >&2; exit 1; }
PY="$(command -v python || true)"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    PY="$CONDA_PREFIX/bin/python"
fi
if [ ! -x "$PY" ]; then
    echo "❌ env $CONDA_ENV 里没有 python" >&2; exit 1
fi
echo "🐍 $PY  ($CONDA_PREFIX)"

mkdir -p "$OUT_DIR"

# ── 调度参数 ───────────────────────────────────────────────────────────────
# 3DGS 默认：position_lr_max_steps=30000, densify_from=500, densify_until=15000,
#            opacity_reset=3000, densification_interval=100
if [ "$COMPRESS_SCHEDULE" = "1" ]; then
    LR_MAX_STEPS="${LR_MAX_STEPS:-$ITERATIONS}"
    DENSIFY_FROM=$(( ITERATIONS / 60 > 1 ? ITERATIONS / 60 : 1 ))     # ≈500/30000
    DENSIFY_UNTIL=$(( ITERATIONS / 2 ))                                # ≈15000/30000
    OPACITY_RESET=$(( ITERATIONS / 10 ))                               # ≈3000/30000
else
    LR_MAX_STEPS="${LR_MAX_STEPS:-30000}"
    DENSIFY_FROM=500
    DENSIFY_UNTIL=15000
    OPACITY_RESET=3000
fi
POSITION_LR_SCALE="${POSITION_LR_SCALE:-1}"
# 3DGS 默认 position_lr_init=0.00016 / final=0.0000016（两者同比例缩放）
LR_INIT=$("$PY" -c "print(${POSITION_LR_SCALE} * 0.00016)")
LR_FINAL=$("$PY" -c "print(${POSITION_LR_SCALE} * 0.0000016)")
case "$DENSIFY" in
    0) DENSIFY_ARGS=(--densify_until_iter 0) ;;
    1) DENSIFY_ARGS=(--densify_from_iter "$DENSIFY_FROM" --densify_until_iter "$DENSIFY_UNTIL"
                     --opacity_reset_interval "$OPACITY_RESET") ;;
    *) DENSIFY_ARGS=(--densify_from_iter 500 --densify_until_iter 15000
                     --opacity_reset_interval 3000) ;;
esac

FLAGS=(
    -s "$SCENE_DIR" -m "$OUT_DIR"
    --iterations "$ITERATIONS"
    --resolution "$RES"
    --position_lr_max_steps "$LR_MAX_STEPS"
    --position_lr_init "$LR_INIT"
    --position_lr_final "$LR_FINAL"
    --test_iterations 99999999
    --save_iterations "$ITERATIONS"
    --port 0 --disable_viewer
    "${DENSIFY_ARGS[@]}"
)
[ -n "$INIT_CKPT" ] && FLAGS+=(--start_checkpoint "$INIT_CKPT")

echo "🏋️ 短精调: $(basename "$SCENE_DIR") → $(basename "$OUT_DIR")"
echo "   init:      ${INIT_CKPT:-COLMAP 点云 (create_from_pcd)}"
echo "   iters=$ITERATIONS  res=$RES  compress_schedule=$COMPRESS_SCHEDULE  densify=$DENSIFY"
echo "   lr_max_steps=$LR_MAX_STEPS  position_lr=$LR_INIT→$LR_FINAL (x$POSITION_LR_SCALE)  densify=[$DENSIFY_FROM,$DENSIFY_UNTIL]  opacity_reset=$OPACITY_RESET"

# torch>=2.6 的 torch.load 默认 weights_only=True；我们的 ckpt 含 optimizer state_dict
# （纯 tensor + 内建类型），正常可加载。留这个开关是为了万一报
# "Weights only load failed" 时不用改代码就能绕过。
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

T0=$(date +%s)
( cd "$GS_DIR" && "$PY" train.py "${FLAGS[@]}" )
RC=$?
T1=$(date +%s)
TRAIN_SEC=$(( T1 - T0 ))
if [ $RC -ne 0 ]; then echo "❌ 训练失败 (rc=$RC)" >&2; exit 1; fi
echo "✅ 训练完成 ${TRAIN_SEC}s"

PLY="$OUT_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
if [ -f "$PLY" ]; then
    echo "💾 $PLY  ($(du -h "$PLY" | cut -f1))"
else
    echo "⚠️ 未找到 $PLY" >&2
fi

# ── 渲染训练视角 ───────────────────────────────────────────────────────────
RENDER_SEC=0
if [ "$RENDER" = "1" ]; then
    T0=$(date +%s)
    ( cd "$GS_DIR" && "$PY" render.py -s "$SCENE_DIR" -m "$OUT_DIR" \
        --iteration "$ITERATIONS" --resolution "$RES" 2>&1 | tail -3 )
    T1=$(date +%s)
    RENDER_SEC=$(( T1 - T0 ))
    echo "✅ 渲染完成 ${RENDER_SEC}s → $OUT_DIR/train/ours_$ITERATIONS/"
fi

cat > "$OUT_DIR/short_train_summary.json" <<EOF
{
  "scene_dir": "$SCENE_DIR",
  "out_dir": "$OUT_DIR",
  "init_ckpt": "${INIT_CKPT:-}",
  "init_kind": "$([ -n "$INIT_CKPT" ] && echo feedforward || echo colmap_pcd)",
  "iterations": $ITERATIONS,
  "resolution": $RES,
  "compress_schedule": $COMPRESS_SCHEDULE,
  "densify": $DENSIFY,
  "lr_max_steps": $LR_MAX_STEPS,
  "position_lr_scale": $POSITION_LR_SCALE,
  "position_lr_init": $LR_INIT,
  "position_lr_final": $LR_FINAL,
  "densify_from": $DENSIFY_FROM,
  "densify_until": $DENSIFY_UNTIL,
  "opacity_reset_interval": $OPACITY_RESET,
  "train_seconds": $TRAIN_SEC,
  "render_seconds": $RENDER_SEC
}
EOF
echo "📋 $OUT_DIR/short_train_summary.json"
