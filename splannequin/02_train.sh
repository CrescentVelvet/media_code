#!/usr/bin/env bash
# 02_train.sh — 训练 splannequin（双检测时间锚定正则）动态高斯。
#
# 默认 splannequin 模式（带 hidden/defective 锚定）；
# BASELINE=1 跑 4DGaussians 基线（arguments/baseline.py，用于消融对照）。
# CONFIG_FILE 可覆盖配置（如 arguments_splannequin_15k.py 加速方案 A）。
#
# 断点续跑：官方代码遇 NaN 会 execv 从零重跑（进度全丢）；00c 补丁改为
# exit(42)。本脚本每 CKPT_EVERY iter 存 checkpoint，非零退出时自动从
# 最近 checkpoint 恢复（最多 MAX_RETRIES 次）。
#
# 输出：$EXP_DIR/<mode>/<scene_name>/（官方默认写仓内 ./exp，这里外置）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

SCENE_DIR="${SCENE_DIR:?❌ SCENE_DIR not set (dir with images/ + sparse/, run 01 first)}"
SCENE_NAME="${SCENE_NAME:-$(basename "$SCENE_DIR")}"
if [ "${BASELINE:-0}" = "1" ]; then TRAIN_MODE="baseline"; else TRAIN_MODE="splannequin"; fi
# CONFIG_FILE：覆盖默认配置（如加速方案 A 的半量 schedule，须为 WSL 内可读路径）
CONFIG_FILE="${CONFIG_FILE:-./arguments/${TRAIN_MODE}.py}"
PORT="${PORT:-6017}"
CKPT_EVERY="${CKPT_EVERY:-1500}"
# 崩溃是随机性的（同一窗口重过大概率能过），允许同一 checkpoint 重试
SAME_CKPT_LIMIT="${SAME_CKPT_LIMIT:-3}"
MAX_RETRIES="${MAX_RETRIES:-12}"

if [ ! -d "$SCENE_DIR/sparse" ]; then
    echo "❌ ERROR: $SCENE_DIR/sparse not found — run 01_colmap.sh first" >&2
    exit 1
fi

OUT_DIR="$EXP_DIR/$TRAIN_MODE/$SCENE_NAME"
mkdir -p "$OUT_DIR"

echo "🚀 [02] Training ($TRAIN_MODE)"
echo "  📁 scene:  $SCENE_DIR"
echo "  💾 output: $OUT_DIR"
echo "  🎮 GPU:    ${CUDA_VISIBLE_DEVICES:-auto}"
export PYTORCH_ALLOC_CONF="max_split_size_mb:512"

cd "$SPLANNEQUIN_DIR"
# 覆盖到官方 30k 上限，超出实际 iterations 的项自动不触发
CKPT_LIST=$(seq "$CKPT_EVERY" "$CKPT_EVERY" 30000 | tr '\n' ' ')

attempt=0
same_ckpt_count=0
START_CKPT=""
while :; do
    RESUME_ARGS=""
    [ -n "$START_CKPT" ] && RESUME_ARGS="--start_checkpoint $START_CKPT"
    python train.py \
        -s "$SCENE_DIR" \
        -m "$OUT_DIR" \
        --port "$PORT" \
        --eval \
        --train_mode "$TRAIN_MODE" \
        --resolution "${RESOLUTION:-1}" \
        --expname "${SCENE_NAME}_${TRAIN_MODE}" \
        --configs "$CONFIG_FILE" \
        --checkpoint_iterations $CKPT_LIST \
        $RESUME_ARGS $EXTRA_TRAIN_ARGS
    rc=$?
    [ $rc -eq 0 ] && break
    attempt=$((attempt + 1))
    LATEST=$(ls -t "$OUT_DIR"/chkpnt_*.pth 2>/dev/null | head -1)
    if [ -z "$LATEST" ]; then
        echo "❌ FAILED (rc=$rc, 无 checkpoint 可续)" >&2
        exit 1
    fi
    if [ "$LATEST" = "$START_CKPT" ]; then
        same_ckpt_count=$((same_ckpt_count + 1))
    else
        same_ckpt_count=0
    fi
    if [ $attempt -gt "$MAX_RETRIES" ] || [ $same_ckpt_count -ge "$SAME_CKPT_LIMIT" ]; then
        echo "❌ FAILED (rc=$rc, attempt=$attempt, 同一 ckpt 连崩 $same_ckpt_count 次, ckpt=$LATEST)" >&2
        exit 1
    fi
    START_CKPT="$LATEST"
    echo "⚠️ train.py rc=$rc — 从 $START_CKPT 续跑（第 $attempt/$MAX_RETRIES 次，同 ckpt 第 $same_ckpt_count/$SAME_CKPT_LIMIT 次）"
    sleep 5
done
echo "🎉 Done. Model at $OUT_DIR"
echo "  Next: SCENE_DIR=$SCENE_DIR bash $SCRIPT_DIR/03_render.sh"
