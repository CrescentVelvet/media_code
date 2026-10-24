#!/usr/bin/env bash
# 03_render.sh — 渲染。模式（官方 render.sh）：
#   normal              时间正常流逝的视频
#   fixed-time-view     冻结单时刻（本方法核心：全场景钉在 t*）
#   time-sweeping-view  固定视角扫时间
#   fixed-time-smooth   冻结时刻 + 插值平滑轨迹
#
# TIME_INDEX 为空 = 每 8 帧一个时刻全部渲染；或逗号分隔如 "0,1,2,3,4"。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

SCENE_DIR="${SCENE_DIR:?❌ SCENE_DIR not set}"
SCENE_NAME="${SCENE_NAME:-$(basename "$SCENE_DIR")}"
if [ "${BASELINE:-0}" = "1" ]; then TRAIN_MODE="baseline"; else TRAIN_MODE="splannequin"; fi
MODEL_DIR_PATH="${MODEL_DIR_PATH:-$EXP_DIR/$TRAIN_MODE/$SCENE_NAME}"
TIME_INDEX="${TIME_INDEX:-}"
RENDER_MODES="${RENDER_MODES:-normal fixed-time-view}"

if [ ! -d "$MODEL_DIR_PATH" ]; then
    echo "❌ ERROR: model dir not found: $MODEL_DIR_PATH" >&2
    echo "       Run 02_train.sh first, or set MODEL_DIR_PATH." >&2
    exit 1
fi

echo "🚀 [03] Rendering"
echo "  🏋️ model: $MODEL_DIR_PATH"
echo "  🎬 modes: $RENDER_MODES  time_index: ${TIME_INDEX:-all(every 8th)}"
export PYTORCH_ALLOC_CONF="max_split_size_mb:512"

cd "$SPLANNEQUIN_DIR"
python render.py \
    --model_path "$MODEL_DIR_PATH" \
    --time_index "$TIME_INDEX" \
    --render_modes $RENDER_MODES \
    --configs "./arguments/${TRAIN_MODE}.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 Done. Renders under $MODEL_DIR_PATH"
