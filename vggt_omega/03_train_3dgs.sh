#!/usr/bin/env bash
# 03_train_3dgs.sh — 传统 3DGS 训练（转发 vggt_human/04，官方 gaussian-splatting）
# 前置: 同场景已跑过 02（$RESULTS_DIR/<场景名>/03_source 就绪）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

export RAW_DIR="${RAW_DIR:?❌ 请指定采集包路径 RAW_DIR=（与 02 相同）}"
export SCENE_NAME="${SCENE_NAME:-$(basename "$RAW_DIR" | sed 's/^[0-9a-f]\{32\}-//' | tr -d '[:space:]')}"
export ITERATIONS="${ITERATIONS:-30000}"
export RES="${RES:-2}"   # 3DGS 训练分辨率因子；som_human 口径全分辨率传 RES= 空

SCENE_RESULTS="$RESULTS_DIR/$SCENE_NAME"
SOURCE_DIR="$SCENE_RESULTS/03_source"
GAUSSIAN_DIR="$SCENE_RESULTS/04_model_3dgs"

if [ ! -d "$SOURCE_DIR/images" ] || [ ! -d "$SOURCE_DIR/sparse/0" ]; then
    echo "❌ ERROR: COLMAP 场景未就绪: $SOURCE_DIR（先跑 02）" >&2
    exit 1
fi

echo "🚀 [03] 3DGS 训练: $SCENE_NAME（$ITERATIONS iters）"
CONDA_ENV=vggt_human \
MODEL_DIR="$MODEL_DIR" \
SOURCE_DIR="$SOURCE_DIR" \
GAUSSIAN_DIR="$GAUSSIAN_DIR" \
ITERATIONS="$ITERATIONS" RES="$RES" \
bash "$MEDIA_CODE_ROOT/vggt_human/04_train_3dgs.sh"
if [ $? -ne 0 ]; then echo "❌ FAILED" >&2; exit 1; fi

echo "🎉 Done. 高斯模型: $GAUSSIAN_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
