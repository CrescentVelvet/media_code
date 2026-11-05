#!/usr/bin/env bash
# 03_train_3dgs.sh — 传统 3DGS 训练（DA3 点云初始化，转发 vggt_human/04）
# 前置: 同场景已跑过 02（$RESULTS_DIR/<场景名>/colmap 就绪）
# DA3 的 colmap 产物目录布局以 02 实际输出为准（images/ + sparse/0/）；
# 若顶层布局不同，用 SOURCE_DIR 显式指到含 images/ + sparse/0/ 的那层。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

export RAW_DIR="${RAW_DIR:?❌ 请指定采集包路径 RAW_DIR=（与 02 相同）}"
export SCENE_NAME="${SCENE_NAME:-$(basename "$RAW_DIR" | sed 's/^[0-9a-f]\{32\}-//' | tr -d '[:space:]')}"
export ITERATIONS="${ITERATIONS:-30000}"
export RES="${RES:-2}"   # 3DGS 训练分辨率因子；要全分辨率传 RES= 空

SCENE_RESULTS="$RESULTS_DIR/$SCENE_NAME"
SOURCE_DIR="${SOURCE_DIR:-$SCENE_RESULTS/colmap}"
GAUSSIAN_DIR="$SCENE_RESULTS/04_model_3dgs"

if [ ! -d "$SOURCE_DIR/images" ] || [ ! -d "$SOURCE_DIR/sparse/0" ]; then
    echo "❌ ERROR: COLMAP 场景未就绪: $SOURCE_DIR（先跑 02）" >&2
    echo "       若 02 产物布局不同，用 SOURCE_DIR=<实际路径> 显式指定" >&2
    exit 1
fi

echo "🚀 [03] 3DGS 训练: $SCENE_NAME（$ITERATIONS iters，DA3 点云初始化）"
MODEL_DIR="$DA3_MODEL_DIR" \
SOURCE_DIR="$SOURCE_DIR" \
GAUSSIAN_DIR="$GAUSSIAN_DIR" \
ITERATIONS="$ITERATIONS" RES="$RES" \
bash "$MEDIA_CODE_ROOT/vggt_human/04_train_3dgs.sh"
if [ $? -ne 0 ]; then echo "❌ FAILED" >&2; exit 1; fi

echo "🎉 Done. 高斯模型: $GAUSSIAN_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
