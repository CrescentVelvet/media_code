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
# 02 的 DA3 export_to_colmap 把 cameras/images/points3D.bin 平铺在场景根，且没有
# images/ 目录与 sparse/0/ 布局——先用 da3_human 的组装脚本整理成原版 3DGS 可读场景
SCENE_DIR="$SCENE_RESULTS/colmap_scene"
SOURCE_DIR="${SOURCE_DIR:-$SCENE_DIR}"
GAUSSIAN_DIR="$SCENE_RESULTS/04_model_3dgs"
IMAGES_DIR="$RESULTS_DIR/_captures/$SCENE_NAME/images"

if [ ! -f "$SCENE_RESULTS/images.bin" ]; then
    echo "❌ ERROR: DA3 colmap 导出不完整: $SCENE_RESULTS/images.bin 不存在（先跑 02）" >&2
    exit 1
fi
if [ ! -d "$SOURCE_DIR/images" ] || [ ! -d "$SOURCE_DIR/sparse/0" ]; then
    echo "🧩 组装 COLMAP 场景: $SOURCE_DIR（images 软链 + sparse/0 + 点云降采样）"
    ASSEMBLE_PYTHON="${ASSEMBLE_PYTHON:-$HOME/miniconda3/envs/da3_human/bin/python}"
    [ -x "$ASSEMBLE_PYTHON" ] || ASSEMBLE_PYTHON=python
    DA3_INFER_DIR="$SCENE_RESULTS" \
    SCENE_DIR="$SOURCE_DIR" \
    IMAGES_DIR="$IMAGES_DIR" \
    "$ASSEMBLE_PYTHON" "$MEDIA_CODE_ROOT/da3_human/build_da3_colmap_scene.py"
    if [ $? -ne 0 ]; then echo "❌ 场景组装失败" >&2; exit 1; fi
fi
if [ ! -d "$SOURCE_DIR/images" ] || [ ! -d "$SOURCE_DIR/sparse/0" ]; then
    echo "❌ ERROR: COLMAP 场景未就绪: $SOURCE_DIR" >&2
    exit 1
fi

echo "🚀 [03] 3DGS 训练: $SCENE_NAME（$ITERATIONS iters，DA3 点云初始化）"
# 注意必须显式传 CONDA_ENV=vggt_human：本仓 _env.sh 导出 CONDA_ENV=da3，
# 会泄进 04 让它去激活不存在的 da3 env（CUDA 扩展 import 失败）
CONDA_ENV=vggt_human \
MODEL_DIR="$DA3_MODEL_DIR" \
SOURCE_DIR="$SOURCE_DIR" \
GAUSSIAN_DIR="$GAUSSIAN_DIR" \
ITERATIONS="$ITERATIONS" RES="$RES" \
bash "$MEDIA_CODE_ROOT/vggt_human/04_train_3dgs.sh"
if [ $? -ne 0 ]; then echo "❌ FAILED" >&2; exit 1; fi

echo "🎉 Done. 高斯模型: $GAUSSIAN_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
