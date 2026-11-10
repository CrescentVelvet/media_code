#!/usr/bin/env bash
# 90c_train_da3_3dgs.sh — 横评 DA3 基线：传统 3DGS 训练（DA3 点云初始化，转发 vggt_human/04）
# 收编自原 da3/03_train_3dgs.sh（2026-11-10 撤销 da3/ 目录；历史见 tag da3-baseline-20261109 及更早）。
# 前置: 同场景已跑过 90b（$RESULTS_DIR/<场景名>/ 下有 DA3 colmap 导出）。
# DA3 的 colmap 产物目录布局以 90b 实际输出为准（images/ + sparse/0/）；
# 若顶层布局不同，用 SOURCE_DIR 显式指到含 images/ + sparse/0/ 的那层。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA_CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${DA3_RESULTS_DIR:-$HOME/output/da3_results}"
# 与 90b 同口径；04_train_3dgs 只把它当 MODEL_DIR 透传，权重本体仍走 HF 缓存/本地目录
DA3_MODEL_DIR="${DA3_MODEL_DIR:-${DA3_HUMAN_MODEL_DIR:-$HOME/model/da3_human}}"

export RAW_DIR="${RAW_DIR:?❌ 请指定采集包路径 RAW_DIR=（与 90b 相同）}"
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
    echo "❌ ERROR: DA3 colmap 导出不完整: $SCENE_RESULTS/images.bin 不存在（先跑 90b）" >&2
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

echo "🚀 [90c] 3DGS 训练: $SCENE_NAME（$ITERATIONS iters，DA3 点云初始化）"
# 注意必须显式传 CONDA_ENV=vggt_human：否则 _env 默认 env 泄进 04，
# 会激活不存在的 env（CUDA 扩展 import 失败）——原 da3/_env.sh 的坑，收编时保留此防线
CONDA_ENV=vggt_human \
MODEL_DIR="$DA3_MODEL_DIR" \
SOURCE_DIR="$SOURCE_DIR" \
GAUSSIAN_DIR="$GAUSSIAN_DIR" \
ITERATIONS="$ITERATIONS" RES="$RES" \
bash "$MEDIA_CODE_ROOT/vggt_human/04_train_3dgs.sh"
if [ $? -ne 0 ]; then echo "❌ FAILED" >&2; exit 1; fi

echo "🎉 Done. 高斯模型: $GAUSSIAN_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
