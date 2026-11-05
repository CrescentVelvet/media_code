#!/usr/bin/env bash
# 04_run_da3_gs_baseline.sh — DA3 点云初始化 3DGS（对照组 2）
# 数据：02 的 DA3 colmap 导出（DA3 自估位姿 + 深度反投影点云）整理成 COLMAP 场景后训练
# 用法: bash da3_human/04_run_da3_gs_baseline.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

export DA3_INFER_DIR="${DA3_INFER_DIR:-$RESULTS_DIR/da3_inference/studio_human}"
export SCENE_DIR="${SCENE_DIR:-$RESULTS_DIR/da3_colmap_scene}"
export IMAGES_DIR="${IMAGES_DIR:-$HOME/output/vggt_human_results/source/images}"
export GAUSSIAN_DIR="${GAUSSIAN_DIR:-$RESULTS_DIR/baseline_3dgs_da3init}"

if [ ! -f "$DA3_INFER_DIR/images.bin" ]; then
    echo "❌ ERROR: DA3 colmap 导出不存在: $DA3_INFER_DIR/images.bin" >&2
    echo "       先跑: bash $SCRIPT_DIR/02_run_da3_inference.sh" >&2
    exit 1
fi

echo "🚀 [04] DA3-GS 基线 (DA3 点云初始化)"
echo "  📂 DA3 导出: $DA3_INFER_DIR"
echo "  📂 场景目录: $SCENE_DIR（images 软链 + sparse/0 搬运）"
echo "  ✂️ MAX_INIT_POINTS: ${MAX_INIT_POINTS:-1000000}（DA3 原始点云 1429 万，超限体素降采样）"

export MAX_INIT_POINTS="${MAX_INIT_POINTS:-1000000}"
python "$SCRIPT_DIR/build_da3_colmap_scene.py"
if [ $? -ne 0 ]; then echo "❌ FAILED: 场景整理" >&2; exit 1; fi

export SOURCE_DIR="$SCENE_DIR"
source "$SCRIPT_DIR/_train_3dgs.sh"
if [ $? -ne 0 ]; then echo "❌ FAILED" >&2; exit 1; fi
echo "🎉 Done. 产物: $GAUSSIAN_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
