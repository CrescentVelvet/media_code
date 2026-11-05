#!/usr/bin/env bash
# 04b_run_da3_gs_colmapintr.sh — B-04b 假设1隔离实验
# 变量：04 的 DA3 点云 + DA3 位姿不变，仅每相机内参替换为 COLMAP 标定值
# 判定：若 PSNR 明显回升 → 主因是 DA3 预测内参偏差；若不回升 → 主因是深度噪声/densify 膨胀
# 用法: bash da3_human/04b_run_da3_gs_colmapintr.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

export DA3_INFER_DIR="${DA3_INFER_DIR:-$RESULTS_DIR/da3_inference/studio_human}"
export SCENE_DIR="${SCENE_DIR:-$RESULTS_DIR/da3_colmap_scene_colmapintr}"
export IMAGES_DIR="${IMAGES_DIR:-$HOME/output/vggt_human_results/source/images}"
export GAUSSIAN_DIR="${GAUSSIAN_DIR:-$RESULTS_DIR/baseline_3dgs_da3init_colmapintr}"
export INTR_REF_DIR="${INTR_REF_DIR:-$HOME/output/vggt_human_results/source/sparse/0}"

if [ ! -f "$DA3_INFER_DIR/images.bin" ]; then
    echo "❌ ERROR: DA3 colmap 导出不存在: $DA3_INFER_DIR（先跑 02）" >&2
    exit 1
fi
if [ ! -f "$INTR_REF_DIR/cameras.bin" ] && [ ! -f "$INTR_REF_DIR/cameras.txt" ]; then
    echo "❌ ERROR: 参考 COLMAP 模型不存在: $INTR_REF_DIR（需要 cameras.bin|txt）" >&2
    exit 1
fi

echo "🚀 [04b] DA3-GS + COLMAP 内参（假设1隔离）"
echo "  📂 DA3 导出: $DA3_INFER_DIR（点云/位姿 = 04 相同）"
echo "  🔧 内参来源: $INTR_REF_DIR（按图像名匹配替换）"
echo "  💾 输出:     $GAUSSIAN_DIR"

export MAX_INIT_POINTS="${MAX_INIT_POINTS:-500000}"
python "$SCRIPT_DIR/build_da3_colmap_scene.py"
if [ $? -ne 0 ]; then echo "❌ FAILED: 场景整理" >&2; exit 1; fi

export SOURCE_DIR="$SCENE_DIR"
source "$SCRIPT_DIR/_train_3dgs.sh"
if [ $? -ne 0 ]; then echo "❌ FAILED" >&2; exit 1; fi
echo "🎉 Done. 产物: $GAUSSIAN_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
