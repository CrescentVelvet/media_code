#!/usr/bin/env bash
# 06_train_da3_2dgs.sh — 阶段 C 核心训练：anchor + 有界 offset 双分支 2DGS
# 运行 env：vggt_human（gsplat JIT 需 nvcc+cu121 工具链，da3_human env 无 nvcc，见 06 脚本头注）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

export GS_CONDA_ENV="${GS_CONDA_ENV:-vggt_human}"
export ANCHOR_DIR="${ANCHOR_DIR:-$RESULTS_DIR/anchor_pointcloud/studio_human}"
export DA3_INFER_DIR="${DA3_INFER_DIR:-$RESULTS_DIR/da3_inference/studio_human}"
export IMAGES_DIR="${IMAGES_DIR:-$HOME/output/vggt_human_results/source/images}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/da3_2dgs_dual/ours}"
# 常用开关（默认值在 06 脚本内）：
#   OFFSET_PIX=1.0 OFFSET_DEPTH=0.02  ITERATIONS=30000 SH_DEGREE=3 GEO_RES=4
#   LAMBDA_DIST=0.1 LAMBDA_NORM=0.1   PRUNE_OPACITY=0  SMOKE=1(冒烟)

[ -f "$ANCHOR_DIR/anchors.npz" ] || { echo "❌ ERROR: 缺 anchors.npz，先跑 05" >&2; exit 1; }
[ -f "$DA3_INFER_DIR/exports/mini_npz/results.npz" ] || { echo "❌ ERROR: 缺 DA3 导出，先跑 02" >&2; exit 1; }
[ -d "$IMAGES_DIR" ] || { echo "❌ ERROR: 图像目录不存在: $IMAGES_DIR" >&2; exit 1; }

# 训练切 vggt_human env（gsplat 在这里 JIT 编译并验证过）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$GS_CONDA_ENV" || { echo "❌ ERROR: env $GS_CONDA_ENV" >&2; exit 1; }
python -c "import gsplat, torch" 2>/dev/null || { echo "❌ ERROR: gsplat 未装（vggt_human env）" >&2; exit 1; }

echo "🚀 [06] 双分支 2DGS 训练（env: $GS_CONDA_ENV）"
echo "  📦 anchors: $ANCHOR_DIR"
echo "  📁 输出: $OUTPUT_DIR"
echo "  🎮 GPU: ${GPU:-0}"
python "$SCRIPT_DIR/06_train_da3_2dgs.py"
[ $? -ne 0 ] && { echo "❌ FAILED" >&2; exit 1; }
echo "🎉 Done."
