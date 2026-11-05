#!/usr/bin/env bash
# 05_build_anchor_pointcloud.sh — 阶段 C [锚定]：DA3 深度反投影 anchor 点云
# 筛选：多视角深度一致性（reproj 深度差 > τ 剔除）+ 跨视角体素去重
# 用法: bash da3_human/05_build_anchor_pointcloud.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

export DA3_INFER_DIR="${DA3_INFER_DIR:-$RESULTS_DIR/da3_inference/studio_human}"
export ANCHOR_DIR="${ANCHOR_DIR:-$RESULTS_DIR/anchor_pointcloud/studio_human}"
export IMAGES_DIR="${IMAGES_DIR:-$HOME/output/vggt_human_results/source/images}"
# 筛选超参（计划阶段 C；D 阶段消融可扫）
export ANCHOR_CONF_PCT="${ANCHOR_CONF_PCT:-40}"          # conf 百分位阈值
export ANCHOR_TAU_REL="${ANCHOR_TAU_REL:-0.03}"          # 邻域深度一致相对阈值
export ANCHOR_TAU_ABS="${ANCHOR_TAU_ABS:-0.05}"          # 绝对下限（米）
export ANCHOR_NEIGHBOR_VIEWS="${ANCHOR_NEIGHBOR_VIEWS:-8}"   # 最近邻检查视角数
export ANCHOR_MIN_CONSISTENT="${ANCHOR_MIN_CONSISTENT:-3}"   # 至少一致视角数
export ANCHOR_VOXEL="${ANCHOR_VOXEL:-0}"                 # 去重体素（米）；0 = 0.004×深度中值

if [ ! -f "$DA3_INFER_DIR/exports/mini_npz/results.npz" ] && [ ! -f "$DA3_INFER_DIR/exports/npz/results.npz" ]; then
    echo "❌ ERROR: DA3 npz 导出不存在: $DA3_INFER_DIR（先跑 02）" >&2
    exit 1
fi

echo "🚀 [05] anchor 点云构建"
echo "  📥 DA3 导出: $DA3_INFER_DIR"
echo "  💾 anchor:   $ANCHOR_DIR"
echo "  🔍 筛选: conf≥P$ANCHOR_CONF_PCT, τ_rel=$ANCHOR_TAU_REL, τ_abs=$ANCHOR_TAU_ABS,"
echo "            邻域 $ANCHOR_NEIGHBOR_VIEWS 视角 / 至少一致 $ANCHOR_MIN_CONSISTENT, voxel=${ANCHOR_VOXEL:-auto}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate da3_human
python "$SCRIPT_DIR/05_build_anchor_pointcloud.py"
if [ $? -ne 0 ]; then echo "❌ FAILED" >&2; exit 1; fi
echo "🎉 Done. anchor 产物在 $ANCHOR_DIR"
