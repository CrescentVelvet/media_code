#!/usr/bin/env bash
# 03_run_3dgs_baseline.sh — 原版 3DGS 基线（COLMAP sparse 点云初始化）
# 目的：复现同事观察——跨视角位姿不一致时原版 3DGS 出 floater/重影（对照组，无 DA3 几何）
# 用法: bash da3_human/03_run_3dgs_baseline.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

export SOURCE_DIR="${SOURCE_DIR:-$HOME/output/vggt_human_results/source}"
export GAUSSIAN_DIR="${GAUSSIAN_DIR:-$RESULTS_DIR/baseline_3dgs_colmap}"

echo "🚀 [03] 原版 3DGS 基线 (COLMAP init)"
echo "  📌 说明: 官方 3DGS 无「随机初始化」CLI 开关，本基线 = COLMAP sparse 点云初始化"
echo "           （即 vggt_human 现有 scene_gs 初始化路径，正是要替换的环节）"

source "$SCRIPT_DIR/_train_3dgs.sh"
if [ $? -ne 0 ]; then echo "❌ FAILED" >&2; exit 1; fi
echo "🎉 Done. 产物: $GAUSSIAN_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
