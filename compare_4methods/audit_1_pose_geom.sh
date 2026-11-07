#!/usr/bin/env bash
# AUDIT-1 运行器：位姿/内参/几何口径体检（只读）
# 用法：bash compare_4methods/audit_1_pose_geom.sh [SRCDIR]
set -o pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate querysplat 2>/dev/null || true
SC="${1:-$HOME/output/da3_gs_results/佳佳抱枕_p1m}"
SRCDIR="$SC" exec python "$(dirname "$0")/audit_1_pose_geom.py"
