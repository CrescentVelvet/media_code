#!/usr/bin/env bash
# AUDIT-4 运行器：内参先验扫掠（fx/fy 各向异性 + 焦距整体）
# 用法：bash compare_4methods/audit_4_k_prior.sh [SRCDIR]
set -o pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate querysplat 2>/dev/null || true
SC="${1:-$HOME/output/da3_gs_results/佳佳抱枕_p1m}"
SRCDIR="$SC" exec python "$(dirname "$0")/audit_4_k_prior.py"
