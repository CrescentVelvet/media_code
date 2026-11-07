#!/usr/bin/env bash
# AUDIT-8 运行器：仅本视角的几何残差 + 多壳深度离散
# 用法：bash compare_4methods/audit_8_selfview_geom.sh [SRCDIR] [VIEWS]
set -o pipefail
B=/home/velvet/output/da3_gs_results
SC="${1:-$B/佳佳抱枕_p1m}"
PLY="${PLY:-$SC/gs_ply/0000.ply}"
NPZ="$SC/cameras.npz"
OUT="${OUT:-$SC}"
VIEWS="${2:-0,8,16}"
PLY="$PLY" NPZ="$NPZ" OUT="$OUT" VIEWS="$VIEWS" \
  exec /home/velvet/miniconda3/envs/querysplat/bin/python \
  "$(dirname "$0")/audit_8_selfview_geom.py"
