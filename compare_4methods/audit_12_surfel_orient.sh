#!/usr/bin/env bash
# AUDIT-12 运行器：DA3 的盘法向是「朝相机」还是「朝真实表面」
# 用法：bash compare_4methods/audit_12_surfel_orient.sh [SRCDIR] [VIEWS]
set -o pipefail
B=/home/velvet/output/da3_gs_results
SC="${1:-$B/佳佳抱枕_p1m}"
PLY="${PLY:-$SC/gs_ply/0000.ply}"
NPZ="${NPZ:-$SC/cameras.npz}"
OUT="${OUT:-$SC/audit_surfel_orient.json}"
VIEWS="${2:-0,8,16,24}"
PLY="$PLY" NPZ="$NPZ" OUT="$OUT" VIEWS="$VIEWS" \
  exec /home/velvet/miniconda3/envs/querysplat/bin/python \
  "$(dirname "$0")/audit_12_surfel_orient.py"
