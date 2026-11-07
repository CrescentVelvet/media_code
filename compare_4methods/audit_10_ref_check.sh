#!/usr/bin/env bash
# AUDIT-10 运行器：评测参考图的 resize 口径核对
set -o pipefail
B=/home/velvet/output/da3_gs_results
NPZ="${1:-$B/佳佳抱枕_p1m/official_dump.npz}"
IMAGES_DIR="${IMAGES_DIR:-/home/velvet/output/da3_results/_captures/佳佳抱枕/images}"
NPZ="$NPZ" IMAGES_DIR="$IMAGES_DIR" \
  exec /home/velvet/miniconda3/envs/querysplat/bin/python \
  "$(dirname "$0")/audit_10_ref_check.py"
