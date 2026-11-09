#!/usr/bin/env bash
# AUDIT-23 一键：采集视角的方位角覆盖体检（这批相机是不是完整环绕？）
set -o pipefail
B=/home/velvet/output/da3_gs_results
SC="${SC:-$B/佳佳抱枕_p1m_off}"
NPZ="${NPZ:-$B/佳佳抱枕_p1m/official_dump.npz}"
OUT="${OUT:-$SC/coverage.json}"
ENVS=/home/velvet/miniconda3/envs/querysplat

[ -f "$NPZ" ] || { echo "❌ 缺 $NPZ"; exit 1; }

if [ -f /home/velvet/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1090
  source /home/velvet/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENVS"
fi

NPZ="$NPZ" OUT="$OUT" exec python /mnt/c/code/media_code/compare_4methods/audit_23_coverage.py
