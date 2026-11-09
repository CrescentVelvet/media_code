#!/usr/bin/env bash
# AUDIT-21 一键：人脸窗订正 + 全部现有渲染重测（只读 PNG）
set -o pipefail
B=/home/velvet/output/da3_gs_results
SC="${SC:-$B/佳佳抱枕_p1m_off}"
REF_NPZ="${REF_NPZ:-$B/佳佳抱枕_p1m/official_dump.npz}"
ENVS=/home/velvet/miniconda3/envs/querysplat
[ -f "$REF_NPZ" ] || { echo "❌ 缺 $REF_NPZ"; exit 1; }
if [ -f /home/velvet/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1090
  source /home/velvet/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENVS"
fi
SC="$SC" REF_NPZ="$REF_NPZ" exec python /mnt/c/code/media_code/compare_4methods/audit_21_face_recheck.py
