#!/usr/bin/env bash
# AUDIT-16 一键：I-03 损失归属诊断（只读 PNG/npy，秒级）
set -o pipefail
B=/home/velvet/output/da3_gs_results
OUT="${OUT:-$B/佳佳抱枕_p1m_off/i03_2dgs}"
NPZ="${NPZ:-$B/佳佳抱枕_p1m/official_dump.npz}"
CODE=/mnt/c/code/media_code
ENVS=/home/velvet/miniconda3/envs/querysplat

if [ -f /home/velvet/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1090
  source /home/velvet/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENVS"
fi
export CUDA_HOME="$ENVS"
export CPATH="$ENVS/targets/x86_64-linux/include"
export LIBRARY_PATH="$ENVS/targets/x86_64-linux/lib"

NPZ="$NPZ" OUT="$OUT" VIEWS="${VIEWS:-0,8,16}" \
  exec python "$CODE/compare_4methods/audit_16_i03_diag.py"
