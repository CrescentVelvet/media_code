#!/usr/bin/env bash
# AUDIT-15b 一键：I-03 主判据的可判性诊断（只读，不渲染，几十秒）
set -o pipefail
B=/home/velvet/output/da3_gs_results
SC_REF="$B/佳佳抱枕_p1m"
CODE=/mnt/c/code/media_code
ENVS=/home/velvet/miniconda3/envs/querysplat

NPZ="${NPZ:-$SC_REF/official_dump.npz}"
OUT="${OUT:-/tmp/audit_15b.json}"

if [ -f /home/velvet/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1090
  source /home/velvet/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENVS"
fi
export CUDA_HOME="$ENVS"
export CPATH="$ENVS/targets/x86_64-linux/include"
export LIBRARY_PATH="$ENVS/targets/x86_64-linux/lib"

NPZ="$NPZ" OUT="$OUT" exec python "$CODE/compare_4methods/audit_15b_depth_baseline.py"
