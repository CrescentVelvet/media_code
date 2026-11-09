#!/usr/bin/env bash
# AUDIT-25 一键：QuerySplat 的 3:4 口径判据（A/B/C/D 四方案 PSNR 三视角投票）
set -o pipefail
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

exec python "$CODE/compare_4methods/audit_25_qs_aspect.py"
