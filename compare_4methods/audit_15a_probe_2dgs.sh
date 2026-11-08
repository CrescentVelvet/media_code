#!/usr/bin/env bash
# AUDIT-15a 一键：2DGS 光栅器 API 探针（秒级，无需 PLY/相机）
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

exec python "$CODE/compare_4methods/audit_15a_probe_2dgs.py" "$@"
