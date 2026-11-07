#!/usr/bin/env bash
# AUDIT-3 运行器：opacity / scale 全局增益扫掠（判"成倍口径错"）
# 用法：bash compare_4methods/audit_3_param_sweep.sh [direct|res1]
set -eo pipefail
ENVS="${QUERYSPLAT_ENV:-$HOME/miniconda3/envs/querysplat}"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$ENVS"
else
  export PATH="$ENVS/bin:$PATH"
fi
export CUDA_HOME="$ENVS"
export CPATH="$ENVS/targets/x86_64-linux/include"
export LIBRARY_PATH="$ENVS/targets/x86_64-linux/lib"

B="$HOME/output/da3_gs_results"
NPZ="$B/佳佳抱枕_p1m/cameras.npz"
case "${1:-direct}" in
  direct) PLY="$B/佳佳抱枕_p1m/gs_ply/0000.ply"; OUT="$B/佳佳抱枕_p1m/audit_direct" ;;
  res1)   PLY="$B/佳佳抱枕_res_ab/final_res1/point_cloud/iteration_2000/point_cloud.ply"
          OUT="$B/佳佳抱枕_res_ab/audit_res1" ;;
  *) echo "未知标签 $1（direct|res1）"; exit 2 ;;
esac
echo "PLY=$PLY"
PLY="$PLY" NPZ="$NPZ" OUT="$OUT" exec python "$(dirname "$0")/audit_3_param_sweep.py"
