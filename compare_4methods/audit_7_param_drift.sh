#!/usr/bin/env bash
# AUDIT-7 运行器：训练前后参数漂移（σ / opacity / xyz / rotation）
# 用法：bash compare_4methods/audit_7_param_drift.sh [RES ...]   默认 1 2
set -o pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate querysplat 2>/dev/null || true
B="$HOME/output/da3_gs_results"
LIST="${*:-1 2}"
for R in $LIST; do
  echo "================ RES=$R ================"
  INIT="$B/佳佳抱枕_p1m/ffinit_ckpt/point_cloud/iteration_0/point_cloud.ply" \
  INIT_SRC_PLY="$B/佳佳抱枕_p1m/gs_ply/0000.ply" \
  TRAINED="$B/佳佳抱枕_res_ab/final_res$R/point_cloud/iteration_2000/point_cloud.ply" \
  NPZ="$B/佳佳抱枕_p1m/cameras.npz" OUT="$B/佳佳抱枕_res_ab/final_res$R" RES="$R" \
  python "$(dirname "$0")/audit_7_param_drift.py" || exit 1
done
