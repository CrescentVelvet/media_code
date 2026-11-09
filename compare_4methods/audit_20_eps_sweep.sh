#!/usr/bin/env bash
# AUDIT-20 一键：2DGS 更糊是不是 eps2d（最小足迹下限）语义不同
set -o pipefail
B=/home/velvet/output/da3_gs_results
SC="${SC:-$B/佳佳抱枕_p1m_off}"
NPZ="${NPZ:-$B/佳佳抱枕_p1m/official_dump.npz}"
PLY="${PLY:-$SC/official_export/i12_fullpts_sh/0000.ply}"
OUT="${OUT:-$SC/i03b_oracle}"
CODE=/mnt/c/code/media_code
ENVS=/home/velvet/miniconda3/envs/querysplat
for f in "$NPZ" "$PLY"; do [ -f "$f" ] || { echo "❌ 缺 $f"; exit 1; }; done
if [ -f /home/velvet/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1090
  source /home/velvet/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENVS"
fi
export CUDA_HOME="$ENVS"
export CPATH="$ENVS/targets/x86_64-linux/include"
export LIBRARY_PATH="$ENVS/targets/x86_64-linux/lib"
SC="$SC" NPZ="$NPZ" PLY="$PLY" OUT="$OUT" VIEWS="${VIEWS:-0,8,16,24}" \
  EPS="${EPS:-0.3,0.1,0.01}" exec python "$CODE/compare_4methods/audit_20_eps_sweep.py"
