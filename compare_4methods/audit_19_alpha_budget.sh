#!/usr/bin/env bash
# AUDIT-19 一键：为什么 2DGS 更糊 —— 每像素的 α 覆盖预算
#
# 判据（写死）：2DGS 的 α 均值 / 3DGS < 0.8 → 覆盖变薄成立。
set -o pipefail
B=/home/velvet/output/da3_gs_results
SC="${SC:-$B/佳佳抱枕_p1m_off}"
NPZ="${NPZ:-$B/佳佳抱枕_p1m/official_dump.npz}"
PLY="${PLY:-$SC/official_export/i12_fullpts_sh/0000.ply}"
OUT="${OUT:-$SC/i03b_oracle}"
CODE=/mnt/c/code/media_code
ENVS=/home/velvet/miniconda3/envs/querysplat

for f in "$NPZ" "$PLY" "$OUT/o2_normals.npy"; do
  [ -f "$f" ] || { echo "❌ 缺 $f（先跑 audit_17_i03b_oracle.sh）"; exit 1; }
done

if [ -f /home/velvet/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1090
  source /home/velvet/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENVS"
fi
export CUDA_HOME="$ENVS"
export CPATH="$ENVS/targets/x86_64-linux/include"
export LIBRARY_PATH="$ENVS/targets/x86_64-linux/lib"

SC="$SC" NPZ="$NPZ" PLY="$PLY" OUT="$OUT" VIEWS="${VIEWS:-0,8,16,24}" \
  exec python "$CODE/compare_4methods/audit_19_alpha_budget.py"
