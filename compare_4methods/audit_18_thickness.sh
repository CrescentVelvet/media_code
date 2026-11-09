#!/usr/bin/env bash
# AUDIT-18 一键：盘厚度体检 —— DA3 的高斯到底是不是"薄盘"
#
# 动机：I-03b 给了 oracle 法向，2DGS 反而更差。若法向不是瓶颈，就回到更基础的假设：
#       2DGS 把每个高斯当成零厚度 surfel（第三轴完全不参与投影）。
#       该假设只在 s_min/s_mid ≪ 1 时成立。
#
# 判据（写死）：s_min/s_mid 中位 > 0.15 → 「不是薄盘」成立。
set -o pipefail
B=/home/velvet/output/da3_gs_results
PLY="${PLY:-$B/佳佳抱枕_p1m_off/official_export/i12_fullpts_sh/0000.ply}"
CODE=/mnt/c/code/media_code
ENVS=/home/velvet/miniconda3/envs/querysplat

[ -f "$PLY" ] || { echo "❌ 缺 $PLY"; exit 1; }
echo "PLY : $PLY"
echo

if [ -f /home/velvet/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1090
  source /home/velvet/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENVS"
fi

PLY="$PLY" RES_SCALE="${RES_SCALE:-3.806}" FOCAL="${FOCAL:-375.62}" Z_MED="${Z_MED:-1.5}" \
  exec python "$CODE/compare_4methods/audit_18_thickness.py"
