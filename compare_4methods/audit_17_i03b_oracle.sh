#!/usr/bin/env bash
# AUDIT-17 一键：I-03b —— 把盘法向换成 oracle（GT 深度图表面法向），验 H2
#
# 单变量：同一个 i12_fullpts_sh PLY + 同一组相机 + 同 sh_degree=2 + 同轴序约定
#         唯一变量：旋转矩阵第三列（盘法向）从 DA3 学出来的 → 换成 GT 真相
#
# 三档：
#   3dgs       EWA 基线（参照，22.11 dB）
#   o2_oracle  每点唯一法向（各视角命中法向的平均）——可被监督项逼出来的现实上限
#   o1_perview 逐视角 oracle —— 上界
#
# 用法：
#   bash compare_4methods/audit_17_i03b_oracle.sh                 # 全 32 视角
#   VIEWS=0,8,16 bash compare_4methods/audit_17_i03b_oracle.sh    # 快验
#
# 判据见 IMPROVEMENTS.md §3.5.2：
#   J-PSNR  全图 oracle ≥ 3DGS − 0.15 dB
#   J-FACE  人脸 oracle ≥ 3DGS − 0.15 dB
set -o pipefail
B=/home/velvet/output/da3_gs_results
SC="${SC:-$B/佳佳抱枕_p1m_off}"
NPZ="${NPZ:-$B/佳佳抱枕_p1m/official_dump.npz}"
PLY="${PLY:-$SC/official_export/i12_fullpts_sh/0000.ply}"
OUT="${OUT:-$SC/i03b_oracle}"
CODE=/mnt/c/code/media_code
ENVS=/home/velvet/miniconda3/envs/querysplat

for f in "$NPZ" "$PLY"; do
  [ -f "$f" ] || { echo "❌ 缺 $f"; exit 1; }
done
mkdir -p "$OUT"

echo "PLY : $PLY"
echo "NPZ : $NPZ"
echo "OUT : $OUT"
echo

if [ -f /home/velvet/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1090
  source /home/velvet/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENVS"
fi
export CUDA_HOME="$ENVS"
export CPATH="$ENVS/targets/x86_64-linux/include"
export LIBRARY_PATH="$ENVS/targets/x86_64-linux/lib"

SC="$SC" NPZ="$NPZ" PLY="$PLY" OUT="$OUT" \
VIEWS="${VIEWS:-all}" GATE="${GATE:-0.03}" \
  exec python "$CODE/compare_4methods/audit_17_i03b_oracle.py"
