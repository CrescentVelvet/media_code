#!/usr/bin/env bash
# AUDIT-15 一键：I-03 —— 只换 2DGS 光栅器（头不动、不重训）
#
# 单变量：同一个 i12_fullpts_sh PLY（官方等价 22.11 dB）+ 同一组相机
#         + 同 sh_degree=2 + 同 eps2d=0.3 + 同 slant mask + 同视角对
# 唯一变量：光栅器（3DGS EWA / 2DGS ray-splat），外加 2DGS 必需的轴重排（E7）
#
# 用法：
#   bash compare_4methods/audit_15_2dgs_raster.sh              # 全 32 视角
#   VIEWS=0,8,16 bash compare_4methods/audit_15_2dgs_raster.sh # 快验（3 视角）
#
# 判据见 IMPROVEMENTS.md §3.5.0（渲染前写死）：
#   P-D   掠射区 2DGS/3DGS ≤0.90    P-D2  2DGS 掠射绝对值 <2.40%
#   P-COV 深度覆盖比 ≥0.95          P-E7  naive/重排 α像素比 ≤0.30
set -o pipefail
B=/home/velvet/output/da3_gs_results
SC="${SC:-$B/佳佳抱枕_p1m_off}"
NPZ="${NPZ:-$B/佳佳抱枕_p1m/official_dump.npz}"
PLY="${PLY:-$SC/official_export/i12_fullpts_sh/0000.ply}"
OUT="${OUT:-$SC/i03_2dgs}"
CODE=/mnt/c/code/media_code
ENVS=/home/velvet/miniconda3/envs/querysplat

for f in "$NPZ" "$PLY"; do
  [ -f "$f" ] || { echo "❌ 缺 $f"; exit 1; }
done
mkdir -p "$OUT"

echo "PLY : $PLY ($(du -h "$PLY" | cut -f1))"
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
VIEWS="${VIEWS:-all}" NAIVE_VIEWS="${NAIVE_VIEWS:-0,8,16}" \
  exec python "$CODE/compare_4methods/audit_15_2dgs_raster.py"
