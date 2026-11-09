#!/usr/bin/env bash
# AUDIT-22 一键：把 DA3-GS 的 PLY 绕一圈渲染成 MP4（与 supersplat 手动查看对照）
# 默认读「完整输出」档 PLY（610 万点 / 全 SH / 953 MB），即 supersplat 要打开的那个文件。
set -o pipefail
B=/home/velvet/output/da3_gs_results
SC="${SC:-$B/佳佳抱枕_p1m_off}"
NPZ="${NPZ:-$B/佳佳抱枕_p1m/official_dump.npz}"
PLY="${PLY:-$SC/official_export/i12_fullpts_sh/0000.ply}"
OUT="${OUT:-$SC/orbit_video}"
CODE=/mnt/c/code/media_code
ENVS=/home/velvet/miniconda3/envs/querysplat

for f in "$NPZ" "$PLY"; do
  [ -f "$f" ] || { echo "❌ 缺 $f"; exit 1; }
done

if [ -f /home/velvet/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1090
  source /home/velvet/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENVS"
fi
export CUDA_HOME="$ENVS"
export CPATH="$ENVS/targets/x86_64-linux/include"
export LIBRARY_PATH="$ENVS/targets/x86_64-linux/lib"

SC="$SC" NPZ="$NPZ" PLY="$PLY" OUT="$OUT" \
NFRAMES="${NFRAMES:-300}" FPS="${FPS:-30}" OUTW="${OUTW:-540}" \
RADIUS_K="${RADIUS_K:-1.06}" ELEV="${ELEV:-}" BG="${BG:-white}" ARC="${ARC:-0}" OUT_NAME="${OUT_NAME:-orbit}" \
  exec python "$CODE/compare_4methods/audit_22_orbit_video.py"
