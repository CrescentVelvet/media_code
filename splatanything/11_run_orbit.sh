#!/usr/bin/env bash
# 11_run_orbit.sh — 对 02 的输出目录渲 360° 轨道视频（mp4）。
#
# 用法：
#   RESULT=~/output/splatanything_results/rgb/portrait_1059 bash 11_run_orbit.sh
#
# Env:
#   RESULT=           02 的输出目录（含 gaussians.ply / cameras.json / rgb/）——必填
#   OUT=              输出 mp4，默认 $RESULT/orbit.mp4
#   FRAMES=120        帧数
#   FPS=30            帧率
#   RADIUS_SCALE=1.0  轨道半径 = 源相机到中心的中位距离 × 此系数
#   ELEVATION_DEG=    固定俯仰角（度）；缺省 = 源相机中位仰角
#   EPS2D=0.01        屏幕空间膨胀
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"        # conda env 激活（_gsplat_env.sh 依赖 $CONDA_PREFIX 里的 python/nvcc）
source "$SCRIPT_DIR/_gsplat_env.sh"

if [ -z "${RESULT:-}" ] || [ ! -f "$RESULT/gaussians.ply" ]; then
    echo "❌ ERROR: 需要 RESULT 指向 02 的输出目录（当前: '${RESULT:-}'）" >&2
    exit 1
fi
: "${FRAMES:=120}"
: "${FPS:=30}"
: "${RADIUS_SCALE:=1.0}"
: "${EPS2D:=0.01}"

ARGS=(--result "$RESULT" --frames "$FRAMES" --fps "$FPS"
      --radius-scale "$RADIUS_SCALE" --eps2d "$EPS2D")
[ -n "${OUT:-}" ] && ARGS+=(--out "$OUT")
[ -n "${ELEVATION_DEG:-}" ] && ARGS+=(--elevation-deg "$ELEVATION_DEG")

python "$SCRIPT_DIR/11_render_orbit.py" "${ARGS[@]}"
