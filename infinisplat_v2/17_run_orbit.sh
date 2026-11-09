#!/usr/bin/env bash
# 17) 把 scene.ply 沿水平轨道环绕一圈渲染成 MP4（机位/轨道拟合复用 16_extrap_eval）
#
# 用法:
#   RESULT=~/output/infinisplat_v2_results/jiajia TAG=orig bash 17_run_orbit.sh
#   RESULT=... TAG=ft ARC=1 bash 17_run_orbit.sh      # 只走有观测的方位跨度
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${RESULT:?ERROR: 需要 RESULT（02 的输出目录，含 scene.ply/cameras.npz/result.json）}"
: "${TAG:=orbit}"
: "${OUT:=$RESULT/orbit_video}"
: "${NFRAMES:=240}"
: "${FPS:=30}"
: "${OUTW:=540}"
: "${ELEV:=}"            # 空 = 观测俯仰中位（随原环）；\"水平\"字面意义就 ELEV=0
: "${ARC:=0}"
: "${RADIUS_K:=1.0}"

echo "🚀 [17] 轨道环绕视频 | $TAG"
echo "  📁 PLY: $RESULT/scene.ply"
echo "  💾 输出: $OUT/$TAG.mp4（${NFRAMES}帧/${FPS}fps）"

ARGS=(--result "$RESULT" --out "$OUT" --tag "$TAG"
      --nframes "$NFRAMES" --fps "$FPS" --outw "$OUTW" --radius-k "$RADIUS_K")
[ -n "$ELEV" ] && ARGS+=(--elev "$ELEV")
[ "$ARC" = "1" ] && ARGS+=(--arc)

python "$SCRIPT_DIR/17_orbit_video.py" "${ARGS[@]}"
if [ $? -ne 0 ]; then
  echo "❌ FAILED" >&2
  exit 1
fi
echo "🎉 Done: $OUT/$TAG.mp4"
