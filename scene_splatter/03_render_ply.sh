#!/usr/bin/env bash
# 03_render_ply.sh — scene.ply → 轨道渲染视频（备用件，见 03_render_ply.py 头注）
#
# 用法：
#   PLY=<scene.ply> OUT=<orbit.mp4> bash scene_splatter/03_render_ply.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

PLY="${PLY:-}"
OUT="${OUT:-}"

if [ -z "$PLY" ] || [ ! -f "$PLY" ]; then
    echo "❌ ERROR: PLY 未设置或不存在: $PLY" >&2; exit 1
fi
if [ -z "$OUT" ]; then
    echo "❌ ERROR: OUT 未设置（输出 mp4 路径）" >&2; exit 1
fi

echo "🚀 [03] 渲染 PLY → $OUT"
python "$SCRIPT_DIR/03_render_ply.py" "$PLY" "$OUT"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 Done."
