#!/usr/bin/env bash
# 02_prepare_images.sh — Remy 采集包 → 去畸变 PNG + conditions.npz（GT 位姿/内参）。
#
# 用法:
#   SCENE=jiajia_baozhen \
#   RAW_DIR="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕" \
#     bash pi3x/02_prepare_images.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE="${SCENE:-jiajia_baozhen}"
RAW_DIR="${RAW_DIR:-}"
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/$SCENE}"

echo "🚀 [02] 采集包 → Pi3X 输入"
echo "  📂 输入: $RAW_DIR"
echo "  💾 输出: $OUT_DIR"

if [ -z "$RAW_DIR" ] || [ ! -f "$RAW_DIR/transforms.json" ]; then
    echo "❌ ERROR: 必须指定 RAW_DIR（含 transforms.json 的采集包路径）" >&2
    exit 1
fi

RAW_DIR="$RAW_DIR" OUT_DIR="$OUT_DIR" python "$SCRIPT_DIR/prepare_images.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 [02] Done."
