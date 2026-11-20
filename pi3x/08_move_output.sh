#!/usr/bin/env bash
# 08_move_output.sh — WSL 专用：把结果从 Linux fs 搬到 Windows 盘。
#
# 用法:
#   bash pi3x/08_move_output.sh           # ~/output/pi3x_results → /mnt/d/output/results/pi3x_results
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

DST_ROOT="${D_RESULTS_DIR:-/mnt/d/output/results/pi3x_results}"

echo "🚀 [08] 搬运输出"
echo "  📂 源: $RESULTS_DIR"
echo "  💾 目标: $DST_ROOT"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ ERROR: 源目录不存在: $RESULTS_DIR" >&2
    exit 1
fi

mkdir -p "$DST_ROOT"
cp -r "$RESULTS_DIR/." "$DST_ROOT/"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: 复制中断" >&2
    exit 1
fi
echo "🎉 [08] Done. 源目录保留未删（确认无误后可手动 rm -rf $RESULTS_DIR）"
