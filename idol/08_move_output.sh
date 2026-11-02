#!/usr/bin/env bash
# 08_move_output.sh —— 把 Linux fs 上的结果搬到 /mnt/d/output（WSL 专用）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

DEST_ROOT="${DEST_ROOT:-/mnt/d/output}"
DEST="${DEST:-$DEST_ROOT/idol_results}"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ ERROR: 结果目录不存在: $RESULTS_DIR" >&2
    exit 1
fi

echo "🚀 [08] 搬移结果"
echo "  📁 源:   $RESULTS_DIR"
echo "  💾 目标: $DEST"
mkdir -p "$DEST"
# 只搬产物，不带 __pycache__ 之类
rsync -a --info=progress2 "$RESULTS_DIR"/ "$DEST"/
if [ $? -ne 0 ]; then
    echo "❌ rsync 失败" >&2
    exit 1
fi
echo "📊 文件数: $(find "$DEST" -type f | wc -l)  | 体积: $(du -sh "$DEST" | cut -f1)"
echo "🎉 [08] Done."
