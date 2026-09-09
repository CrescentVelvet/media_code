#!/usr/bin/env bash
# 09_move_output.sh — WSL 专用：把结果从 Linux fs 搬到 Windows 盘。
# 训练写 Linux fs（~/output/...）快；搬完存 /mnt/d/output/ 方便查看。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

DEST="${DEST:-/mnt/d/output/flame_human_results}"
SRC="${RESULTS_DIR:-$REPO_DIR/../flame_human_results}"

echo "🚚 [flame_human 09] 搬运结果"
echo "  📤 源: $SRC"
echo "  📥 目标: $DEST"

if [ ! -d "$SRC" ]; then
    echo "❌ 源目录不存在: $SRC"
    exit 1
fi

mkdir -p "$DEST"
# 大文件多，用 rsync 便于断点续传；没有就退回 cp
if command -v rsync >/dev/null 2>&1; then
    rsync -ah --info=progress2 "$SRC"/ "$DEST"/
else
    cp -a "$SRC"/. "$DEST"/
fi

if [ $? -ne 0 ]; then
    echo "❌ FAILED: 搬运失败"
    exit 1
fi

echo "✅ 搬运完成"
du -sh "$DEST" 2>/dev/null
ls -1 "$DEST"
