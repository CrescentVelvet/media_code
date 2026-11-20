#!/usr/bin/env bash
# 08_move_output.sh — 将 ZipSplat 结果复制到 D:\output\results（不删除 Linux 热数据）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
SOURCE_DIR="${1:-${SOURCE_DIR:-$RESULTS_DIR}}"
DEST_DIR="${DEST_DIR:-/mnt/d/output/results/zipsplat_results}"
[ -d "$SOURCE_DIR" ] || { echo "❌ 源结果不存在: $SOURCE_DIR" >&2; exit 1; }
mkdir -p "$DEST_DIR"
echo "🚀 复制实验结果（保留源文件）"
echo "  📁 $SOURCE_DIR"
echo "  💾 $DEST_DIR"
# DrvFs 可能拒绝 cp -a 的属主/时间戳写入；文件内容仍须逐文件校验。
cp -r "$SOURCE_DIR/." "$DEST_DIR/" || exit 1
python3 - "$SOURCE_DIR" "$DEST_DIR" <<'PY'
from pathlib import Path
import sys
src, dst = map(Path, sys.argv[1:])
files = [p for p in src.rglob("*") if p.is_file()]
missing = [str(p.relative_to(src)) for p in files
           if not (dst / p.relative_to(src)).is_file()
           or (dst / p.relative_to(src)).stat().st_size != p.stat().st_size]
if missing:
    raise SystemExit("❌ 复制校验失败: " + repr(missing[:10]))
print(f"✅ 逐文件长度校验通过: {len(files)} files")
PY
[ $? -ne 0 ] && exit 1
echo "🎉 Done. 源结果仍保留在 $SOURCE_DIR"
