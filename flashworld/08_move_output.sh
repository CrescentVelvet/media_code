#!/usr/bin/env bash
# 08_move_output.sh — WSL 专用：把结果从 Linux fs 搬到 Windows 盘（vhdx 空间有限）。
#
# 默认把 $RESULTS_DIR（~/output/flashworld_results）整棵搬到 /mnt/d/output/flashworld_results。
# 大 PLY（几十 MB~几百 MB 一个）搬完 Linux 侧不留副本；加 KEEP_SRC=1 改成复制。
#
# 用法：
#   bash flashworld/08_move_output.sh
#   KEEP_SRC=1 bash flashworld/08_move_output.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SRC="${SRC:-$RESULTS_DIR}"
DST="${DST:-/mnt/d/output/flashworld_results}"
KEEP_SRC="${KEEP_SRC:-0}"

if [ ! -d "$SRC" ]; then
    echo "❌ ERROR: 源目录不存在: $SRC" >&2
    exit 1
fi
mkdir -p "$DST"

echo "🚀 [08] 搬运结果 → D 盘"
echo "  📁 源:   $SRC  ($(du -sh "$SRC" 2>/dev/null | cut -f1))"
echo "  💾 目标: $DST"
echo ""

if [ "$KEEP_SRC" = "1" ]; then
    cp -a "$SRC"/. "$DST"/
    rc=$?
    action="copied"
else
    # 逐顶层条目 mv：中途断掉也不会出现「半个目录没了」的状态
    for entry in "$SRC"/*; do
        [ -e "$entry" ] || continue
        base="$(basename "$entry")"
        if [ -e "$DST/$base" ]; then
            # 目标已存在 → 只补拷贝，**不删源**（避免 rm -rf 误伤；确认无误后手动清）
            echo "  ⏭️  目标已存在，拷贝合并（源保留）: $base"
            cp -a "$entry"/. "$DST/$base"/
        else
            mv "$entry" "$DST"/
        fi
    done
    rc=$?
    action="moved"
fi

if [ $rc -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "  ✅ $action: $(du -sh "$DST" 2>/dev/null | cut -f1) 在 $DST"
echo ""
echo "🎉 Done."
