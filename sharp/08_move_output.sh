#!/usr/bin/env bash
# 08_move_output.sh — 把推理结果从 WSL Linux fs 剪切到 Windows D: 盘（释放 vhdx 空间）。
#
# WSL 专用：结果先写在 ~/output/（Linux fs，I/O 快），跑完搬到 /mnt/d/output/ 归档。
# 跨文件系统 mv = copy + delete，用 rsync 保证安全：
#   1. rsync 复制（逐文件校验）
#   2. 全部成功才删源
#   3. 任一文件失败则保留源，不删
#
# 用法：
#   bash sharp/08_move_output.sh
#   SRC=~/output/sharp_results DST=/mnt/d/output/sharp_results bash sharp/08_move_output.sh
#   DRY_RUN=1 bash sharp/08_move_output.sh      # 预览，不实际移动
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SRC="${SRC:-$RESULTS_DIR}"
DST="${DST:-/mnt/d/output/sharp_results}"

echo "📦 [08] 搬运结果到 Windows D: 盘"
echo "  📁 源:   $SRC"
echo "  💾 目标: $DST"
[ "${DRY_RUN:-0}" = "1" ] && echo "  ⏭️  DRY_RUN=1（预览，不实际移动）"
echo ""

# ── 前置检查 ───────────────────────────────────────────────────────────────
if [ ! -d "$SRC" ]; then
    echo "❌ ERROR: 源目录不存在: $SRC" >&2
    exit 1
fi

_dst_parent="$(dirname "$DST")"
if [ ! -d "$_dst_parent" ]; then
    echo "❌ ERROR: 目标父目录不存在: $_dst_parent" >&2
    echo "       确认 D: 盘已挂载: ls /mnt/d/" >&2
    exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "❌ ERROR: rsync 未安装。安装: sudo apt install rsync" >&2
    exit 1
fi

# ── 空间检查 ───────────────────────────────────────────────────────────────
_src_size_kb=$(du -sk "$SRC" 2>/dev/null | cut -f1)
_src_size_gb=$(awk "BEGIN{printf \"%.1f\", $_src_size_kb/1024/1024}")
_dst_avail_kb=$(df -k "$_dst_parent" 2>/dev/null | tail -1 | awk '{print $4}')
_dst_avail_gb=$(awk "BEGIN{printf \"%.1f\", $_dst_avail_kb/1024/1024}")

echo "  📐 源大小:   $_src_size_gb GB"
echo "  📐 目标可用: $_dst_avail_gb GB"
if [ "$_src_size_kb" -ge "$_dst_avail_kb" ] 2>/dev/null; then
    echo "❌ ERROR: 目标空间不足" >&2
    exit 1
fi

# ── 目标非空时确认 ─────────────────────────────────────────────────────────
if [ -d "$DST" ] && [ "$(ls -A "$DST" 2>/dev/null)" ]; then
    echo "⚠️  目标目录非空: $DST（rsync 会合并/覆盖同名文件）"
    echo "  继续请按 Enter，取消请 Ctrl+C"
    [ "${DRY_RUN:-0}" != "1" ] && read -r _ < /dev/tty
fi

# ── 执行 ───────────────────────────────────────────────────────────────────
mkdir -p "$DST"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo ""
    echo "🔍 DRY RUN — 实际执行会:"
    echo "  rsync -av --remove-source-files \"$SRC/\" \"$DST/\""
    echo "  rm -rf \"$SRC\""
    echo ""
    echo "✅ DRY RUN 完成（未移动任何文件）"
    exit 0
fi

echo ""
echo "📦 rsync 复制中..."
if rsync -av --remove-source-files "$SRC/" "$DST/"; then
    find "$SRC" -type d -empty -delete 2>/dev/null || true
    if [ -d "$SRC" ] && [ -z "$(ls -A "$SRC" 2>/dev/null)" ]; then
        rmdir "$SRC" 2>/dev/null || true
    fi
    echo "  ✅ rsync 完成"
else
    echo "❌ rsync 失败！源文件保留在 $SRC（未删除）" >&2
    exit 1
fi

echo ""
echo "🎉 [08] Done."
echo "  📁 结果已搬到: $DST"
echo "  💾 WSL Linux fs 已释放空间"
