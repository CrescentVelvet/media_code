#!/usr/bin/env bash
# 08_move_output.sh —【WSL 专用】把**单个场景**的全部产物从 Linux fs 剪切到 D: 盘。
#
# 目录约定（2026-10-08 起）：一个输入数据一个文件夹，训练/渲染/数据集全部收在
#   $RESULTS_DIR/<scene>/            （Linux fs，训练期间）
#   /mnt/d/output/deformable_human_results/<scene>/   （归档后）
#
# ⚠️ 有任务正在读写该场景目录时禁止执行（08 会移走输入数据集 colmap_scene）。
#
# 用法：
#   SCENE_NAME=hand_motion bash deformable_human/08_move_output.sh
#   SCENE_NAME=hand_motion FORCE=1 bash deformable_human/08_move_output.sh  # 目标非空时合并
#   SCENE_NAME=hand_motion DRY_RUN=1 bash deformable_human/08_move_output.sh  # 预览
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE_NAME="${SCENE_NAME:?❌ ERROR: 必须指定 SCENE_NAME（按场景搬运）}"
SRC="${SRC:-$RESULTS_DIR/$SCENE_NAME}"
DST="${DST:-/mnt/d/output/deformable_human_results/$SCENE_NAME}"

echo "📦 [08] 搬运结果到 Windows D: 盘"
echo "  📁 源:   $SRC"
echo "  💾 目标: $DST"
[ "${DRY_RUN:-0}" = "1" ] && echo "  ⏭️  DRY_RUN=1（预览，不实际移动）"
echo ""

if [ ! -d "$SRC" ]; then
    echo "❌ ERROR: 源目录不存在: $SRC" >&2
    exit 1
fi
_dst_parent="$(dirname "$DST")"
if [ ! -d "$_dst_parent" ]; then
    echo "❌ ERROR: 目标父目录不存在: $_dst_parent（确认 D: 盘已挂载: ls /mnt/d/）" >&2
    exit 1
fi
if ! command -v rsync >/dev/null 2>&1; then
    echo "❌ ERROR: rsync 未安装（conda install -c conda-forge rsync 或 apt）" >&2
    exit 1
fi

_src_size_kb=$(du -sk "$SRC" 2>/dev/null | cut -f1)
_dst_avail_kb=$(df -k "$_dst_parent" 2>/dev/null | tail -1 | awk "{print \$4}")
echo "  📐 源大小:   $(awk "BEGIN{printf \"%.1f\", $_src_size_kb/1024/1024}") GB"
echo "  📐 目标可用: $(awk "BEGIN{printf \"%.1f\", $_dst_avail_kb/1024/1024}") GB"
if [ "$_src_size_kb" -ge "$_dst_avail_kb" ] 2>/dev/null; then
    echo "❌ ERROR: 目标空间不足" >&2
    exit 1
fi

if [ -d "$DST" ] && [ "$(ls -A "$DST" 2>/dev/null)" ]; then
    echo "⚠️  目标目录非空: $DST（rsync 会合并/覆盖同名文件）"
    if [ "${FORCE:-0}" = "1" ]; then
        echo "  FORCE=1，直接合并"
    elif [ -t 0 ]; then
        echo "  继续请按 Enter，取消请 Ctrl+C"
        read -r _ < /dev/tty || true
    else
        echo "❌ ERROR: 非交互模式且目标非空，需显式 FORCE=1 确认合并" >&2
        exit 1
    fi
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "🔍 DRY RUN — 实际执行会: rsync -av --remove-source-files \"$SRC/\" \"$DST/\" 然后清理空目录"
    exit 0
fi

mkdir -p "$DST"
echo "📦 rsync 复制中..."
if rsync -av --remove-source-files "$SRC/" "$DST/"; then
    find "$SRC" -type d -empty -delete 2>/dev/null || true
    if [ -d "$SRC" ] && [ -z "$(ls -A "$SRC" 2>/dev/null)" ]; then
        rmdir "$SRC" 2>/dev/null || true
    fi
    [ -d "$SRC" ] && echo "  ⚠️ 源目录仍有残留: $SRC（手动检查）" || echo "  🎉 源目录已清理"
else
    echo "❌ rsync 失败！源文件保留在 $SRC（未删除）" >&2
    exit 1
fi

echo ""
echo "🎉 [08] Done. 结果已搬到: $DST"
