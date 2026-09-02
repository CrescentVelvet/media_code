#!/usr/bin/env bash
# 08_move_output.sh — 将训练/推理结果从 WSL Linux fs 剪切到 Windows D: 盘（释放 vhdx 空间）。
#
# WSL 专用：Linux fs (~/output/) 空间有限（100GB vhdx），跑完 pipeline 后把结果搬到
# /mnt/d/output/ 腾出空间给下一轮。跨文件系统 mv = copy + delete，用 rsync 保证安全：
#   1. rsync 复制（逐文件校验）
#   2. 全部成功才 rm -rf 源
#   3. 任一文件失败则保留源，不删
#
# 用法：
#   bash vggt_human/08_move_output.sh
#   SRC=~/output/vggt_human_results DST=/mnt/d/output/vggt_human_results bash vggt_human/08_move_output.sh
#   DRY_RUN=1 bash vggt_human/08_move_output.sh   # 预览，不实际移动
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# 源目录（默认 RESULTS_DIR，与 01-07 一致）
SRC="${SRC:-$RESULTS_DIR}"
# 目标目录（默认 D:\output\vggt_human_results）
DST="${DST:-/mnt/d/output/vggt_human_results}"
# 下载临时目录、pipeline 中间产物也一起搬（如果存在）
MOVE_EXTRA_DIRS="${MOVE_EXTRA_DIRS:-1}"

echo "📦 [08] 搬运结果到 Windows D: 盘"
echo "  📁 源:   $SRC"
echo "  💾 目标: $DST"
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "  ⏭️  DRY_RUN=1（预览，不实际移动）"
fi
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

# 检查 rsync
if ! command -v rsync >/dev/null 2>&1; then
    echo "❌ ERROR: rsync 未安装。安装: sudo apt install rsync" >&2
    exit 1
fi

# ── 源大小 vs 目标可用空间 ─────────────────────────────────────────────────
_src_size_kb=$(du -sk "$SRC" 2>/dev/null | cut -f1)
_src_size_gb=$(awk "BEGIN{printf \"%.1f\", $_src_size_kb/1024/1024}")
_dst_avail_kb=$(df -k "$_dst_parent" 2>/dev/null | tail -1 | awk "{print \$4}")
_dst_avail_gb=$(awk "BEGIN{printf \"%.1f\", $_dst_avail_kb/1024/1024}")

echo "  📐 源大小:       $_src_size_gb GB"
echo "  📐 目标可用:     $_dst_avail_gb GB"

if [ "$_src_size_kb" -ge "$_dst_avail_kb" ] 2>/dev/null; then
    echo "❌ ERROR: 目标空间不足（源 $_src_size_gb GB > 可用 $_dst_avail_gb GB）" >&2
    exit 1
fi

# ── 目标已存在时确认 ─────────────────────────────────────────────────────────
if [ -d "$DST" ] && [ "$(ls -A "$DST" 2>/dev/null)" ]; then
    echo "⚠️  目标目录非空: $DST"
    echo "  rsync 会合并/覆盖同名文件。如需完全覆盖，先手动删除目标。"
    echo "  继续请按 Enter，取消请 Ctrl+C"
    [ "${DRY_RUN:-0}" != "1" ] && read -r _ < /dev/tty
fi

# ── 执行搬运 ─────────────────────────────────────────────────────────────────
mkdir -p "$DST"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo ""
    echo "🔍 DRY RUN — 实际执行会:"
    echo "  rsync -av --remove-source-files \"$SRC/\" \"$DST/\""
    echo "  rm -rf \"$SRC\""
    echo "  (额外目录同理)"
    echo ""
    echo "✅ DRY RUN 完成（未移动任何文件）"
    exit 0
fi

echo ""
echo "📦 rsync 复制中..."
if rsync -av --remove-source-files "$SRC/" "$DST/"; then
    echo "  ✅ rsync 完成（文件已复制并从源删除）"
    # rsync --remove-source-files 不删空目录，手动清理
    find "$SRC" -type d -empty -delete 2>/dev/null || true
    # 如果源目录还在但已空，删掉
    if [ -d "$SRC" ] && [ -z "$(ls -A "$SRC" 2>/dev/null)" ]; then
        rmdir "$SRC" 2>/dev/null || true
    fi
    if [ -d "$SRC" ]; then
        echo "  ⚠️ 源目录仍有残留（可能有非空子目录）: $SRC"
        echo "     手动检查: ls -la $SRC"
    else
        echo "  🎉 源目录已清理: $SRC"
    fi
else
    echo "❌ rsync 失败！源文件保留在 $SRC（未删除）" >&2
    echo "  已复制的文件在 $DST（不完整）" >&2
    exit 1
fi

# ── 额外目录（pipeline 中间产物等）───────────────────────────────────────────
if [ "$MOVE_EXTRA_DIRS" = "1" ]; then
    for _extra in \
        "$REPO_DIR/../vggt_human_results" \
        "$REPO_DIR/../vggt_human_experiments"; do
        if [ -d "$_extra" ] && [ "$(ls -A "$_extra" 2>/dev/null)" ]; then
            # 额外产物与主结果放到同一目标父目录下（与 DST 保持一致）
            _extra_dst="$(dirname "$DST")/$(basename "$_extra")"
            echo ""
            echo "📦 搬运额外目录: $_extra -> $_extra_dst"
            mkdir -p "$_extra_dst"
            if rsync -av --remove-source-files "$_extra/" "$_extra_dst/"; then
                find "$_extra" -type d -empty -delete 2>/dev/null || true
                echo "  ✅ 完成"
            else
                echo "  ⚠️ rsync 失败，源保留" >&2
            fi
        fi
    done
fi

echo ""
echo "🎉 [08] Done."
echo "  📁 结果已搬到: $DST"
echo "  💾 WSL Linux fs 已释放空间"
