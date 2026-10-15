#!/usr/bin/env bash
# 09b_collect_preview.sh — 把 09 批量任务的渲染预览图收集到 Windows 侧
# 用法: bash 09b_collect_preview.sh [BATCH_ROOT] [PREVIEW_DIR]
set -u

BATCH_ROOT="${1:-$HOME/output/hand_micro}"
PREVIEW_DIR="${2:-/mnt/d/output/hand_micro_preview}"

mkdir -p "$PREVIEW_DIR"

echo "BATCH_ROOT=$BATCH_ROOT"
echo "PREVIEW_DIR=$PREVIEW_DIR"

for WORK in "$BATCH_ROOT"/*/; do
    NAME="$(basename "$WORK")"
    if [ ! -d "$WORK/05_novel_renders" ]; then
        echo "SKIP $NAME (no 05_novel_renders)"
        continue
    fi
    mkdir -p "$PREVIEW_DIR/$NAME"
    # 彩色渲染图
    cp "$WORK"/05_novel_renders/*.png "$PREVIEW_DIR/$NAME/" 2>&1
    # alpha 覆盖图，加 alpha_ 前缀避免覆盖同名渲染图
    for f in "$WORK"/05_novel_alpha/*.png; do
        [ -e "$f" ] || continue
        cp "$f" "$PREVIEW_DIR/$NAME/alpha_$(basename "$f")" 2>&1
    done
    N="$(ls -1 "$PREVIEW_DIR/$NAME" | wc -l)"
    echo "$NAME -> $N files"
done

echo "=== TOTAL FILES ==="
find "$PREVIEW_DIR" -type f | wc -l
