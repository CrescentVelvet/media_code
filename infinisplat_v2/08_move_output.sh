#!/usr/bin/env bash
# 08_move_output.sh — 把 WSL Linux fs 上的结果搬到 Windows 盘（/mnt/d/output）。
#
# 训练/推理输出写在 ~/output 是 Linux fs（快），跨盘拷贝到 /mnt/d 后可在 Windows 侧直接看。
# 默认不删源目录，确认无误后再手动清理。
#
# Env (all optional):
#   DEST_ROOT=/mnt/d/output            # 目标根
#   ONLY=hypersim_provided             # 只搬某个子目录（默认全部）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${DEST_ROOT:=/mnt/d/output}"
DEST="$DEST_ROOT/infinisplat_v2_results_$(date +%Y%m%d)"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ ERROR: 结果目录不存在: $RESULTS_DIR" >&2
    exit 1
fi
mkdir -p "$DEST"

echo "🚀 [08] 搬结果到 Windows 盘"
echo "  📁 源: $RESULTS_DIR"
echo "  💾 目标: $DEST"

if [ -n "${ONLY:-}" ]; then
    ITEMS=("$RESULTS_DIR/$ONLY")
else
    ITEMS=("$RESULTS_DIR"/*)
fi

for src in "${ITEMS[@]}"; do
    [ -e "$src" ] || continue
    name=$(basename "$src")
    # 大场景 PLY 动辄几百 MB，用 rsync 增量；没有 rsync 就退回 cp
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --info=progress2 "$src" "$DEST/" && echo "  ✅ $name"
    else
        cp -r "$src" "$DEST/" && echo "  ✅ $name"
    fi
done

du -sh "$DEST" 2>/dev/null
echo "🎉 Done: $DEST"
