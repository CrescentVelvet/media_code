#!/usr/bin/env bash
# 04_preview.sh —— 把 PLY 投影成 6 视角 PNG 拼图，肉眼验形状
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

PLY="${PLY:-}"
OUT="${OUT:-}"
# 变量名不能叫 SIZE：conda gcc 的 activate 已把 SIZE 占成 binutils 的 size 工具路径
PREVIEW_SIZE="${PREVIEW_SIZE:-480}"

if [ -z "$PLY" ]; then
    echo "❌ ERROR: 必须指定 PLY（.ply 文件路径，或含 .ply 的目录）" >&2
    echo "   例: PLY=$RESULTS_DIR/fur-egg-chair.ply bash $0" >&2
    exit 1
fi

echo "🚀 [04] 渲染 PLY 预览"
shopt -s nullglob
if [ -d "$PLY" ]; then
    FILES=("$PLY"/*.ply)
else
    FILES=("$PLY")
fi
if [ ${#FILES[@]} -eq 0 ]; then
    echo "❌ ERROR: 没找到 .ply: $PLY" >&2
    exit 1
fi

for f in "${FILES[@]}"; do
    PLY="$f" OUT="${OUT:-${f%.ply}_preview.png}" PREVIEW_SIZE="$PREVIEW_SIZE" \
        python "$SCRIPT_DIR/04_preview_gaussian.py" || { echo "❌ $f 渲染失败" >&2; exit 1; }
    OUT=""
done

echo "🎉 [04] Done."
