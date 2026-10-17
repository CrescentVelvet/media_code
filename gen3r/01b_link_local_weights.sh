#!/usr/bin/env bash
# 01b_link_local_weights.sh — 用迅雷/IDM 下到 D:\wheel\gen3r_ms 后，校验并接到 ~/model/gen3r
#
# 两种接法（默认 link = 直接指过去，省 37GB 复制）：
#   LINK_MODE=link   MODEL_DIR 指到 /mnt/d/wheel/gen3r_ms（读取走 drvfs，稍慢）
#   LINK_MODE=copy   复制到 ~/model/gen3r（Linux fs，加载快，耗时几分钟）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

SRC="${SRC:-/mnt/d/wheel/gen3r_ms}"
LINK_MODE="${LINK_MODE:-link}"
MODEL_DIR="${MODEL_DIR:-$HOME/model/gen3r}"

echo "🚀 [01b] 校验本地权重"
echo "  📥 src:  $SRC"
echo "  💾 dest: $MODEL_DIR (mode=$LINK_MODE)"

if [ ! -d "$SRC" ]; then
    echo "❌ ERROR: $SRC 不存在（先把 download_urls.md 里的文件按相对路径下到这里）" >&2
    exit 1
fi

MISSING=0
python "$SCRIPT_DIR/verify_weights.py" --root "$SRC" || MISSING=1
if [ "$MISSING" -ne 0 ]; then
    echo "❌ 文件不全，见上方列表" >&2
    exit 1
fi

if [ "$LINK_MODE" = "link" ]; then
    echo "🎉 权重齐了。推理时直接指定："
    echo "   MODEL_DIR=$SRC bash gen3r/02_run_inference.sh"
    exit 0
fi

echo "📦 复制到 $MODEL_DIR（37.7GB，几分钟）"
mkdir -p "$MODEL_DIR"
cp -r "$SRC"/. "$MODEL_DIR"/
echo "🎉 Done: $MODEL_DIR"
