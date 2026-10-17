#!/usr/bin/env bash
# 01b_link_local_weights.sh — 用迅雷/IDM 下到 D:\wheel\gen3r_ms 后，校验并接到 ~/model/gen3r
#
# 迅雷批量下载常把 text_encoder/xxx 摊平成 xxx 放根目录，本脚本按**文件名**兜底匹配，
# 再统一整理成上游要的相对路径结构（软链接），不再要求你手工保持子目录。
#
#   LINK_MODE=link   MODEL_DIR 里建软链指向 /mnt/d（默认，0 拷贝）
#   LINK_MODE=copy   真正复制到 ~/model/gen3r（Linux fs，加载快，几分钟）
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
    echo "❌ ERROR: $SRC 不存在（先把 download_urls.txt 里的 URL 批量下载到这里）" >&2
    exit 1
fi

if [ "$LINK_MODE" = "copy" ]; then
    TREE="$(mktemp -d "$MODEL_DIR.tree.XXXXXX")"
    python "$SCRIPT_DIR/verify_weights.py" --root "$SRC" --link-into "$TREE" || {
        echo "❌ 文件不全，见上方列表" >&2; rm -rf "$TREE"; exit 1; }
    echo "📦 复制到 $MODEL_DIR（37.7GB，几分钟）"
    mkdir -p "$MODEL_DIR"
    cp -rL "$TREE"/. "$MODEL_DIR"/
    rm -rf "$TREE"
    echo "🎉 Done: $MODEL_DIR"
else
    python "$SCRIPT_DIR/verify_weights.py" --root "$SRC" --link-into "$MODEL_DIR" || {
        echo "❌ 文件不全，见上方列表" >&2; exit 1; }
    echo "🎉 权重齐了。直接跑："
    echo "   bash gen3r/02_run_inference.sh"
fi
