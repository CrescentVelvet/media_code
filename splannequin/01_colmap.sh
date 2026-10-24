#!/usr/bin/env bash
# 01_colmap.sh — 对自备数据跑 COLMAP（官方 scripts/colmap.py 的包装）。
#
# 输入目录结构（官方约定）：
#   <SCENE_DIR>/images/*.png
# 输出：
#   <SCENE_DIR>/sparse/  database.db  colmap_log.txt
#
# colmap 二进制来自独立 conda env `colmap`（conda-forge colmap 4.2.1 cuda_129；
# 见 som_human README：需补 openimageio=3.1，直调要 LD_LIBRARY_PATH=env/lib）。
# 本脚本用 splannequin env 的 python 跑 colmap.py（只用标准库），
# 把 colmap env 的 bin 加进 PATH 供其调起二进制。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

SCENE_DIR="${SCENE_DIR:?❌ SCENE_DIR not set (dir containing images/)}"
COLMAP_ENV="${COLMAP_ENV:-colmap}"

if [ ! -d "$SCENE_DIR/images" ]; then
    echo "❌ ERROR: $SCENE_DIR/images not found" >&2
    echo "       Expected layout: <SCENE_DIR>/images/0001.png ..." >&2
    exit 1
fi

COLMAP_BIN_DIR="$(conda info --base)/envs/$COLMAP_ENV/bin"
if [ ! -x "$COLMAP_BIN_DIR/colmap" ]; then
    echo "❌ ERROR: colmap not found in env '$COLMAP_ENV'" >&2
    echo "       Install: conda create -n colmap --override-channels -c conda-forge 'colmap=*cuda*' openimageio=3.1 -y" >&2
    exit 1
fi
export PATH="$COLMAP_BIN_DIR:$PATH"
# colmap 4.x 直调二进制需要 env 自己的 lib 优先
export LD_LIBRARY_PATH="$(conda info --base)/envs/$COLMAP_ENV/lib:${LD_LIBRARY_PATH:-}"

echo "🚀 [01] COLMAP reconstruction"
echo "  📁 scene: $SCENE_DIR"
echo "  📸 images: $(ls "$SCENE_DIR/images" | wc -l) files"
echo "  🔧 colmap: $(colmap -v 2>/dev/null | head -1 || echo "$COLMAP_BIN_DIR/colmap")"

python "$SPLANNEQUIN_DIR/scripts/colmap.py" "$SCENE_DIR" --gpu
if [ $? -ne 0 ]; then
    echo "❌ FAILED (see $SCENE_DIR/colmap_log.txt)" >&2
    exit 1
fi
echo "🎉 Done. sparse model at $SCENE_DIR/sparse/"
