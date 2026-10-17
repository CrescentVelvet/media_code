#!/usr/bin/env bash
# 03_to_colmap.sh — Gen3R 产物 → COLMAP 场景（可直接喂 3DGS 训练）
#
# 为什么要这一步：Gen3R 只输出点云 + 相机 + 视频，不输出 Gaussian。
# 转成 COLMAP 文本格式后，vggt_human/04_train_3dgs.sh（原始 graphdeco 3DGS）
# 或任何吃 COLMAP 的 3DGS 实现都能直接训练。
#
# Env:
#   GEN3R_OUT   Gen3R 单次产物目录（含 rgb.mp4 / cameras.json / pcds.ply）
#   OUT_DIR     输出场景目录（默认 $GEN3R_OUT/colmap_scene）
#   MAX_POINTS  初始化点云上限（默认 200000）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

if [ -z "${GEN3R_OUT:-}" ]; then
    echo "❌ ERROR: GEN3R_OUT 未设置（指向 02 的产物目录，含 rgb.mp4）" >&2
    exit 1
fi
OUT_DIR="${OUT_DIR:-$GEN3R_OUT/colmap_scene}"
MAX_POINTS="${MAX_POINTS:-200000}"

echo "🚀 [03] Gen3R → COLMAP"
echo "  📥 in:  $GEN3R_OUT"
echo "  💾 out: $OUT_DIR"
echo "  🔵 max points: $MAX_POINTS"

GEN3R_OUT="$GEN3R_OUT" OUT_DIR="$OUT_DIR" MAX_POINTS="$MAX_POINTS" \
    python "$SCRIPT_DIR/convert_to_colmap.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi

echo "🎉 Done: $OUT_DIR"
echo "   训练示例：SOURCE_DIR=$OUT_DIR bash vggt_human/04_train_3dgs.sh"
