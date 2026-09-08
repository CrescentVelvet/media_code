#!/usr/bin/env bash
# 01b_build_lm468_embedding.sh — 生成 MediaPipe 468 点 → FLAME 顶点的重心嵌入。
#
# 阶段四要用 468 点算重投影残差（68 点解不了 100 维 exp），而 FLAME 只自带 68 点嵌入，
# 所以现场用 MediaPipe 自带的 canonical_face_model.obj 与 FLAME mean face 对齐生成。
# 前置：01_download_models.sh 已通过（FLAME2020.pkl 就位）。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [flame_human 01b] 构建 MediaPipe 468 → FLAME 重心嵌入"
echo "  🗿 FLAME  : $FLAME_MODEL"
echo "  💾 输出   : $FLAME_LM468_EMBEDDING"

if [ ! -f "$FLAME_MODEL" ]; then
    echo "❌ 缺少 FLAME 模型: $FLAME_MODEL"
    echo "   → 先跑 bash 01_download_models.sh"
    exit 1
fi

OUT_NPZ="${OUT_NPZ:-$FLAME_LM468_EMBEDDING}" \
FLAME_MODEL="$FLAME_MODEL" \
CANONICAL_FACE_MODEL="${CANONICAL_FACE_MODEL:-}" \
ICP_ITERS="${ICP_ITERS:-50}" \
    python "$SCRIPT_DIR/build_lm468_embedding.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: 嵌入构建失败"
    exit 1
fi

echo "🎉 Done. 下一步：bash 02_detect_faces.sh"
