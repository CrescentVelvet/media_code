#!/usr/bin/env bash
# 09_enhance_post.sh — 渲染后 2D 增强（HYPIR 对人脸区域后处理）
#
# 动机（2026-09-11）：head 提锐度三路实验（①scale/SH ②HYPIR 监督 ③监督区）
# 全为负 —— 3DGS 侧的柔化是"信息缺失"，逐帧 2D 增强喂训练会被平均掉。
# 唯一能立刻见效的是**对渲染结果做 2D 后处理**：直接给输出补高频细节。
# 代价：逐帧独立增强 → 帧间可能闪烁（不保证时序一致）。
#
# 步骤：① render_composite.py 渲染三分支合成帧 → ② enhance_faces.py 用
#       03_match 的 p0 脸框裁剪 + HYPIR + 羽化融合回原图。
#
# 用法: bash 09_enhance_post.sh            # 全 59 帧
#       FRAMES=15,29 bash 09_enhance_post.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$SCRIPT_DIR" source "$SCRIPT_DIR/_env.sh"

RAW_DIR="${RAW_DIR:-$RESULTS_DIR/09_render/raw}"
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/09_render}"

echo "🚀 [09] 渲染 + HYPIR 后处理增强"
echo "  🎬 渲染 → $RAW_DIR"
echo "  🖼️  增强 → $OUT_DIR/images"

FRAMES="${FRAMES:-}" python "$SCRIPT_DIR/render_composite.py"
if [ $? -ne 0 ]; then
    echo "❌ [09] 渲染失败" >&2
    exit 1
fi

# HYPIR 依赖在 vggt_human 环境（同 01d）
ENHANCE_ENV="${ENHANCE_ENV:-vggt_human}"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$ENHANCE_ENV" 2>/dev/null && echo "  🔁 切换到 $ENHANCE_ENV 环境（HYPIR）"

MATCH_JSON="${MATCH_JSON:-$RESULTS_DIR/03_match/face_person_match.json}" \
SRC_IMAGES_DIR="$RAW_DIR" \
OUT_DIR="$OUT_DIR" \
HYPIR_DIR="$HYPIR_DIR" \
HYPIR_BASE_MODEL="$HYPIR_BASE_MODEL" \
HYPIR_WEIGHT="$HYPIR_WEIGHT" \
FACE_PADDING="${FACE_PADDING:-0.35}" \
UPSCALE="${UPSCALE:-2}" \
DEBUG_DIR="${DEBUG_DIR:-}" \
    python "$SCRIPT_DIR/enhance_faces.py"
if [ $? -ne 0 ]; then
    echo "❌ [09] 增强失败" >&2
    exit 1
fi
echo "🎉 [09] Done → $OUT_DIR/images"
