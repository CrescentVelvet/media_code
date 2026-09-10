#!/usr/bin/env bash
# 05_align_3dmm.sh — 阶段四：FLAME 三阶段对齐（global / local / coeff）。
#
# 前置：01b（lm468 嵌入）、04（face_recon.json）。
# 关键修正（详见脚本头注释）：
#   4.1 固定 local_SRT 为 PnP 初值 → 残差才反映 landmark 质量
#   reject 从 4.1 推迟到 4.2 之后 → 不再系统性砍掉大角度转头帧
#   >55° 的帧重置为 PnP 初值（不是归零）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [flame_human 05] 阶段四 3DMM 三阶段对齐"
echo "  📥 输入: ${RECON_JSON:-$RESULTS_DIR/04_recon/face_recon.json}"
echo "  🗿 FLAME: $FLAME_MODEL"
echo "  💾 输出: ${OUT_JSON:-$RESULTS_DIR/05_align/head_align.json}"

for f in "$FLAME_MODEL" "$FLAME_LM468_EMBEDDING"; do
    if [ ! -f "$f" ]; then
        echo "❌ 缺少: $f"
        echo "   → 先跑 01_download_models.sh 与 01b_build_lm468_embedding.sh"
        exit 1
    fi
done

RECON_JSON="${RECON_JSON:-$RESULTS_DIR/04_recon/face_recon.json}" \
OUT_JSON="${OUT_JSON:-$RESULTS_DIR/05_align/head_align.json}" \
GLOBAL_ITERS="${GLOBAL_ITERS:-300}" \
LOCAL_ITERS="${LOCAL_ITERS:-300}" \
COEFF_ITERS="${COEFF_ITERS:-3000}" \
# 4.3 gamma 放缓（2026-09-10）：原 0.95 逐迭代衰减使 300 步后 LR 剩 2e-7、
# 后 200 步空转 → 全局欠训练（26.5px）；0.9995 + 3000 步实测降到 ~7px
COEFF_GAMMA="${COEFF_GAMMA:-0.9995}" \
LR_GLOBAL="${LR_GLOBAL:-1e-2}" \
LR_LOCAL="${LR_LOCAL:-1e-2}" \
LR_COEFF="${LR_COEFF:-5e-3}" \
REJECT_EXTENT="${REJECT_EXTENT:-5.0}" \
RESET_DEG="${RESET_DEG:-55}" \
ANCHOR_W_LOCAL="${ANCHOR_W_LOCAL:-0.005}" \
ANCHOR_W_COEFF="${ANCHOR_W_COEFF:-0.01}" \
LAM_ID="${LAM_ID:-1e-4}" \
LAM_EXP="${LAM_EXP:-1e-3}" \
DROP_OUTBOUND="${DROP_OUTBOUND:-1}" \
MIN_FRAMES="${MIN_FRAMES:-8}" \
    python "$SCRIPT_DIR/align_3dmm.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: 阶段四失败"
    exit 1
fi

echo "🎉 Done. 下一步：bash 06_init_avatar_gs.sh"
