#!/usr/bin/env bash
# 08_train.sh — 阶段六/八：表情驱动的 AvatarGaussian 训练（单卡）。
#
# 与标准 3DGS 的三处不同见 train_avatar.py 头注释：
#   位置硬绑定（p = bary @ V_f，不是自由参数）
#   local_exp 作为可训练参数被渲染 loss refine
#   densify 时子点继承 face_id / bary
# epoch 48（= EPOCHS-12）触发 fix_and_sync_exp + checkpoint（变差可回滚）。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

PID="${PID:-0}"   # 与 06 产物 avatar_p0.ply 对齐（曾因默认 00 找不到文件）
echo "🚀 [flame_human 08] 阶段六 AvatarGaussian 训练 (p${PID})"
echo "  📦 高斯 : ${AVATAR_PLY:-$RESULTS_DIR/06_avatar_gs/avatar_p${PID}.ply}"
echo "  🗿 FLAME: $FLAME_MODEL"
echo "  💾 输出 : ${OUT_DIR:-$RESULTS_DIR/08_train}"

if [ ! -f "${AVATAR_PLY:-$RESULTS_DIR/06_avatar_gs/avatar_p${PID}.ply}" ]; then
    echo "❌ 缺 AvatarGaussian 初始 PLY，先跑 bash 06_init_avatar_gs.sh"
    exit 1
fi
if [ ! -d "$GS_DIR" ]; then
    echo "❌ 缺 gaussian-splatting: $GS_DIR（先跑 00a_setup_env.sh）"
    exit 1
fi

AVATAR_PLY="${AVATAR_PLY:-$RESULTS_DIR/06_avatar_gs/avatar_p${PID}.ply}" \
BIND_NPZ="${BIND_NPZ:-}" \
ALIGN_JSON="${ALIGN_JSON:-$RESULTS_DIR/05_align/head_align.json}" \
IMAGES_DIR="${IMAGES_DIR:-$SOURCE_DIR/images}" \
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/08_train}" \
PID="$PID" \
EPOCHS="${EPOCHS:-60}" \
SH_DEGREE="${SH_DEGREE:-3}" \
FACE_ENHANCE_EPOCH="${FACE_ENHANCE_EPOCH:-48}" \
LR="${LR:-1e-3}" \
LR_EXP="${LR_EXP:-1e-3}" \
DENSIFY_FROM="${DENSIFY_FROM:-5}" \
DENSIFY_UNTIL="${DENSIFY_UNTIL:-40}" \
DENSIFY_EVERY="${DENSIFY_EVERY:-5}" \
DENSIFY_GRAD="${DENSIFY_GRAD:-2e-4}" \
MIN_OPACITY="${MIN_OPACITY:-0.005}" \
LAMBDA_DSSIM="${LAMBDA_DSSIM:-0.2}" \
SH_EVERY="${SH_EVERY:-10}" \
SEED="${SEED:-0}" \
    python "$SCRIPT_DIR/train_avatar.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: 训练失败"
    exit 1
fi

echo "🎉 Done."
