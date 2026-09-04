#!/usr/bin/env bash
# 06e_closeup_inject.sh — 增强近景整视角注入 → 3DGS 续训（一条龙）
#
# 背景（AGENTS.md §12）：masked-L1 人脸监督无效，但把 06c 的 512 HYPIR 增强近景
# 作为完整训练视角（真实位姿注入 COLMAP）参与整图 loss 实测有效——近景 LPIPS 全降、
# 锐度 +28%~324%，训练视角无退化（06f 对照排除续训本身贡献）。
#
# 前提:
#   04/04b 训完; 06c_512_enhanced/ 含 06c_closeup_poses_p*.json + enhanced_pXX/images/
#   （由 06c_closeup_finetune.sh + face_enhance.py WHOLE_IMAGE=1 产出）
# Env (all optional, defaults shown):
#   RESULTS_DIR=                 # 输出根
#   SOURCE_DIR=                  # COLMAP 场景 (默认 03b_source_ba)
#   GAUSSIAN_DIR=                # 基线模型 (默认 04b_model_3dgs_ba)
#   ITERATION=30000              # 起始 checkpoint
#   EXTRA_ITERS=20000            # 续训步数
#   LR_SCALE=0.2                 # 学习率缩放
#   CLOSEUP_ROOT=                # 近景数据根 (默认 $RESULTS_DIR/06c_512_enhanced)
#   PERSONS=                     # 可选 pid 过滤, 如 p00,p01,p02 (默认全部)
#   06E_SOURCE_DIR=              # 注入场景输出 (默认 $RESULTS_DIR/06e_source_closeup)
#   GAUSSIAN_CONTINUE_DIR=       # 输出模型 (默认 $RESULTS_DIR/06e_model_3dgs_closeup)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR required}"
SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/03b_source_ba}"
GAUSSIAN_DIR="${GAUSSIAN_DIR:-$RESULTS_DIR/04b_model_3dgs_ba}"
ITERATION="${ITERATION:-30000}"
EXTRA_ITERS="${EXTRA_ITERS:-20000}"
LR_SCALE="${LR_SCALE:-0.2}"
CLOSEUP_ROOT="${CLOSEUP_ROOT:-$RESULTS_DIR/06c_512_enhanced}"
PERSONS="${PERSONS:-}"
SCENE_OUT="${06E_SOURCE_DIR:-$RESULTS_DIR/06e_source_closeup}"
GAUSSIAN_CONTINUE_DIR="${GAUSSIAN_CONTINUE_DIR:-$RESULTS_DIR/06e_model_3dgs_closeup}"

START_PLY="$GAUSSIAN_DIR/point_cloud/iteration_$ITERATION/point_cloud.ply"
[ -f "$START_PLY" ] || { echo "❌ start ply not found: $START_PLY" >&2; exit 1; }
[ -d "$CLOSEUP_ROOT" ] || { echo "❌ closeup root not found: $CLOSEUP_ROOT" >&2; exit 1; }

echo "💉 [06e] 近景整视角注入 + 续训"
echo "  📂 scene in:   $SOURCE_DIR"
echo "  📂 closeup:    $CLOSEUP_ROOT"
echo "  📂 scene out:  $SCENE_OUT"
echo "  📂 model out:  $GAUSSIAN_CONTINUE_DIR (iter $ITERATION -> $((ITERATION+EXTRA_ITERS)))"
echo ""

# 1) 注入（inject_closeup_cameras.py 自带脸部中心投影 sanity）
INJECT_ARGS=(
    --source_dir "$SOURCE_DIR"
    --poses_dir "$CLOSEUP_ROOT"
    --enhanced_root "$CLOSEUP_ROOT"
    --out_dir "$SCENE_OUT"
)
[ -n "$PERSONS" ] && INJECT_ARGS+=(--persons "$PERSONS")
python "$SCRIPT_DIR/inject_closeup_cameras.py" "${INJECT_ARGS[@]}" || {
    echo "❌ inject failed" >&2; exit 1; }

# 2) 续训（06d 配方，SOURCE_DIR 换成注入场景）
RESULTS_DIR="$RESULTS_DIR" \
SOURCE_DIR="$SCENE_OUT" \
GAUSSIAN_DIR="$GAUSSIAN_DIR" \
ITERATION="$ITERATION" EXTRA_ITERS="$EXTRA_ITERS" LR_SCALE="$LR_SCALE" \
FACE_WEIGHT=0 \
GAUSSIAN_CONTINUE_DIR="$GAUSSIAN_CONTINUE_DIR" \
    bash "$SCRIPT_DIR/06d_continue_train.sh" || {
    echo "❌ continue-train failed" >&2; exit 1; }

echo ""
echo "✅ [06e] Done. model:"
echo "  $GAUSSIAN_CONTINUE_DIR/point_cloud/iteration_$((ITERATION+EXTRA_ITERS))/point_cloud.ply"
