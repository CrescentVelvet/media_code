#!/usr/bin/env bash
# 08e_prune_all.sh — 三分支投票/尺度剪枝（治人物周围黑雾 floater/幽灵点）
#
# 定稿配方（2026-09-10 逐分支消融归因）：
#   head : 尺度门控 SCALE_MAX=0.02 —— **收益全部来源**（330 个巨型高斯，
#          max scale 0.21 vs 头半径 0.13，opacity~0.7，糊出大片暗雾；
#          位置投票 0 删除说明"位置没飞、是尺度炸了"）
#   scene: 反向投票 DROP_RATIO=0.85 —— 只删「>85% 视角落在 person 内」的
#          幽灵点（阈值 0.6 会误删"被人物长期遮挡的合法背景"→ 轮廓空洞）
#   body : 默认**关闭**（消融显示无关/略有害）；需用时 BODY_PRUNE=1
#
# ⚠️ scale 门控不可通用：scene 加 SCALE_MAX=0.03 直接崩到 16.37dB
#    （scene 的大尺度高斯是墙/桌/地面等合法大平面）
#
# 用法: bash 08e_prune_all.sh
set -eo pipefail   # 不用 -u：proxy.env 的 PYTHONPATH 追加引用未绑定变量会误报
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$SCRIPT_DIR" source "$SCRIPT_DIR/_env.sh"

PID="${PID:-0}"
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/08e_pruned}"
mkdir -p "$OUT_DIR"

echo "▶ [08e] 三分支剪枝 → $OUT_DIR"

# 1. head：尺度门控（收益主力）+ 反 floater 投票
MODE=head \
  IN_CKPT="${HEAD_CKPT_IN:-$RESULTS_DIR/08_train/avatar_p${PID}_final.pth}" \
  OUT_CKPT="$OUT_DIR/avatar_pruned_p${PID}.pth" \
  KEEP_RATIO="${HEAD_KEEP_RATIO:-0.5}" \
  SCALE_MAX="${HEAD_SCALE_MAX:-0.02}" \
  PID="$PID" python "$SCRIPT_DIR/08e_prune.py"

# 2. scene：反向投票删人像区幽灵点（高阈值，避免误删被遮挡背景）
MODE=scene \
  IN_PLY="${SCENE_PLY_IN:-$RESULTS_DIR/08d_finetune_scene/scene_ft_p${PID}.ply}" \
  OUT_PLY="$OUT_DIR/scene_pruned_p${PID}.ply" \
  DROP_RATIO="${SCENE_DROP_RATIO:-0.85}" \
  PID="$PID" python "$SCRIPT_DIR/08e_prune.py"

# 3. body（可选，默认关闭）：轮廓外 floater + 头盒内点
if [ "${BODY_PRUNE:-0}" = "1" ]; then
  MODE=body \
    IN_PLY="${BODY_PLY:-$RESULTS_DIR/07_body_gs_src/body_gs_p${PID}.ply}" \
    OUT_PLY="$OUT_DIR/body_pruned_p${PID}.ply" \
    KEEP_RATIO="${BODY_KEEP_RATIO:-0.5}" \
    PID="$PID" python "$SCRIPT_DIR/08e_prune.py"
else
  echo "  ⏭️  跳过 body 剪枝（BODY_PRUNE=1 可启用；消融显示无关/略有害）"
fi

echo "✓ [08e] 完成 → $OUT_DIR"
