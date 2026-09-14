#!/usr/bin/env bash
# 08c_prune_body.sh — 08b body finetune 产物的后处理剪枝（头盒 3D + mask 投票）。
#
# ⚠️ 背景：08b/08c 是**负结果**分支（body finetune 后 composite 反而变差，
# 见 EXPERIMENTS.md 2026-09-10）。当前推荐链路**不使用**本步骤，脚本保留
# 用于复现实验与后续排查 floater 的工具复用。
#
# 用法: bash 08c_prune_body.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$SCRIPT_DIR" source "$SCRIPT_DIR/_env.sh"

PID="${PID:-0}"
echo "🚀 [flame_human 08c] body 产物剪枝"
echo "  📥 输入: ${IN_PLY:-$RESULTS_DIR/08b_finetune_body/body_ft_p$PID.ply}"
echo "  💾 输出: ${OUT_PLY:-$RESULTS_DIR/08c_pruned/body_ft_pruned_p$PID.ply}"

IN_PLY="${IN_PLY:-$RESULTS_DIR/08b_finetune_body/body_ft_p$PID.ply}" \
OUT_PLY="${OUT_PLY:-$RESULTS_DIR/08c_pruned/body_ft_pruned_p$PID.ply}" \
ALIGN_JSON="${ALIGN_JSON:-$RESULTS_DIR/05_align/head_align.json}" \
SOURCE_DIR="${SOURCE_DIR:-}" \
MASKS_DIR="${MASKS_DIR:-$RESULTS_DIR/01c_sam3_person_masks}" \
MESH_DIR="${MESH_DIR:-$RESULTS_DIR/06_avatar_gs}" \
KEEP_RATIO="${KEEP_RATIO:-0.5}" \
HEAD_BBOX_MARGIN="${HEAD_BBOX_MARGIN:-0.15}" \
PID="$PID" \
    python "$SCRIPT_DIR/08c_prune_body.py"
if [ $? -ne 0 ]; then
    echo "❌ [08c] 剪枝失败" >&2
    exit 1
fi
echo "🎉 [08c] Done."
