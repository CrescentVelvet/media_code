#!/usr/bin/env bash
# 08d_finetune_scene.sh — scene 高斯定向 finetune（治帧24-33大yaw欠拟合）
# 用法: bash 08d_finetune_scene.sh
set -eo pipefail   # 不用 -u：proxy.env 的 PYTHONPATH 追加引用未绑定变量会误报
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$SCRIPT_DIR" source "$SCRIPT_DIR/_env.sh"

export SCENE_PLY="${SCENE_PLY:-$RESULTS_DIR/07_body_gs_src/scene_gs.ply}"
export OUT_DIR="${OUT_DIR:-$RESULTS_DIR/08d_finetune_scene}"
export EPOCHS="${EPOCHS:-30}"
export LR="${LR:-1e-3}"
export LAMBDA_INSIDE="${LAMBDA_INSIDE:-0.3}"
export MAX_GAUSS="${MAX_GAUSS:-1500000}"

echo "▶ [08d] scene finetune"
echo "  SCENE_PLY=$SCENE_PLY"
echo "  OUT_DIR=$OUT_DIR  EPOCHS=$EPOCHS  LR=$LR"
python "$SCRIPT_DIR/finetune_scene.py"
echo "✓ [08d] 完成"
