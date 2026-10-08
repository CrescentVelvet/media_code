#!/usr/bin/env bash
# run_all.sh — 一键 pipeline：01 准备数据 → 02 vanilla 训练 → 03 渲染评测。
# ⚠️ 不含环境搭建：首次先跑 00a_setup_env.sh（WSL）或 00_setup_env.sh（服务器）。
#
# 用法：
#   GPU=0 VIDEO_PATH=/mnt/d/dataset/xxx.mp4 SCENE_NAME=human_seq \
#     bash deformable_human/run_all.sh
# 各步 env var（VIDEO_FPS / ITERATIONS / MODE 等）原样透传，见各脚本头部注释。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 [run_all] deformable_human: 01 → 02 → 03"
echo ""

bash "$SCRIPT_DIR/01_prepare_data.sh" || { echo "❌ [01] 失败，中止" >&2; exit 1; }
echo ""
bash "$SCRIPT_DIR/02_train.sh"        || { echo "❌ [02] 失败，中止" >&2; exit 1; }
echo ""
bash "$SCRIPT_DIR/03_render.sh"       || { echo "❌ [03] 失败，中止" >&2; exit 1; }

echo ""
echo "🎉 [run_all] 全流程完成。"
echo "  💡 WSL 上训练产物在 Linux fs，记得: bash deformable_human/08_move_output.sh"
