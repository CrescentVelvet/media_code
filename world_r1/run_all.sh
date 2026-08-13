#!/usr/bin/env bash
# run_all.sh — World-R1 一键全流程: 验证环境 → 下权重 → 训练 → 推理。
#
# 用法 (默认 8 GPU: 2 server + 6 train):
#   bash world_r1/run_all.sh
#
# 4 GPU 示例:
#   SERVER_VISIBLE_DEVICES=0,1 TRAIN_VISIBLE_DEVICES=2,3 NUM_PROCESSES=2 \
#     bash world_r1/run_all.sh
#
# 跳过训练只做推理:
#   SKIP_TRAIN=1 LORA_PATH=../../model/world_r1_lora/checkpoint-60 \
#     bash world_r1/run_all.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 World-R1 全流程"
echo "========================================"

# 1. 环境搭建 + 验证
echo ""
echo "🔧 [1/4] 环境验证"
bash "$SCRIPT_DIR/00_setup_env.sh"
if [ $? -ne 0 ]; then echo "❌ 环境验证失败" >&2; exit 1; fi

# 2. 下载权重
echo ""
echo "📦 [2/4] 下载权重"
bash "$SCRIPT_DIR/01_download_models.sh"
if [ $? -ne 0 ]; then echo "❌ 权重下载失败" >&2; exit 1; fi

# 3. RL 训练
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    echo ""
    echo "🏋️ [3/4] RL 训练"
    bash "$SCRIPT_DIR/03_run_training.sh"
    if [ $? -ne 0 ]; then echo "❌ 训练失败" >&2; exit 1; fi
else
    echo ""
    echo "⏭️ [3/4] SKIP_TRAIN=1, 跳过训练"
fi

# 4. 推理
echo ""
echo "🎬 [4/4] 推理"
bash "$SCRIPT_DIR/04_run_inference.sh"
if [ $? -ne 0 ]; then echo "❌ 推理失败" >&2; exit 1; fi

echo ""
echo "🎉 全流程完成!"
echo "========================================"
