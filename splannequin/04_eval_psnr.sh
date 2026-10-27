#!/usr/bin/env bash
# 04_eval_psnr.sh — 独立评测：复刻 train.py 的 test PSNR/L1（训练外用）。
# 用于 --test_iterations 999999 关掉训练内评测后补测，或跨分辨率公平对比。
#
# 用法：
#   MODEL_DIR_PATH=<模型目录> bash 04_eval_psnr.sh          # 训练分辨率口径
#   MODEL_DIR_PATH=<模型目录> RESOLUTION=1 bash 04_eval_psnr.sh  # 全分辨率口径
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

MODEL_DIR_PATH="${MODEL_DIR_PATH:?❌ MODEL_DIR_PATH not set (dir with point_cloud/ + cfg_args.json)}"
export MODEL_DIR_PATH

echo "🚀 [04] Eval test PSNR"
echo "  🏋️ model: $MODEL_DIR_PATH"
echo "  📐 resolution: ${RESOLUTION:-cfg_args 原值}"

cd "$SPLANNEQUIN_DIR"
python "$SCRIPT_DIR/04_eval_psnr.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 Done."
