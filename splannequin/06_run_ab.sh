#!/usr/bin/env bash
# 06_run_ab.sh — 加速方案 A/B 串行实验链（double_wave 同场景对照）。
#   A = 8k 等比缩放（arguments_splannequin_8k.py，唯一变量=迭代数 ×0.8）
#   B = 10k + densify 阈值×2（arguments_splannequin_10k_dgx2.py，唯一变量=点数预算）
# 每个实验：训练（关周期评测）→ 04_eval r2 训练口径 → 04_eval r1 公平口径。
# 独立成脚本的原因：多行内联脚本经 wsl.exe 参数传递会被拆坏（变量预展开丢失），
# 落盘成文件用单行命令调用可绕开（2026-10-27 实测踩坑）。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP=/home/velvet/output/splannequin_results/exp
DATA=/home/velvet/output/splannequin_results/data/double_wave_colmap

run_one() {
    local NAME="$1" CONFIG="$2"
    echo "🚀 [06] ${NAME} 训练开始 $(date +%H:%M:%S)"
    SCENE_DIR="$DATA" SCENE_NAME="$NAME" RESOLUTION=2 CONFIG_FILE="$CONFIG" GPU=0 \
      EXTRA_TRAIN_ARGS="--test_iterations 999999" \
      bash "$SCRIPT_DIR/02_train.sh" 2>&1 | tee "$EXP/${NAME}_train.log" | tail -3
    echo "🚀 [06] ${NAME} 评测 r2（训练口径）"
    MODEL_DIR_PATH="$EXP/splannequin/$NAME" RESOLUTION=2 bash "$SCRIPT_DIR/04_eval_psnr.sh" | grep -aE "✅|❌|PSNR|saved"
    echo "🚀 [06] ${NAME} 评测 r1（公平口径）"
    MODEL_DIR_PATH="$EXP/splannequin/$NAME" RESOLUTION=1 bash "$SCRIPT_DIR/04_eval_psnr.sh" | grep -aE "✅|❌|PSNR|saved"
    echo "🎉 [06] ${NAME} 完成 $(date +%H:%M:%S)"
}

run_one double_wave_colmap_8k_r2        "$SCRIPT_DIR/arguments_splannequin_8k.py"
run_one double_wave_colmap_10k_dgx2_r2 "$SCRIPT_DIR/arguments_splannequin_10k_dgx2.py"
echo "🎉 ALL_AB_DONE"
