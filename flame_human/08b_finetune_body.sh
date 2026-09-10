#!/usr/bin/env bash
# 08b_finetune_body.sh — body 高斯在 59 帧 finetune（修 nn 模型 yaw 欠拟合
# 带来的静态偏差 + 动态姿态的平均化吸收）。
#
# 输入：07 的 body_p00.ply（07a 投票提取 + 07 补缝）
# 监督：p0 person mask 区域（01c alpha）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🧍 [flame_human 08b] body finetune"

OUT_DIR="${OUT_DIR:-$RESULTS_DIR/08b_finetune_body}" \
EPOCHS="${EPOCHS:-60}" \
PID="${PID:-0}" \
    python "$SCRIPT_DIR/finetune_body.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: 08b 失败"
    exit 1
fi

echo "🎉 Done. 下一步：composite_check.py 换 BODY_PLY 验证"
