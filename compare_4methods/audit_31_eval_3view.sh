#!/usr/bin/env bash
# audit_31_eval_3view.sh — 三图五方法评测的统一环境包装（复用 94 的 CUDA/conda 包装）
# 前置: 90a（is qs vggt da3 四步骤）+ 90b（DA3-GS 全 SH 导出 + 98 相机）全部完成
# 用法:
#   bash compare_4methods/audit_31_eval_3view.sh                 # 全五方法
#   ONLY=querysplat bash compare_4methods/audit_31_eval_3view.sh # 单方法调试
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 [audit_31] 三图评测（统一渲 32 GT 相机 × 五方法）"
bash "$SCRIPT_DIR/94_render.sh" "$SCRIPT_DIR/audit_31_eval_3view.py"
RC=$?
if [ $RC -ne 0 ]; then
    echo "❌ FAILED (exit $RC)" >&2
    exit 1
fi
echo "🎉 Done."
