#!/usr/bin/env bash
# audit_31_eval_3view.sh — 三图五方法评测的 querysplat env 包装（照 audit_24.sh 模式）
# 前置: 90a（is qs vggt da3 四步骤）+ 90b（DA3-GS 全 SH 导出 + 98 相机）全部完成
# 用法:
#   bash compare_4methods/audit_31_eval_3view.sh                 # 全五方法
#   ONLY=querysplat bash compare_4methods/audit_31_eval_3view.sh # 单方法调试
set -o pipefail
CODE=/mnt/c/code/media_code
ENVS=/home/velvet/miniconda3/envs/querysplat

if [ -f /home/velvet/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1090
  source /home/velvet/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENVS"
fi
export CUDA_HOME="$ENVS"
export CPATH="$ENVS/targets/x86_64-linux/include"
export LIBRARY_PATH="$ENVS/targets/x86_64-linux/lib"

echo "🚀 [audit_31] 三图评测（统一渲 32 GT 相机 × 五方法）"
ONLY="${ONLY:-}" OUT="${OUT:-}" GT_CAMS="${GT_CAMS:-}" BASE3="${BASE3:-}" \
  CROP_SQ="${CROP_SQ:-}" REFINE="${REFINE:-}" REFINE_RES="${REFINE_RES:-}" \
  REFINE_ITERS="${REFINE_ITERS:-}" EVAL_RES="${EVAL_RES:-}" \
  exec python "$CODE/compare_4methods/audit_31_eval_3view.py"
