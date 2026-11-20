#!/usr/bin/env bash
# 03b_eval_3view_compare.sh — ZipSplat 三图与历史 DA3/SA 同 512² 留出评测
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
EVAL_PYTHON="${EVAL_PYTHON:-$HOME/miniconda3/envs/querysplat/bin/python}"
EVAL_ENV="${EVAL_ENV:-$HOME/miniconda3/envs/querysplat}"
OUT_DIR="${OUT_DIR:-$HOME/output/zipsplat_results/eval_3view_r512_black}"
CASE_FILTER="${CASE_FILTER:-ZipSplat}"
RESULT_NAME="${RESULT_NAME:-${CASE_FILTER// /_}_metrics.json}"
[ -x "$EVAL_PYTHON" ] || { echo "❌ 缺评测解释器: $EVAL_PYTHON" >&2; exit 1; }
python "$SCRIPT_DIR/prepare_3view_compare.py" || exit 1
mkdir -p "$OUT_DIR"
echo "🚀 [03b] 3 输入 / 32 GT 相机 / 29 留出 / 中心方形 512² / 黑底"
echo "  🎯 $CASE_FILTER（相机仅 source-only 对齐，不进入模型）"
CUDA_HOME="$EVAL_ENV" CPATH="$EVAL_ENV/targets/x86_64-linux/include" \
  LIBRARY_PATH="$EVAL_ENV/targets/x86_64-linux/lib" \
  EXTRA_JSON="$OUT_DIR/extra_methods.json" ONLY="$CASE_FILTER" \
  OUT="$OUT_DIR/$RESULT_NAME" CROP_SQ=1 REFINE=1 EVAL_RES=512 BG=auto \
  SAVE_DIR="$OUT_DIR/renders" SAVE_VIEWS=2,21 \
  "$EVAL_PYTHON" "$MEDIA_CODE_ROOT/compare_4methods/audit_31_eval_3view.py" || exit 1
echo "🎉 Done: $OUT_DIR/$RESULT_NAME"
