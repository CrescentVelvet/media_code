#!/usr/bin/env bash
# 26_prepare_paired.sh -- 只生成配对协议与新身份原生 ROI，不跑模型。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/d/output/results/splatanything_results/densitok_portrait_probe/protocol_v1}"
NEW_GT_PKG="${NEW_GT_PKG:-}"
export ROOT NEW_GT_PKG
CONDA_ENV=splatanything source "$SCRIPT_DIR/_env.sh"
if [ ! -f "$ROOT/manifest.json" ]; then
  "$HOME/miniconda3/envs/splatanything/bin/python" "$SCRIPT_DIR/26_prepare_paired.py" || exit 1
elif [ -n "${ONLY_LANDMARKS:-}" ]; then
  echo "✅ 协议已冻结；只继续地标检测，不重写 manifest。"
else
  echo "❌ 协议已存在；不得覆盖。只补地标时设 ONLY_LANDMARKS=1。" >&2
  exit 1
fi
if [ -n "$NEW_GT_PKG" ]; then
  "$HOME/miniconda3/envs/sam3/bin/python" "$SCRIPT_DIR/26b_new_gt_landmarks.py" || exit 1
fi
echo "🎉 输入协议已冻结；新身份 ROI 目检通过后才运行模型。"
