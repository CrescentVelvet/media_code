#!/usr/bin/env bash
# 27_run_paired_eval.sh -- 只读已有前向，显式启用 CUDA 渲染；不训练、不改 PLY。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/d/output/results/splatanything_results/densitok_portrait_probe/protocol_v5}"
OUT="${OUT:-${OUT:?必须显式指定全新的评测输出目录}}"
export CONDA_ENV=splatanything
source "$SCRIPT_DIR/_env.sh"
source "$SCRIPT_DIR/_gsplat_env.sh"
if [ ! -x "$CONDA_PREFIX/bin/python" ] || [ -z "${CUDA_HOME:-}" ]; then
  echo "❌ gsplat/CUDA 环境未就绪" >&2
  exit 1
fi
"$CONDA_PREFIX/bin/python" -c 'import torch; assert torch.cuda.is_available()' || exit 1
ARGS=(--root "$ROOT" --out "$OUT" --official-repo "$SPLATANYTHING_DIR" --device cuda)
if [ -n "${ONLY_PACK:-}" ]; then
  ARGS+=(--only-pack "$ONLY_PACK")
fi
"$CONDA_PREFIX/bin/python" "$SCRIPT_DIR/27_evaluate_paired.py" "${ARGS[@]}" || exit 1
echo "🎉 回放评测已生成；重复渲染已检查，仍需逐图 QA、物体控制及最终晋级判断。"
