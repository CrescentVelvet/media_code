#!/usr/bin/env bash
# 03c_render_3view_orbit.sh — 三图 ZipSplat / SplatAnything 单向 360° 同轨迹视频
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
EVAL_PYTHON="${EVAL_PYTHON:-$HOME/miniconda3/envs/querysplat/bin/python}"
EVAL_ENV="${EVAL_ENV:-$HOME/miniconda3/envs/querysplat}"
[ -x "$EVAL_PYTHON" ] || { echo "❌ 缺渲染解释器: $EVAL_PYTHON" >&2; exit 1; }
mkdir -p "${OUT:-$HOME/output/zipsplat_results/jiajia_3v_orbit_360}"
CUDA_HOME="$EVAL_ENV" CPATH="$EVAL_ENV/targets/x86_64-linux/include" \
  LIBRARY_PATH="$EVAL_ENV/targets/x86_64-linux/lib" \
  "$EVAL_PYTHON" "$SCRIPT_DIR/render_3view_orbit.py"
