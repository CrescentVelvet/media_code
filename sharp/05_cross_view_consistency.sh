#!/usr/bin/env bash
# S1a 跨视角自洽 —— 三图包的同一批点在各视角之间互投影 + 跨视角渲染。
#
# 用法（必须经本 wrapper 跑：_env.sh 定 CUDA_HOME，否则 gsplat 静默禁用）：
#   GPU=0 SCENE=~/output/splatanything_results/_scenes/jiajia_hold_3v \
#     bash sharp/05_cross_view_consistency.sh
#
# ⚠️ 不用 set -u：proxy.env 里有 `export PYTHONPATH="...:$PYTHONPATH"`，
#    PYTHONPATH 未定义时会直接崩（03/04 也因此只开 -eo pipefail）。
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

SCENE="${SCENE:-$HOME/output/splatanything_results/_scenes/jiajia_hold_3v}"
TAG="${TAG:-$(basename "$SCENE")}"
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/S1a_crossview_$TAG}"
RENDER_SCALE="${RENDER_SCALE:-0.5}"
BG="${BG:-black}"

python "$SCRIPT_DIR/05_cross_view_consistency.py" \
    --scene "$SCENE" \
    --out "$OUT_DIR" \
    --render-scale "$RENDER_SCALE" \
    --bg "$BG" \
    ${SHARP_CKPT:+--ckpt "$SHARP_CKPT"} \
    "$@"
