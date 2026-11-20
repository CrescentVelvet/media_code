#!/usr/bin/env bash
# 02a_smoke_jiajia.sh — 佳佳抱枕三视角纯前馈冒烟，不启动完整 TokenBA
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCENE_NAME="${SCENE_NAME:-jiajia_smoke_3v}"
export IMAGES_DIR="${IMAGES_DIR:-$HOME/output/infinisplat_v2_results/_captures/佳佳抱枕_3v/images}"
export MAX_VIEWS=3
export BA=0
export SAVE_FEEDFORWARD=1
export RENDER_VIDEO=0
bash "$SCRIPT_DIR/02_run_inference.sh"
