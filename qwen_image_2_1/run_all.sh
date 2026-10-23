#!/usr/bin/env bash
# run_all.sh — 一键：校验权重 -> t2i 冒烟（默认参数出一张 2048x2048）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/01_check_models.sh" || exit 1
bash "$SCRIPT_DIR/02_run_inference.sh"
