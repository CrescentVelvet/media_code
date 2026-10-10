#!/usr/bin/env bash
# run_all.sh — 一键全流程：建环境(00a) → 搬数据(01) → 微调+评估(02)。
# 用法：SCENE=dynamic_dance bash gavs/run_all.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/00a_setup_env.sh" || exit 1
bash "$SCRIPT_DIR/01_download_models.sh" || exit 1
SCENE="${SCENE:-dynamic_dance}" bash "$SCRIPT_DIR/02_run_finetune.sh" || exit 1

echo "🎉 [run_all] GaVS 全流程完成（场景: ${SCENE:-dynamic_dance}）"
