#!/usr/bin/env bash
# run_all.sh — 一键全流程：建环境 → 下权重 → 单图推理。
#
# 用法：
#   GPU=0 INPUT=/path/to/image.png bash taoflowforge/run_all.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/3] 环境 ==="
bash "$SCRIPT_DIR/00a_setup_env.sh" || exit 1

echo ""
echo "=== [2/3] 权重 ==="
bash "$SCRIPT_DIR/01_download_models.sh" || exit 1

echo ""
echo "=== [3/3] 推理 ==="
if [ -z "${INPUT:-}" ]; then
    echo "⚠️ INPUT 未设置，跳过推理。用法：" >&2
    echo "  GPU=0 INPUT=/path/to/image.png bash taoflowforge/run_all.sh" >&2
    exit 0
fi
bash "$SCRIPT_DIR/02_run_inference.sh" || exit 1

echo ""
echo "🎉 run_all Done."
