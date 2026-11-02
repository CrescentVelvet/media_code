#!/usr/bin/env bash
# run_all.sh —— 一键：建环境 -> 下权重 -> 推理
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-$HOME/repos/idol-sdk/demo/examples/fur-egg-chair.webp}"
export IMAGE

echo "🚀 [run_all] Idol 全流程"
bash "$SCRIPT_DIR/00_setup_env.sh" || { echo "❌ 00 失败" >&2; exit 1; }
bash "$SCRIPT_DIR/01_download_models.sh" || { echo "❌ 01 失败" >&2; exit 1; }
bash "$SCRIPT_DIR/02_run_inference.sh" || { echo "❌ 02 失败" >&2; exit 1; }
echo "🎉 [run_all] Done."
