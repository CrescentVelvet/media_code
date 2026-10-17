#!/usr/bin/env bash
# run_all.sh — 一键：图 → json → 3DGS（假设环境与权重都就绪）。
#
# 首次跑前请按顺序做：
#   1) bash flashworld/00a_setup_env.sh        # WSL 建 env + clone + 编译
#   2) bash flashworld/01_download_models.sh   # 下 54 GB 权重（或迅雷 + 01b）
#
# 用法：
#   GPU=0 IMAGE=/mnt/d/dataset/scene.jpg bash flashworld/run_all.sh
#   GPU=0 IMAGE=/mnt/d/dataset/batch VIDEO=1 bash flashworld/run_all.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

if [ -z "${IMAGE:-}" ]; then
    echo "❌ ERROR: IMAGE 未设置" >&2
    echo "   例: GPU=0 IMAGE=/mnt/d/dataset/scene.jpg bash flashworld/run_all.sh" >&2
    exit 1
fi

echo "════════ 1/2 生成输入 json ════════"
bash "$SCRIPT_DIR/02_make_input.sh" || exit 1

echo ""
echo "════════ 2/2 推理 ════════"
INPUT_DIR="${INPUT_DIR:-$RESULTS_DIR/02_input_json}" bash "$SCRIPT_DIR/02_run_inference.sh" || exit 1

echo ""
echo "🎉 全流程完成 → $RESULTS_DIR/03_gaussians"
