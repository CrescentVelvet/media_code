#!/usr/bin/env bash
# run_all.sh — 一键：建环境 → 下权重 → 跑示例推理 → 自接渲染评测。
#
# Env (all optional):
#   EXAMPLE=dtu_provided   # 官方示例名（hypersim_provided / dtu_provided / hypersim_da3_pose）
#   SKIP_SETUP=1           # 跳过 00（环境已就绪）
#   SKIP_DOWNLOAD=1        # 跳过 01（权重已就绪）
#   MODE=holdout           # 03 的评测口径：holdout / rerender
#   N_INPUT=8 N_TARGET=4   # holdout 划分规模
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${EXAMPLE:=dtu_provided}"
: "${MODE:=holdout}"

echo "🚀 [run_all] InfiniSplat v2 复现链路"
echo "  📁 官方仓: $INFINISPLAT_V2"
echo "  💾 权重:   $WEIGHTS_DIR"
echo "  📊 输出:   $RESULTS_DIR"

if [ "${SKIP_SETUP:-0}" != "1" ]; then
    bash "$SCRIPT_DIR/00_setup_env.sh" || exit 1
fi
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    bash "$SCRIPT_DIR/01_download_models.sh" || exit 1
fi

echo "🚀 推理示例: $EXAMPLE"
EXAMPLE="$EXAMPLE" bash "$SCRIPT_DIR/02_run_inference.sh" || exit 1

echo "🚀 渲染评测: MODE=$MODE"
if [ "$MODE" = "holdout" ]; then
    EXAMPLE="$EXAMPLE" MODE=holdout bash "$SCRIPT_DIR/03_eval_render.sh" || exit 1
else
    RESULT="$RESULTS_DIR/$EXAMPLE" \
    IMAGES_DIR="$INFINISPLAT_V2/examples/data/$EXAMPLE/images" \
    MODE=rerender bash "$SCRIPT_DIR/03_eval_render.sh" || exit 1
fi

echo "🎉 全流程完成。结果在 $RESULTS_DIR"
