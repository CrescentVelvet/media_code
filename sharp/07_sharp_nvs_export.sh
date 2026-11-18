#!/usr/bin/env bash
# 07_sharp_nvs_export.sh — SHARP 前馈结果导出成 SA `08_nvs_eval.py` 可直接消费的格式。
#
# 目的（v4 §4 第 3 条「SA 相同 target 配对」）：让 SHARP 与 SA 走**同一个评测器**，
# 在同一批留出视角上打同一把尺子，而不是各写一套再比数字。
#
# ⚠️ 必须经本 wrapper 跑：`_env.sh` 定 CUDA_HOME 与 conda env，否则 predictor 加载会失败。
#
# 用法：
#   SCENE=~/output/splatanything_results/_scenes/jiajia_hold_3v \
#   VIEWS=000,004,022 ANCHOR=000 TAG=SH1b_nvs_3v \
#     bash sharp/07_sharp_nvs_export.sh
#
#   # 13v（需先对 13 张逐一前馈，约 13× 前馈时间）
#   SCENE=~/output/splatanything_results/_scenes/jiajia_hold_13v TAG=SH1b_nvs_13v \
#     bash sharp/07_sharp_nvs_export.sh
#
# 之后立刻用**同一个**评测器打分（RESOLUTION 必须与 SA 侧一致，默认 504 504）：
#   RESULT=$OUT \
#   SOURCE_SCENE=$SCENE \
#   ALL_SCENE=~/output/splatanything_results/_scenes/jiajia_hold \
#     bash splatanything/08_nvs_eval.sh
#
# ⚠️ 不用 set -u：proxy.env 有 `export PYTHONPATH="...:$PYTHONPATH"`，
#    PYTHONPATH 未定义时会直接崩（03/04/05 也因此只开 -eo pipefail）。
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE="${SCENE:-$HOME/output/splatanything_results/_scenes/jiajia_hold_3v}"
TAG="${TAG:-$(basename "$SCENE")}"
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/SH1b_nvs_$TAG}"
DEVICE="${DEVICE:-cuda}"

EXTRA=()
[ -n "${VIEWS:-}" ] && EXTRA+=(--views "$VIEWS")
[ -n "${ANCHOR:-}" ] && EXTRA+=(--anchor "$ANCHOR")
[ -n "${CKPT:-}" ] && EXTRA+=(--ckpt "$CKPT")

echo "🚀 [07] SHARP -> NVS 导出"
echo "  📁 场景: $SCENE"
echo "  💾 输出: $OUT_DIR"
echo "  🎯 视角: ${VIEWS:-全部}  锚定: ${ANCHOR:-首个}"
echo "  🎮 DEVICE: $DEVICE"

python "$SCRIPT_DIR/07_sharp_nvs_export.py" \
    --scene "$SCENE" \
    --out "$OUT_DIR" \
    --device "$DEVICE" \
    "${EXTRA[@]}"

echo ""
echo "🎉 [07] Done. 下一步（同一个评测器）："
echo "  RESULT=$OUT_DIR \\"
echo "  SOURCE_SCENE=$SCENE \\"
echo "  ALL_SCENE=$HOME/output/splatanything_results/_scenes/jiajia_hold \\"
echo "    bash splatanything/08_nvs_eval.sh"
