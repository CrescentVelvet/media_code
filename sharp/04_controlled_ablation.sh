#!/usr/bin/env bash
# 04_controlled_ablation.sh — SHARP 两层深度受控消融（S0.5b）。
#
# 与 03 的关系：03 只做「整层抽掉」的黑箱消融并给出 30.6 → 23.9 的事实；
# 04 在同一份前馈输出上做**受控**干预（复制/缩放/opacity/换色），
# 把收益来源定性到「覆盖密度 / 独立信息 / 深度排序」三者之一。
#
# 用法：
#   INPUT=/path/to/img.heic bash sharp/04_controlled_ablation.sh
#   INPUT=... RENDER_SCALE=1.0 bash sharp/04_controlled_ablation.sh   # 全分辨率（慢 4x）
#
# 环境：必须经本脚本（内部 source _env.sh）——直接调 env python 会让 gsplat
# 找不到 CUDA toolkit 而静默禁用，报 'NoneType' has no attribute 'CameraModelType'。
# 不用 set -u：proxy.env 里有 `export PYTHONPATH="...:$PYTHONPATH"` 之类的写法，
# 在 PYTHONPATH 未定义时会直接崩（03 脚本也因此只开 pipefail）。
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

INPUT="${INPUT:-}"
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/04_controlled_ablation}"
RENDER_SCALE="${RENDER_SCALE:-0.5}"
JITTER_PX="${JITTER_PX:-0.5}"

if [ -z "$INPUT" ]; then
    echo "❌ 必须指定 INPUT=<输入图路径>" >&2
    exit 1
fi
if [ ! -f "$INPUT" ]; then
    echo "❌ 输入不存在: $INPUT" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "=========================================================="
echo "SHARP 受控消融（S0.5b）"
echo "  输入      : $INPUT"
echo "  输出      : $OUT_DIR"
echo "  渲染比例  : $RENDER_SCALE"
echo "  权重      : ${SHARP_CKPT:-<官方 URL>}"
echo "=========================================================="

python "$SCRIPT_DIR/04_controlled_ablation.py" \
    --input "$INPUT" \
    --out "$OUT_DIR" \
    --render-scale "$RENDER_SCALE" \
    --jitter-px "$JITTER_PX" \
    ${SHARP_CKPT:+--ckpt "$SHARP_CKPT"} \
    "$@"

echo
echo "✅ 完成: $OUT_DIR/controlled_ablation.json"
