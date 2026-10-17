#!/usr/bin/env bash
# 02_make_input.sh — 单图/图目录 → FlashWorld 输入 json（相机轨迹见 02_make_input.py 头注释）。
#
# 用法：
#   # 官方 demo 那种小幅环绕（±15°），用来对照官方结果
#   IMAGE=/mnt/d/dataset/scene.jpg TRAJ=official_orbit bash flashworld/02_make_input.sh
#   # 本仓默认：±20°（YAW_SPAN=40）环绕，抽 3D 资产覆盖更宽
#   IMAGE=/mnt/d/dataset/scene.jpg bash flashworld/02_make_input.sh
#   # 批量：目录里每张图一个 json
#   IMAGE=/mnt/d/dataset/batch bash flashworld/02_make_input.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

if [ -z "${IMAGE:-}" ]; then
    echo "❌ ERROR: IMAGE 未设置（单张图片或图片目录）" >&2
    echo "   例: IMAGE=/mnt/d/dataset/scene.jpg bash flashworld/02_make_input.sh" >&2
    exit 1
fi

export OUTPUT="${OUTPUT:-$RESULTS_DIR/02_input_json}"
mkdir -p "$OUTPUT"

python "$SCRIPT_DIR/02_make_input.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
