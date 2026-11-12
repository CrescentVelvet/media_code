#!/usr/bin/env bash
# 08_nvs_eval.sh — held-out NVS 评测（WB-2）：source 建云，留出视角只评分。
#
# 用法：
#   RESULT=~/output/splatanything_results/_roundtrip/jiajia_hold_3v \
#   SOURCE_SCENE=~/output/splatanything_results/_scenes/jiajia_hold_3v \
#   ALL_SCENE=~/output/splatanything_results/_scenes/jiajia_hold \
#     bash 08_nvs_eval.sh
#
# Env:
#   RESULT=       02/06 的输出目录（含 gaussians.ply / cameras.json）——必填
#   SOURCE_SCENE= 推理时喂进去的场景目录（含 prior_camera.json）——必填
#   ALL_SCENE=    含全部视角的场景目录（留出 = 全部 − source）——必填
#   RESOLUTION="504 504"  必须与推理时一致
#   OUT=          输出目录，默认 $RESULT/nvs
#   MAX_HELD_OUT=0  最多评几个留出视角
#   EPS2D=0.01  CHUNK=4
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${RESOLUTION:=504 504}"
: "${MAX_HELD_OUT:=0}"
: "${EPS2D:=0.01}"
: "${CHUNK:=4}"

for v in RESULT SOURCE_SCENE ALL_SCENE; do
    if [ -z "${!v:-}" ]; then
        echo "❌ ERROR: 需要 $v（当前为空）" >&2
        exit 1
    fi
done

echo "🚀 [08] held-out NVS"
echo "  ☁️  source 云: $RESULT"
echo "  📸 source 场景: $SOURCE_SCENE"
echo "  🎯 全量场景: $ALL_SCENE"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_gsplat_env.sh"

ARGS=(--result "$RESULT" --source-scene "$SOURCE_SCENE" --all-scene "$ALL_SCENE"
      --resolution $RESOLUTION --eps2d "$EPS2D" --chunk "$CHUNK")
[ -n "${OUT:-}" ] && ARGS+=(--out "$OUT")
[ "$MAX_HELD_OUT" != "0" ] && ARGS+=(--max-held-out "$MAX_HELD_OUT")

python "$SCRIPT_DIR/08_nvs_eval.py" "${ARGS[@]}"
