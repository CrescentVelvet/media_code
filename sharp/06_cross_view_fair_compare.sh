#!/usr/bin/env bash
# 06 跨视角公平对照（identity / planeH / SHARP 三方同掩码）。
#
# 只读 05 的输出，不跑前馈、不占 GPU 显存（纯 numpy + PIL）。
#
#   RUN=~/output/sharp_results/S1a_crossview_13v_small22 \
#   SCENE=~/output/splatanything_results/_scenes/jiajia_hold_13v \
#     bash sharp/06_cross_view_fair_compare.sh
#
# 批量三组（推荐一次跑完）：
#   for tag in small22 small26 large125; do
#     RUN=~/output/sharp_results/S1a_crossview_13v_$tag \
#     SCENE=~/output/splatanything_results/_scenes/jiajia_hold_13v \
#       bash sharp/06_cross_view_fair_compare.sh
#   done
#
# 默认输出写回 RUN 目录（fair_compare.json），便于同一实验自包含。
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

RUN="${RUN:?必须给 RUN=<05 的输出目录>}"
SCENE="${SCENE:?必须给 SCENE=<场景目录，含 images/ 与 prior_camera.json>}"
OUT="${OUT:-$RUN}"

python "$SCRIPT_DIR/06_cross_view_fair_compare.py" \
    --run "$RUN" --scene "$SCENE" --out "$OUT" "$@"
