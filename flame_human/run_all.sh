#!/usr/bin/env bash
# run_all.sh — 一键跑通阶段一 → 阶段八。
# 每一步失败即中断；任一步的输入可用同名 env 覆盖（见 README 的 Config 表）。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [flame_human] 全流程 阶段一 → 阶段八"
echo "  📁 SOURCE_DIR : $SOURCE_DIR"
echo "  🎭 MASKS      : $PERSON_MASKS_DIR"
echo "  💾 RESULTS_DIR: $RESULTS_DIR"

STEPS=(02_detect_faces.sh 03_match_faces.sh 04_recon_faces.sh
       05_align_3dmm.sh 06_init_avatar_gs.sh 07_init_body_gs.sh 08_train.sh)

for s in "${STEPS[@]}"; do
    echo ""
    echo "════════════════════════════════════════"
    echo "▶  $s"
    echo "════════════════════════════════════════"
    bash "$SCRIPT_DIR/$s"
    if [ $? -ne 0 ]; then
        echo "❌ 中断于 $s"
        exit 1
    fi
done

echo ""
echo "🎉 全流程完成。WSL 上可再跑 09_move_output.sh 搬到 Windows 盘。"
