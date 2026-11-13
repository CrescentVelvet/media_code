#!/usr/bin/env bash
# 31_batch_eval.sh — 批量：对 21 批量的每包跑 30 号定量评测（FF 多视角自一致 + Phase 0 登记）
#
# 用法:
#   bash 31_batch_eval.sh <批量输出根> [包名]
#   默认 <批量输出根> = ~/output/da3_portrait_batch_v2
#
# 每包输入（21 产物）：
#   <根>/<包>/images/*.jpg + <根>/<包>/gs_max/0000.ply + <根>/<包>/cameras.npz
# 每包产物：
#   <根>/<包>/eval/multiview_report.json + 汇总图（存在即跳过，断点续跑）
# 环境：querysplat（gsplat 1.5.3 + lpips，与 30_run_viewcount_bench.sh 一致）
set -o pipefail
ROOT="${1:-$HOME/output/da3_portrait_batch_v2}"
ONLY="${2:-}"
M=/mnt/c/code/media_code/da3_human
source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "🚀 [31] 批量 30 号评测"
echo "  📁 输入: $ROOT"

n_ok=0; n_skip=0; n_fail=0
for d in "$ROOT"/*/; do
    name="$(basename "$d")"
    if [ -n "$ONLY" ] && [ "$name" != "$ONLY" ]; then continue; fi
    [ -f "$d/gs_max/0000.ply" ] || { echo "⏭️ 跳过 $name（无 PLY）"; n_skip=$((n_skip+1)); continue; }
    if [ -f "$d/eval/multiview_report.json" ]; then echo "⏭️ 跳过 $name（已有报告）"; n_skip=$((n_skip+1)); continue; fi
    echo "━━ $name ━━"
    conda activate querysplat
    CAPTURE_DIR="$d" FF_DIR="$d" OUT_DIR="$d/eval" \
        python "$M/30_ff_multiview_eval.py" > "$d/eval.log" 2>&1
    if [ -f "$d/eval/multiview_report.json" ]; then
        n_ok=$((n_ok+1))
    else
        echo "❌ $name 评测失败（见 $d/eval.log）"
        n_fail=$((n_fail+1))
    fi
done
echo ""
echo "🎉 [31] 完成: ok=$n_ok skip=$n_skip fail=$n_fail → $ROOT"
