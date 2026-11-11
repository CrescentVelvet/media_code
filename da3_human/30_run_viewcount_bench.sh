#!/usr/bin/env bash
# 30_run_viewcount_bench.sh — 数据集就绪前的纯推理测试批：
#   A. 视角数分桶（同一 32v 采集包，N_VIEWS=3/8/13/18，REF_STRATEGY=first）
#   B. 多包基线摸底（其余真实包全 32 视角，登记 Phase 0 视角覆盖）
# 每条 = 08 前馈导出（gs_max 正式口径）+ 30 多视角评测（PSNR/SSIM/LPIPS/锐度/空洞）。
# 用法: bash da3_human/30_run_viewcount_bench.sh [A|B|all]
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
MODE="${1:-all}"

CAPBASE="$HOME/output/da3_results/_captures"
OUTBASE="$HOME/output/da3_results/_bench_viewcount"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

run_one() {  # run_one <场景名> <N_VIEWS>
    local scene="$1" nv="$2"
    local capdir="$CAPBASE/$scene"
    local tag="${scene}_${nv}v"
    local ffdir="$OUTBASE/$tag/ff"
    local evdir="$OUTBASE/$tag/eval"
    if [ ! -d "$capdir/images" ]; then echo "❌ 跳过 $scene（无采集目录）"; return 0; fi
    if [ -f "$evdir/multiview_report.json" ]; then echo "⏭️ 跳过 $tag（已有报告，断点续跑）"; return 0; fi
    echo ""
    echo "==================== $tag ===================="
    conda activate da3_human
    IMAGES_DIR="$capdir/images" OUT_DIR="$ffdir" \
    PROCESS_RES=504 N_VIEWS=$nv REF_STRATEGY=first EXPORT=max FAST_EXPORT=1 \
    python da3_human/08_bench_official_gs.py || { echo "❌ FAILED 08 $tag" >&2; return 1; }
    conda activate querysplat
    CAPTURE_DIR="$capdir" FF_DIR="$ffdir" OUT_DIR="$evdir" \
    python da3_human/30_ff_multiview_eval.py || { echo "❌ FAILED eval $tag" >&2; return 1; }
}

if [ "$MODE" = "A" ] || [ "$MODE" = "all" ]; then
    for nv in 3 8 13 18; do run_one 佳佳抱枕 $nv; done
fi
if [ "$MODE" = "B" ] || [ "$MODE" = "all" ]; then
    for scene in 三人聚餐 佳佳蹲 女生写字 男生二郎腿 男生室内 男生手机 佳佳抱枕_gs; do
        run_one "$scene" 0
    done
fi
echo ""
echo "🎉 全部完成 → $OUTBASE"
