#!/usr/bin/env bash
# 29_run_d1_diag.sh — D1 诊断包：3v 前馈直出导出（四档）+ 双口径渲染统计
# 口径 FF = 前馈直出（gs_max 正式口径 / gs_official 历史口径）
# 口径 FT = DA3 点云 init + 30k 步 3DGS 精化（04_model_3dgs）
# 用法: bash da3_human/29_run_d1_diag.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

SCENE="${SCENE:-佳佳抱枕_3v}"
CAPTURE_DIR="${CAPTURE_DIR:-$HOME/output/da3_results/_captures/$SCENE}"
RESULT_DIR="${RESULT_DIR:-$HOME/output/da3_results/$SCENE}"
FF_DIR="$RESULT_DIR/d1_ff_export"
FT_DIR="$RESULT_DIR/04_model_3dgs"
DIAG_DIR="$RESULT_DIR/d1_diag"

if [ ! -d "$CAPTURE_DIR/images" ]; then
    echo "❌ ERROR: 采集目录不存在: $CAPTURE_DIR/images" >&2
    exit 1
fi
if [ ! -f "$FT_DIR/point_cloud/iteration_30000/point_cloud.ply" ]; then
    echo "❌ ERROR: 30k 精化 PLY 不存在: $FT_DIR/point_cloud/iteration_30000/point_cloud.ply" >&2
    exit 1
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "🚀 [D1-A] DA3 前馈直出导出（4 档，da3_human env）"
echo "  🖼️ 输入: $CAPTURE_DIR/images"
echo "  💾 输出: $FF_DIR"
conda activate da3_human
IMAGES_DIR="$CAPTURE_DIR/images" \
OUT_DIR="$FF_DIR" \
PROCESS_RES=504 N_VIEWS=0 REF_STRATEGY=first EXPORT=all FAST_EXPORT=1 \
python da3_human/08_bench_official_gs.py
if [ $? -ne 0 ]; then echo "❌ FAILED at step A" >&2; exit 1; fi

echo "🚀 [D1-B] 双口径渲染 + 统计（querysplat env, gsplat 1.5.3）"
echo "  💾 输出: $DIAG_DIR"
conda activate querysplat
CAPTURE_DIR="$CAPTURE_DIR" FF_DIR="$FF_DIR" FT_DIR="$FT_DIR" OUT_DIR="$DIAG_DIR" \
python da3_human/29_d1_diag_3view.py
if [ $? -ne 0 ]; then echo "❌ FAILED at step B" >&2; exit 1; fi

echo "🎉 D1 诊断完成 → $DIAG_DIR"
