#!/usr/bin/env bash
# 06_run_roundtrip.sh — 包 06_ply_roundtrip.py：跑一次推理，验"PLY == 内存高斯"。
#
# 用法：
#   INPUT=$SPLATANYTHING_RESULTS_DIR/_scenes/jiajia_hold_3v bash 06_run_roundtrip.sh
#   INPUT=/path/to/scene RESOLUTION="504 504" bash 06_run_roundtrip.sh
#
# Env:
#   INPUT=        场景目录（含 images/，pose 模式需 prior_camera.json）——必填
#   MODE=pose     rgb | pose
#   RESOLUTION="504 504"
#   OUT=          输出根（默认 $SPLATANYTHING_RESULTS_DIR/_roundtrip）
#   EPS2D=0.01    渲染屏幕膨胀（与 03 一致）
#   CHUNK=4       单次渲染视角数
#   PROFILE=1     采样 GPU 峰值显存
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${MODE:=pose}"
: "${RESOLUTION:=504 504}"
: "${EPS2D:=0.01}"
: "${CHUNK:=4}"
: "${PROFILE:=1}"

if [ -z "${INPUT:-}" ] || [ ! -d "$INPUT/images" ]; then
    echo "❌ ERROR: 需要 INPUT 指向含 images/ 的场景目录（当前: '${INPUT:-}'）" >&2
    exit 1
fi
if ! python -c "import torch" >/dev/null 2>&1; then
    echo "❌ ERROR: conda env '$CONDA_ENV' 不可用（先跑 00_setup_env.sh）" >&2
    exit 1
fi

: "${OUT:=$SPLATANYTHING_RESULTS_DIR/_roundtrip}"
mkdir -p "$OUT"

echo "🚀 [06] PLY 往返验证"
echo "  🖼️  场景: $INPUT"
echo "  🎛️  模式: $MODE  分辨率: $RESOLUTION px  eps2d: $EPS2D"

# gsplat 安装 + 编译前置（与 03 共用）
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_gsplat_env.sh"

PEAK_LOG=""
if [ "$PROFILE" = "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
    PEAK_LOG="$OUT/.gpu_peak_$$.log"
    ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) >"$PEAK_LOG" 2>/dev/null &
    SAMPLER_PID=$!
fi

START=$(date +%s)
python "$SCRIPT_DIR/06_ply_roundtrip.py" \
    --input "$INPUT" --mode "$MODE" --resolution $RESOLUTION \
    --out "$OUT" --eps2d "$EPS2D" --chunk "$CHUNK"
RC=$?
END=$(date +%s)

if [ -n "$PEAK_LOG" ]; then
    kill "$SAMPLER_PID" 2>/dev/null
    wait "$SAMPLER_PID" 2>/dev/null
    if [ -s "$PEAK_LOG" ]; then
        awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{printf "  🎮 显存峰值: %.2f GB\n", m/1024}' "$PEAK_LOG"
    fi
    rm -f "$PEAK_LOG"
fi

echo "⏱️  总耗时 $((END - START)) s"
if [ $RC -eq 0 ]; then
    echo "🎉 往返验证通过"
elif [ $RC -eq 2 ]; then
    echo "❌ 往返验证未达 1e-4（指标见 ply_roundtrip.json，先查属性 MAE 定位是哪一段）"
else
    echo "❌ 脚本异常 (exit $RC)" >&2
fi
exit $RC
