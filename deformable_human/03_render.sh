#!/usr/bin/env bash
# 03_render.sh — 渲染训好的 Deformable-GS + 计算 PSNR/SSIM/LPIPS。
#
# 调官方 render.py + metrics.py（cfg_args 自动恢复 source_path 等训练参数）。
# 模式取值与官方 render.py argparse 对齐（镜像 deformable_gaussians/02 的封装）：
#   render   = 渲染全部测试图（默认，配 metrics）
#   time     = 时间插值（D-NeRF 用，出 video.mp4）
#   all      = 时间+视角
#   view     = 视角游走
#   pose     = 位姿插值
#   original = 真实数据时间+视角组合（NeRF-DS 用——看动态重影就选它）
#
# 对「去重影」课题，重点看 original/test 渲染里运动区域是否还有拖影，
# 而不只是 PSNR 数字。
#
# Env (all optional, defaults shown):
#   SCENE_NAME=human_seq
#   MODEL_PATH=            # 默认 $RESULTS_DIR/$SCENE_NAME/model（02 的输出）
#   ITERATION=-1           # 渲染哪个 checkpoint（-1=最新）
#   MODE=render            # render/time/all/view/pose/original
#   SKIP_TRAIN=1           # 1=--skip_train（省时间；0=连 train 视角一起渲）
#   SKIP_TEST=0            # 1=--skip_test
#   QUIET=0                # 1=--quiet
#   RUN_METRICS=1          # 1=渲染后跑 metrics.py（只评 test split）
#   EXTRA_RENDER_ARGS=     # 透传给 render.py
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE_NAME="${SCENE_NAME:-human_seq}"
MODEL_PATH="${MODEL_PATH:-$RESULTS_DIR/$SCENE_NAME/model}"
ITERATION="${ITERATION:--1}"
MODE="${MODE:-render}"
SKIP_TRAIN="${SKIP_TRAIN:-1}"
SKIP_TEST="${SKIP_TEST:-0}"
QUIET="${QUIET:-0}"
RUN_METRICS="${RUN_METRICS:-1}"
EXTRA_RENDER_ARGS="${EXTRA_RENDER_ARGS:-}"

echo "🚀 [03] 渲染 + 评测"
echo "  🏋️ 模型:  $MODEL_PATH"
echo "  🎬 mode:  $MODE   iteration: $ITERATION   metrics: $RUN_METRICS"
echo ""

if [ ! -f "$MODEL_PATH/cfg_args" ]; then
    echo "❌ ERROR: $MODEL_PATH/cfg_args 不存在（02 没跑完？）" >&2
    exit 1
fi

# render.py 的 --skip_train/--skip_test/--quiet 是 store_true，只在 =1 时传
RENDER_FLAGS=(-m "$MODEL_PATH" --iteration "$ITERATION" --mode "$MODE")
[ "$SKIP_TRAIN" = "1" ] && RENDER_FLAGS+=(--skip_train)
[ "$SKIP_TEST"  = "1" ] && RENDER_FLAGS+=(--skip_test)
[ "$QUIET"      = "1" ] && RENDER_FLAGS+=(--quiet)
# shellcheck disable=SC2086
[ -n "$EXTRA_RENDER_ARGS" ] && RENDER_FLAGS+=($EXTRA_RENDER_ARGS)

echo "--- render.py ${RENDER_FLAGS[*]} ---"
( cd "$DG_DIR" && python render.py "${RENDER_FLAGS[@]}" )
if [ $? -ne 0 ]; then
    echo "❌ FAILED: render.py" >&2
    exit 1
fi

if [ "$RUN_METRICS" = "1" ]; then
    echo "--- metrics -> $MODEL_PATH/test/results.json (PSNR/SSIM/LPIPS) ---"
    ( cd "$DG_DIR" && python metrics.py -m "$MODEL_PATH" )
    [ $? -ne 0 ] && { echo "❌ FAILED: metrics.py" >&2; exit 1; }
fi

echo ""
echo "🎉 [03] Done."
echo "  📁 测试视角渲染: $MODEL_PATH/test/ours_$ITERATION/renders/"
echo "  📊 指标:         $MODEL_PATH/test/results.json"
echo "  💡 目视检查重影比看 PSNR 更直接——优先翻 renders 里运动幅度大的帧"
