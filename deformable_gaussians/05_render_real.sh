#!/usr/bin/env bash
# 05_render_real.sh — 渲染真实人体训练结果（薄封装 02_run_inference.sh）。
#
# 专门给 04_train_real.sh 训出来的真实数据场景用——设默认 MODEL_PATH 接 04 输出，
# 默认 MODE=original（NeRF-DS 真实数据专用：时间+视角插值出 video.mp4，看人物微动
# 的动态重建效果）。02 本身已支持所有这些 env vars，05 只是把默认值设成适合真实
# 数据路径的（D-NeRF 复现走 02 直接调；真实人体走 05）。
#
# 与直接调 02 的区别：
#   - 05 默认 SCENE_NAME=real_scene（接 04 默认）
#   - 05 默认 MODEL_PATH=$DG_DIR/output/real_<scene>（接 04 输出，避免覆盖 D-NeRF）
#   - 05 默认 MODE=original（02 默认 render——前者出 video.mp4 看动态，后者只出
#     test 视角 PNG 适合算指标）
#
# 用法示例：
#   # 渲染 04 训出来的 real_<scene>（默认 mode=original 出 video.mp4）
#   GPU=0 SCENE_NAME=alice bash deformable_gaussians/05_render_real.sh
#   # 只渲 test 视角算指标（不出 video）
#   GPU=0 SCENE_NAME=alice MODE=render bash deformable_gaussians/05_render_real.sh
#   # 渲指定 checkpoint（--iteration N）
#   GPU=0 SCENE_NAME=alice ITERATION=10000 bash deformable_gaussians/05_render_real.sh
#
# Env (all optional, defaults shown):
#   SCENE_NAME=real_scene  # 场景名（接 04 默认；影响默认 MODEL_PATH）
#   MODEL_PATH=            # 训练输出目录（默认 $DG_DIR/output/real_<scene>）
#   MODE=original          # render|time|all|view|pose|original (02 同名 var)
#   ITERATION=-1           # -1=最新 checkpoint
#   SKIP_TRAIN=1           # 1=跳过渲染 train split（02 同名 var）
#   SKIP_TEST=0            # 0=渲 test split（02 同名 var）
#   RUN_METRICS=1          # 1=跑 metrics.py 算 PSNR/SSIM/LPIPS（02 同名 var）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# ── Paths ─────────────────────────────────────────────────────────────────
DG_DIR="${DG_DIR:-$REPO_DIR/../Deformable-3D-Gaussians}"
SCENE_NAME="${SCENE_NAME:-real_scene}"
# 接 04 输出（real_ 前缀避免覆盖 D-NeRF 复现的 $DG_DIR/output/<scene>）
MODEL_PATH="${MODEL_PATH:-$DG_DIR/output/real_$SCENE_NAME}"

# ── Params（默认值针对真实数据；02 内同名 var 也都还能被外部覆盖）──────────
# MODE=original: NeRF-DS 真实数据专用，时间+视角插值出 video.mp4 看人物微动效果。
# MODE=render: 只渲所有 test 视角 PNG（不出 video，适合仅算指标）。
# MODE=time: D-NeRF 时间插值（合成数据用，真实数据用 original）。
MODE="${MODE:-original}"
ITERATION="${ITERATION:--1}"
SKIP_TRAIN="${SKIP_TRAIN:-1}"
SKIP_TEST="${SKIP_TEST:-0}"
RUN_METRICS="${RUN_METRICS:-1}"

echo "🚀 [05] 渲染真实人体训练结果（封装 02_run_inference.sh）"
echo "  🤖 代码:       $DG_DIR"
echo "  💾 模型:       $MODEL_PATH"
echo "  📐 iteration:  $ITERATION (-1 = 最新)"
echo "  🎬 mode:       $MODE  (original=NeRF-DS 时间+视角 video; render=仅 test PNG)"
echo "  skip_train=$SKIP_TRAIN  skip_test=$SKIP_TEST  run_metrics=$RUN_METRICS"
echo ""

# ── 0. Sanity checks ──────────────────────────────────────────────────────
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ ERROR: 训练输出目录不存在: $MODEL_PATH" >&2
    echo "       先跑 04 训练:" >&2
    echo "         GPU=0 SCENE_NAME=$SCENE_NAME bash $SCRIPT_DIR/04_train_real.sh" >&2
    echo "       或设 MODEL_PATH 指向已有训练输出" >&2
    exit 1
fi
if [ ! -f "$MODEL_PATH/cfg_args" ]; then
    echo "⚠️  $MODEL_PATH/cfg_args 不存在 — render.py 无法自动恢复 is_blender/source_path。" >&2
    echo "    手放 checkpoint 时补参数: EXTRA_RENDER_ARGS=\"--source_path <data>\" bash $0" >&2
    echo "    （04 训出来的话 cfg_args 应该有；这是提醒不是错误，继续跑）"
fi

# ── 1. 透传给 02_run_inference.sh ─────────────────────────────────────────
# 02 已经支持所有这些 env vars，05 只是设默认值适合真实数据 + 加 sanity check。
export MODEL_PATH SCENE_NAME MODE ITERATION SKIP_TRAIN SKIP_TEST RUN_METRICS

echo "  → 调 02_run_inference.sh ..."
bash "$SCRIPT_DIR/02_run_inference.sh"
rc=$?

if [ $rc -ne 0 ]; then
    echo "❌ [05] FAILED (02 退出码 $rc)" >&2
    exit $rc
fi

echo ""
echo "🎉 [05] Done. 渲染 + 评测完成。"
echo "  🖼️  渲染图:    $MODEL_PATH/test/ours_$ITERATION/renders/*.png"
echo "  📏 深度图:     $MODEL_PATH/test/ours_$ITERATION/depth/*.png"
if [ "$MODE" = "original" ] || [ "$MODE" = "time" ] || [ "$MODE" = "all" ] || \
   [ "$MODE" = "view" ] || [ "$MODE" = "pose" ]; then
    echo "  🎬 动态视频:  $MODEL_PATH/test/interpolate_${MODE}_$ITERATION/renders/video.mp4"
fi
if [ "$RUN_METRICS" = "1" ] && [ "$SKIP_TEST" != "1" ]; then
    echo "  📊 指标:      $MODEL_PATH/test/results.json (PSNR/SSIM/LPIPS)"
fi
echo ""
echo "  💡 提示: 换渲染模式看不同效果"
echo "    MODE=render   只渲 test 视角 PNG（算指标必需）"
echo "    MODE=original NeRF-DS 时间+视角插值出 video.mp4（看微动效果）"
echo "    MODE=time     同视角不同时刻（看时间插值）"
