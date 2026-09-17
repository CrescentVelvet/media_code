#!/usr/bin/env bash
# 02_run_inference.sh — SHARP 前馈推理：单张/多张图 → 3D Gaussian (.ply) [+ 环绕视频 .mp4]
#
# 官方 CLI：sharp predict -i <图或目录> -o <输出目录> -c <权重> [--render]
#   - 输入目录会递归匹配所有支持的图像格式
#   - 每张图输出一个 <stem>.ply（OpenCV 坐标系，场景中心 ≈ (0,0,+z)）
#   - --render 额外输出 <stem>.mp4（环绕轨迹），**仅 CUDA**，依赖 gsplat
#
# ⚠️ 首次带 --render 运行会 JIT 编译 gsplat 的 CUDA 光栅化核（需要 nvcc+g++，
#    本机约几分钟）。编译产物缓存在 ~/.cache/torch_extensions/，只编译一次。
#
# 用法：
#   GPU=0 INPUT=~/my_images bash sharp/02_run_inference.sh
#   GPU=0 INPUT=/mnt/d/dataset/sample/a.jpg RENDER=1 bash sharp/02_run_inference.sh
#   GPU=0 INPUT=~/my_images RESULTS_DIR=~/output/sharp_test bash sharp/02_run_inference.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

INPUT="${INPUT:-}"
RENDER="${RENDER:-0}"
DEVICE="${DEVICE:-default}"
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/02_gaussians}"
VERBOSE="${VERBOSE:-0}"

# ── 前置检查 ───────────────────────────────────────────────────────────────
if [ -z "$INPUT" ]; then
    echo "❌ ERROR: INPUT 未设置（图文件或图片目录）" >&2
    echo "   例: GPU=0 INPUT=/mnt/d/dataset/sample bash sharp/02_run_inference.sh" >&2
    exit 1
fi
if [ ! -e "$INPUT" ]; then
    echo "❌ ERROR: INPUT 不存在: $INPUT" >&2
    exit 1
fi
if [ ! -f "$SHARP_CKPT" ]; then
    echo "❌ ERROR: 权重不存在: $SHARP_CKPT" >&2
    echo "   先跑: bash sharp/01_download_models.sh" >&2
    exit 1
fi
if ! command -v sharp >/dev/null 2>&1; then
    echo "❌ ERROR: 找不到 sharp CLI（conda env '$CONDA_ENV' 未装好？）" >&2
    echo "   先跑: bash sharp/00a_setup_env.sh（或 pip install -e $SHARP_DIR）" >&2
    exit 1
fi

echo "🚀 [02] SHARP 推理"
echo "  🖼️ 输入:   $INPUT"
echo "  💾 输出:   $OUT_DIR"
echo "  🏋️ 权重:   $SHARP_CKPT"
echo "  🎬 渲染视频: $([ "$RENDER" = "1" ] && echo 'on (--render)' || echo 'off')"
echo "  🎮 GPU:    ${CUDA_VISIBLE_DEVICES:-all}"
echo ""

# ── 组装参数 ───────────────────────────────────────────────────────────────
ARGS=(-i "$INPUT" -o "$OUT_DIR" -c "$SHARP_CKPT")
[ "$RENDER" = "1" ] && ARGS+=(--render)
[ "$DEVICE" != "default" ] && ARGS+=(--device "$DEVICE")
[ "$VERBOSE" = "1" ] && ARGS+=(-v)

# ── 执行 ───────────────────────────────────────────────────────────────────
_t0=$(date +%s)
echo "🤖 sharp predict ${ARGS[*]}"
echo ""
sharp predict "${ARGS[@]}"
_rc=$?
_t1=$(date +%s)

if [ $_rc -ne 0 ]; then
    echo "" >&2
    echo "❌ FAILED (exit $_rc)" >&2
    echo "  常见原因：" >&2
    echo "   - --render 报 gsplat/CUDA 编译错 → 检查 nvcc: nvcc --version；gcc: which g++" >&2
    echo "   - 显存不足 → 换单张图试；1536x1536 输入在 3090(24G) 上充裕" >&2
    exit 1
fi

echo ""
echo "⏱️  耗时: $((_t1 - _t0))s"
echo "🎉 [02] Done."
echo "  📁 输出目录: $OUT_DIR"
echo "     *.ply  — 3D Gaussian（拖进 https://playcanvas.com/supersplat/editor 即可查看）"
[ "$RENDER" = "1" ] && echo "     *.mp4  — 环绕轨迹渲染"
