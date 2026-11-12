#!/usr/bin/env bash
# 03_eval_render.sh — 用 gsplat 渲染导出的高斯，出图 + PSNR/SSIM（自洽口径）。
#
# 官方发布包不带光栅化器（rasterization.py 注释 "no rasterization or training"），
# 所以渲染必须自接。gsplat 首次运行会 JIT 编译 CUDA kernel，需要三样东西：
#   1. nvcc        —— conda install -n $CONDA_ENV -c nvidia cuda-nvcc=12.6
#   2. gcc/g++     —— 本机 WSL 没装，默认借用 infinisplat_v2 env 的 conda gcc（GCC_DIR 可改）
#   3. glm 头文件  —— gsplat 1.5.2 不内置（1.5.3 才带），git clone 到 $GLM_DIR
# 另外必须设 TORCH_CUDA_ARCH_LIST，否则 nvcc 编所有架构（编译时间 ×10）。
#
# 用法：
#   RESULT=$HOME/output/splatanything_results/pose/object bash 03_eval_render.sh
#
# Env:
#   RESULT=            02 的输出目录（含 gaussians.ply / cameras.json / rgb/）——必填
#   OUT=               评测输出目录，默认 $RESULT/eval
#   EPS2D=0.01         屏幕空间膨胀（论文值 1e-2）
#   MAX_VIEWS=0        最多评几个视角，0 = 全部
#   GLM_DIR=           glm 头文件根（默认 ~/repos/glm_src，需含 glm/ 子目录）
#   GCC_DIR=           借用哪个 env 的 gcc（默认 infinisplat_v2 的 bin）
#   CUDA_ARCH=         不设则按 GPU 推断（3090 → 8.6）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${EPS2D:=0.01}"
: "${MAX_VIEWS:=0}"
: "${GLM_DIR:=$HOME/repos/glm_src}"
: "${GCC_DIR:=$HOME/miniconda3/envs/infinisplat_v2/bin}"

if [ -z "${RESULT:-}" ] || [ ! -f "$RESULT/gaussians.ply" ]; then
    echo "❌ ERROR: 需要 RESULT 指向含 gaussians.ply 的 02 输出目录（当前: '${RESULT:-}'）" >&2
    exit 1
fi

# ---- gsplat 安装 + 编译前置（与 06 共用）----
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_gsplat_env.sh"

ARGS=(--result "$RESULT" --eps2d "$EPS2D" --max-views "$MAX_VIEWS")
[ -n "${OUT:-}" ] && ARGS+=(--out "$OUT")

python "$SCRIPT_DIR/03_render_eval.py" "${ARGS[@]}" || exit 1
