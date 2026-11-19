#!/usr/bin/env bash
# structsplat/02_run_inference.sh — 任意图片目录 → 前馈高斯重建 + 渲染 + 导出
#
# 为什么需要这个脚本：官方仓只有 train.py / evaluation.py，两者都硬绑 DL3DV
# annotation 格式（{path, src[], tar[]}），没有「给一个图片目录就出结果」的入口。
# infer.py 补这一环，本脚本负责环境变量与常用默认值。
#
# 用法：
#   # 最简（pose-free，默认黑底，导出 ply + splat）
#   bash structsplat/02_run_inference.sh /path/to/images
#
#   # 指定输出目录 / 渲染分辨率 / 采样张数
#   bash structsplat/02_run_inference.sh /path/to/images my_out 768 --max-images 24
#
#   # 用 GT 相机（验证几何，把相机估计误差从渲染误差里分离）
#   bash structsplat/02_run_inference.sh /path/to/images my_out 512 gt \
#        --transforms /path/to/transforms.json
#
#   # 白底
#   bash structsplat/02_run_inference.sh /path/to/images my_out 512 pose-free --white-bg
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

IMAGES="${1:-}"
OUT_NAME="${2:-run_$(date +%m%d_%H%M%S)}"
RENDER_SIZE="${3:-512}"
CAMERA="${4:-pose-free}"
# ⚠️ 不能用 `shift 4 2>/dev/null || true`：参数不足 4 个时 bash 的 shift 会整体失败，
#    位置参数原地不动，于是 EXTRA_ARGS 会把图片路径也带上，argparse 立刻报
#    "unrecognized arguments: <路径>"。按实际个数 shift 才安全。
_N=$#
shift $(( _N < 4 ? _N : 4 ))
EXTRA_ARGS=("$@")

if [ -z "$IMAGES" ]; then
    echo "用法: bash structsplat/02_run_inference.sh <图片目录|glob> [输出名] [渲染尺寸] [pose-free|gt] [额外 infer.py 参数...]" >&2
    exit 1
fi

# ---------- gsplat JIT 编译的前置条件（本机实测必需）----------
# conda 装的 CUDA 把头文件放在 targets/x86_64-linux/include，
# torch 的 cpp_extension 只会加 $CUDA_HOME/include → 不加 CPATH 会报
#   fatal error: cuda_runtime_api.h: No such file or directory
# CUDA_HOME 不设则 gsplat 直接判定「没有 CUDA toolkit」而静默禁用（_C=None）。
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"   # RTX 3090 = sm_86
# deepspeed 顶层 import 时若无 CUDA_HOME 会抛 MissingCUDAException
export DS_BUILD_OPS="${DS_BUILD_OPS:-0}"

OUT_DIR="$RESULTS_DIR/$OUT_NAME"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo " StructSplat 推理"
echo "   输入     : $IMAGES"
echo "   输出     : $OUT_DIR"
echo "   渲染尺寸 : $RENDER_SIZE"
echo "   相机     : $CAMERA"
echo "   env      : $CONDA_ENV  (CUDA_HOME=$CUDA_HOME)"
echo "============================================================"

cd "$STRUCTSPLAT_REPO" || { echo "❌ 仓不存在: $STRUCTSPLAT_REPO" >&2; exit 1; }

# 默认导出两种格式，便于用 SuperSplat / antimatter15 viewer 检查。
# ⚠️ 逐个参数精确比对，不能用 `[[ " ${EXTRA_ARGS[*]} " == *" --export"* ]]`：
#    那样会把 --export-ply / --export-splat 也当成「用户已指定导出」，
#    结果用户只说 --export-ply 时默认的 --export-splat 被误丢掉。
DEFAULT_EXPORT=(--export-ply --export-splat)
for _a in "${EXTRA_ARGS[@]}"; do
    case "$_a" in
        --export|--export-ply|--export-splat|--no-export)
            DEFAULT_EXPORT=()
            break
            ;;
    esac
done

python "$SCRIPT_DIR/infer.py" \
    --images "$IMAGES" \
    --out "$OUT_DIR" \
    --config "$STRUCTSPLAT_REPO/config/dl3dv.yaml" \
    --render-size "$RENDER_SIZE" \
    --camera "$CAMERA" \
    "${DEFAULT_EXPORT[@]}" \
    "${EXTRA_ARGS[@]}"

RC=$?
if [ $RC -ne 0 ]; then
    echo "❌ 推理失败 (exit=$RC)" >&2
    exit $RC
fi

echo ""
echo "🎉 [02] 完成，产物在: $OUT_DIR"
echo "   renders/      渲染图（逐视角）"
echo "   depths/       深度图"
echo "   compare.png   原图 vs 渲染 vs 深度 对比"
echo "   gaussians.ply / gaussians.splat"
echo "   meta.json     元信息"
