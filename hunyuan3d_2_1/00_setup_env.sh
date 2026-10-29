#!/usr/bin/env bash
# 00_setup_env.sh — verify (or create) the conda env, install Hunyuan3D-2.1
# deps and compile its two CUDA extensions.
#
# Official upstream tests with Python 3.10 + torch 2.5.1+cu124. Reuses an
# existing env by default (torch must already be CUDA-enabled); set
# CREATE_ENV=1 to create the env fresh (python 3.10 + torch 2.5.1+cu124).
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CREATE_ENV="${CREATE_ENV:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

if [ "$CREATE_ENV" = "1" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/_env.sh" no-activate
    if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        echo "📦 [00] creating conda env '$CONDA_ENV' (python 3.10)"
        conda create -n "$CONDA_ENV" python=3.10 -y || exit 1
    fi
    conda activate "$CONDA_ENV"
    echo "📦 [00] installing torch 2.5.1 + cu124 (upstream tested combo)"
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124 || exit 1
else
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/_env.sh"
    echo "🔍 [00] verify torch in conda env '$CONDA_ENV'"
    python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("❌ torch.cuda not available — install a CUDA-enabled torch or check GPU visibility.")
PY
fi

if [ "$INSTALL_DEPS" != "1" ]; then
    echo "⏭️ [00] INSTALL_DEPS=0 — skipping dep install. Env '$CONDA_ENV' ready."
    exit 0
fi

if [ ! -f "$HY3D_DIR/requirements.txt" ]; then
    echo "❌ $HY3D_DIR/requirements.txt not found." >&2
    echo "   Run run_all.sh first (it clones the official repo), or set HY3D_DIR." >&2
    exit 1
fi

echo "📦 [00] pip install -r requirements.txt"
# basicsr 的 sdist 构建依赖 torch；pip 构建隔离会重新下载整个 torch（慢且无谓）。
# 先用 --no-build-isolation --no-deps 装好它（复用 env 里的 torch），
# 依赖手动补（tb-nightly 不在国内镜像上，用 tensorboard 平替）。
pip install --no-build-isolation --no-deps "basicsr==1.4.2" || {
    echo "❌ basicsr preinstall failed" >&2; exit 1; }
pip install addict future lmdb yapf tensorboard || exit 1
# lightning_fabric（pytorch_lightning 依赖）import 需要 pkg_resources；
# setuptools>=81 已移除它，固定到 <81。
pip install -q "setuptools<81" || exit 1
# 推理链路用不到的包（官方 requirements 全量装）：tb_nightly（镜像没有）、
# deepspeed（训练用）、gradio/fastapi/uvicorn（demo）、cupy-cuda12x/open3d/
# pythreejs（核心代码未 import）、bpy（330MB，国内镜像没有；GLB 导出已由
# 本目录 _obj2glb.py 的 trimesh 实现替代）。
FILTERED_REQ="$(mktemp)"
grep -vE '^(tb_nightly|deepspeed|gradio|fastapi|uvicorn|cupy-cuda12x|open3d|pythreejs|bpy)([=<>=! ]|$)' \
    "$HY3D_DIR/requirements.txt" > "$FILTERED_REQ"
pip install -r "$FILTERED_REQ" || exit 1

# Two CUDA extensions ship as source and must be compiled in place.
# Needs nvcc (CUDA_HOME) + a C++ toolchain. Paint texture stage will crash
# without them.
# --no-build-isolation：setup.py import torch，构建隔离环境里没有 torch
#（同 basicsr 的坑）；env 里已有 torch 2.5.1 + ninja + pybind11。
# pip 版 torch 的 CUDA 头文件（cusparse/cublas/cusolver 等）在
# site-packages/nvidia/*/include，不在 CUDA_HOME/include 里；
# nvcc/host gcc 编译 ATen CUDAContextLight.h 时找不到 cusparse.h。
# 通过 CPLUS_INCLUDE_PATH / LIBRARY_PATH 把这些目录补进编译搜索路径。
NV_PKGS="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
if [ -d "$NV_PKGS" ]; then
    for d in "$NV_PKGS"/*/include; do
        [ -d "$d" ] && CPLUS_INCLUDE_PATH="$d${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    done
    for d in "$NV_PKGS"/*/lib; do
        [ -d "$d" ] && LIBRARY_PATH="$d${LIBRARY_PATH:+:$LIBRARY_PATH}"
    done
    export CPLUS_INCLUDE_PATH LIBRARY_PATH
    echo "🔧 [00] CPLUS_INCLUDE_PATH += nvidia pip headers ($(ls -d "$NV_PKGS"/*/include 2>/dev/null | wc -l) dirs)"
fi

# WSL 无系统 gcc/c++；conda toolchain 只提供 x86_64-conda-linux-gnu-{c++,cc,...}。
# 官方编译脚本写死 `c++`，这里做一层 shim 软链补齐通用名。
if ! command -v c++ >/dev/null 2>&1; then
    SHIM_DIR="$(mktemp -d)"
    for t in c++ g++ cc gcc; do
        tc="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-$t"
        [ -x "$tc" ] && ln -sf "$tc" "$SHIM_DIR/$t"
    done
    if [ -x "$SHIM_DIR/c++" ]; then
        export PATH="$SHIM_DIR:$PATH"
        echo "🔧 [00] shimmed c++/g++/cc/gcc -> conda toolchain ($CONDA_PREFIX)"
    else
        echo "⚠️ [00] no C++ compiler found; compile steps may fail" >&2
    fi
fi

echo "🔧 [00] compiling custom_rasterizer (CUDA_HOME=$CUDA_HOME)"
( cd "$HY3D_DIR/hy3dpaint/custom_rasterizer" && pip install --no-build-isolation -e . ) || {
    echo "❌ custom_rasterizer build failed" >&2; exit 1; }

echo "🔧 [00] compiling DifferentiableRenderer (compile_mesh_painter.sh)"
( cd "$HY3D_DIR/hy3dpaint/DifferentiableRenderer" && bash compile_mesh_painter.sh ) || {
    echo "❌ DifferentiableRenderer build failed" >&2; exit 1; }

# RealESRGAN upscaler weights used by the paint stage (albedo/MR enhance).
ESRGAN_DIR="$HY3D_DIR/hy3dpaint/ckpt"
ESRGAN_PTH="$ESRGAN_DIR/RealESRGAN_x4plus.pth"
ESRGAN_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
# GitHub releases 本机不可达时的镜像（实测可用 2026-10-28）
ESRGAN_MS_URL="https://modelscope.cn/api/v1/models/AI-ModelScope/RealESRGAN_x4plus/repo?Revision=master&FilePath=RealESRGAN_x4plus.pth"
if [ ! -f "$ESRGAN_PTH" ]; then
    echo "📦 [00] downloading RealESRGAN_x4plus.pth"
    mkdir -p "$ESRGAN_DIR"
    wget -q --timeout=30 "$ESRGAN_URL" -P "$ESRGAN_DIR" \
        || curl -kL --max-time 120 "$ESRGAN_URL" -o "$ESRGAN_PTH" \
        || curl -sL --max-time 300 "$ESRGAN_MS_URL" -o "$ESRGAN_PTH" \
        || { echo "❌ RealESRGAN download failed (try manual download to $ESRGAN_PTH)" >&2; exit 1; }
    # 下载失败可能留下 0 字节/小文件（错误页），校验大小 ~64MB
    SZ=$(stat -c%s "$ESRGAN_PTH" 2>/dev/null || echo 0)
    if [ "$SZ" -lt 60000000 ]; then
        echo "❌ RealESRGAN file too small ($SZ bytes), likely an error page" >&2
        rm -f "$ESRGAN_PTH"; exit 1
    fi
fi

echo "🎉 [00] Done. Env '$CONDA_ENV' ready."
