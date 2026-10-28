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
# 先用 --no-build-isolation 装好它（复用 env 里的 torch），requirements 就不会再触发源码构建。
pip install --no-build-isolation "basicsr==1.4.2" || {
    echo "❌ basicsr preinstall failed" >&2; exit 1; }
pip install -r "$HY3D_DIR/requirements.txt" || exit 1

# Two CUDA extensions ship as source and must be compiled in place.
# Needs nvcc (CUDA_HOME) + a C++ toolchain. Paint texture stage will crash
# without them.
echo "🔧 [00] compiling custom_rasterizer (CUDA_HOME=$CUDA_HOME)"
( cd "$HY3D_DIR/hy3dpaint/custom_rasterizer" && pip install -e . ) || {
    echo "❌ custom_rasterizer build failed" >&2; exit 1; }

echo "🔧 [00] compiling DifferentiableRenderer (compile_mesh_painter.sh)"
( cd "$HY3D_DIR/hy3dpaint/DifferentiableRenderer" && bash compile_mesh_painter.sh ) || {
    echo "❌ DifferentiableRenderer build failed" >&2; exit 1; }

# RealESRGAN upscaler weights used by the paint stage (albedo/MR enhance).
ESRGAN_DIR="$HY3D_DIR/hy3dpaint/ckpt"
ESRGAN_PTH="$ESRGAN_DIR/RealESRGAN_x4plus.pth"
ESRGAN_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
if [ ! -f "$ESRGAN_PTH" ]; then
    echo "📦 [00] downloading RealESRGAN_x4plus.pth"
    mkdir -p "$ESRGAN_DIR"
    wget -q "$ESRGAN_URL" -P "$ESRGAN_DIR" || curl -kL "$ESRGAN_URL" -o "$ESRGAN_PTH" || {
        echo "❌ RealESRGAN download failed (try manual download to $ESRGAN_PTH)" >&2; exit 1; }
fi

echo "🎉 [00] Done. Env '$CONDA_ENV' ready."
