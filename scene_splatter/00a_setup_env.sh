#!/usr/bin/env bash
# Scene Splatter WSL 本机构建 env（无 sudo，全部走 conda + pip）
#
# 官方依赖（README）：python 3.10 + torch 2.1.2/cu118 + requirements.txt +
# 编译 gaussianSplatting/submodules/diff-gaussian-rasterization（需要 nvcc）。
# 本机 pytorch.org 一直可用（AGENTS.md），torch 走官方 cu118 index；
# 其余 pip 包走阿里镜像；nvcc 用 conda 的 cuda-11.8 label（WSL 无 sudo 不能装系统 CUDA）。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

INSTALL_DEPS="${INSTALL_DEPS:-1}"
PY_VER="${PY_VER:-3.10}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
WHEELS_DIR="${WHEELS_DIR:-/mnt/d/wheel}"

echo "🚀 [00a] Scene Splatter WSL env setup"
echo "  📁 官方仓: $SS_DIR"
echo "  🐍 conda:  $CONDA_ENV ($PY_VER)"
echo "  📦 pip:    $PIP_INDEX"
echo "  🎮 GPU:    ${CUDA_VISIBLE_DEVICES:-all}"

# ── 1) clone 官方仓（GitHub 本机直连可用；仓内 gaussianSplatting/flash3d/viewcrafter
#       全是 vendored 普通目录，无 submodule，plain clone 即可）──
if [ ! -d "$SS_DIR/scenesplatter.py" ]; then
    echo "📦 cloning Scene-Splatter → $SS_DIR"
    mkdir -p "$(dirname "$SS_DIR")"
    LD_LIBRARY_PATH= git clone https://github.com/shengjun-zhang/Scene-Splatter "$SS_DIR" || {
        echo "❌ clone failed" >&2; exit 1; }
else
    echo "⏭️  repo exists: $SS_DIR"
fi

# ── 2) conda env ──
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "📦 creating conda env $CONDA_ENV (python=$PY_VER)"
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1
    conda create -n "$CONDA_ENV" python="$PY_VER" -y || { echo "❌ conda create failed" >&2; exit 1; }
else
    echo "⏭️  env exists: $CONDA_ENV"
fi

if [ "$INSTALL_DEPS" != "1" ]; then
    echo "⏭️  INSTALL_DEPS=0, skip pip install"
    echo "🎉 Done (env only)."
    exit 0
fi

PY="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
PIP_FLAGS=(-i "$PIP_INDEX" --timeout 600 --retries 5)
[ -d "$WHEELS_DIR" ] && PIP_FLAGS+=(--find-links "$WHEELS_DIR")

# ── 3) torch 2.1.2 + torchvision 0.16.2（官方指定 cu118；pytorch.org 直连可用）──
echo "📦 installing torch 2.1.2+cu118 + torchvision 0.16.2+cu118"
"$PY" -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118 || {
    echo "❌ torch install failed" >&2; exit 1; }

# ── 4) requirements.txt（去 torch 三件套，官方 README 单独装）──
# ⚠️ requirements 里 pip==23.0.1 会降级 pip，属官方原样行为，不动它。
echo "📦 installing requirements.txt"
"$PY" -m pip install "${PIP_FLAGS[@]}" -r "$SS_DIR/requirements.txt" || {
    echo "❌ requirements install failed" >&2; exit 1; }

# ── 5) nvcc（编译 diff-gaussian-rasterization 用；conda 提供，无需 sudo）──
if [ ! -x "$HOME/miniconda3/envs/$CONDA_ENV/bin/nvcc" ]; then
    echo "📦 installing cuda-nvcc 11.8 (conda)"
    conda install -n "$CONDA_ENV" -c nvidia/label/cuda-11.8.0 cuda-nvcc cuda-cudart-dev -y || {
        echo "❌ cuda-nvcc install failed" >&2; exit 1; }
else
    echo "⏭️  nvcc exists"
fi
export CUDA_HOME="$HOME/miniconda3/envs/$CONDA_ENV"
export PATH="$CUDA_HOME/bin:$PATH"

# ── 6) 编译 diff-gaussian-rasterization（3DGS CUDA 光栅器）──
echo "🔨 building diff-gaussian-rasterization"
cd "$SS_DIR/gaussianSplatting/submodules/diff-gaussian-rasterization" || {
    echo "❌ submodule dir missing: $SS_DIR/gaussianSplatting/submodules/diff-gaussian-rasterization" >&2; exit 1; }
if ! "$PY" -c "import diff_gaussian_rasterization" 2>/dev/null; then
    "$PY" -m pip install . || { echo "❌ rasterizer build failed" >&2; exit 1; }
else
    echo "⏭️  diff_gaussian_rasterization already installed"
fi

# ── 7) 打本仓补丁（官方 main() 不落盘 PLY，补 scene.ply 导出）──
echo "📦 applying media_code patches"
python3 "$SCRIPT_DIR/patch_save_ply.py" || true

# ── 8) 验证 ──
echo "🔍 verifying imports"
cd "$SS_DIR" || exit 1
"$PY" - <<'EOF'
import torch
print(f"✅ torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
from gaussianSplatting.scene import GaussianModel
from gaussianSplatting.gaussian_renderer import render
print("✅ gaussianSplatting import ok")
import flash3d  # noqa: F401
print("✅ flash3d import ok")
EOF
if [ $? -ne 0 ]; then
    echo "❌ import check failed" >&2
    exit 1
fi

echo "🔍 官方示例相机轨迹结构（核对约定用，详见 NOTES.md 第 4 节）"
python3 "$SCRIPT_DIR/02_gen_traj.py" --inspect-example || true

echo "🎉 Done. 下一步：bash scene_splatter/01_download_models.sh"
