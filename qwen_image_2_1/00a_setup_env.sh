#!/usr/bin/env bash
# 00a_setup_env.sh — WSL 本机从零建 conda env 并装依赖。
# 依赖（官方 README 口径）：torch>=2.4、transformers>=5.17、git 版 diffusers、accelerate、pillow。
# torch 2.8.0 及 nvidia-cu12 全家桶在 /mnt/d/wheel 有现成 wheel（cp311），直接复用不重下。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

WHEELS_DIR="${WHEELS_DIR:-/mnt/d/wheel}"
DIFFUSERS_DIR="${DIFFUSERS_DIR:-$HOME/repos/diffusers}"
PIP_FLAGS=(-i https://mirrors.aliyun.com/pypi/simple --find-links "$WHEELS_DIR" --timeout 600 --retries 5)

echo "🚀 [00a] 准备 conda env: $CONDA_ENV"

# conda 26.x 非交互 accept ToS（幂等，已 accept 时静默通过）
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null

if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "📦 创建 env: $CONDA_ENV (python=3.11)"
    conda create -n "$CONDA_ENV" python=3.11 -y
fi
conda activate "$CONDA_ENV"

echo "📦 安装 torch（优先 $WHEELS_DIR 本地 wheel）"
pip install "${PIP_FLAGS[@]}" torch==2.8.0

echo "📦 安装 transformers>=5.17 / accelerate / pillow / safetensors"
pip install "${PIP_FLAGS[@]}" -U "transformers>=5.17" accelerate pillow safetensors

# diffusers 官方要求 git 版（QwenImage21Pipeline 未进 release）。
# 不直装 git+https：pip 的 git 子进程会踩 conda libffi 与系统 libp11-kit 冲突，
# 改为手动 clone（LD_LIBRARY_PATH= 绕开）+ pip install -e。
if [ ! -d "$DIFFUSERS_DIR/.git" ]; then
    echo "📦 clone diffusers -> $DIFFUSERS_DIR"
    mkdir -p "$(dirname "$DIFFUSERS_DIR")"
    LD_LIBRARY_PATH= git clone --depth 1 https://github.com/huggingface/diffusers.git "$DIFFUSERS_DIR"
fi
echo "📦 安装 diffusers（本地源码，editable）"
pip install "${PIP_FLAGS[@]}" -e "$DIFFUSERS_DIR"

echo "🔍 验证"
python - <<'PY'
import torch, transformers, diffusers
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__} diffusers={diffusers.__version__}")
from diffusers import QwenImage21Pipeline
print(f"QwenImage21Pipeline OK: {QwenImage21Pipeline.__module__}")
if not torch.cuda.is_available():
    raise SystemExit("❌ torch.cuda 不可用")
PY
if [ $? -ne 0 ]; then
    echo "❌ 验证失败" >&2
    exit 1
fi

# pip HTTP 缓存吃 vhdx 空间，装完清掉（要留的 wheel 已在 /mnt/d/wheel）
pip cache purge >/dev/null 2>&1

echo "🎉 Done. env='$CONDA_ENV'"
