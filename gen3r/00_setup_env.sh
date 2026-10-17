#!/usr/bin/env bash
# 00_setup_env.sh — 服务器版环境准备（容器内已有 conda / CUDA toolkit）
#
# WSL 本机请用 00a_setup_env.sh（差异见 README_wsl.md）。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

INSTALL_DEPS="${INSTALL_DEPS:-1}"
GEN3R_DIR="${GEN3R_DIR:-$(dirname "$REPO_DIR")/Gen3R}"

echo "🚀 [00] Gen3R env setup (server)"
echo "  📁 repo: $GEN3R_DIR"

# ── 1) clone 官方仓到 media_code 同级 ──
if [ ! -d "$GEN3R_DIR/gen3r" ]; then
    echo "📦 cloning → $GEN3R_DIR"
    LD_LIBRARY_PATH= git clone https://github.com/JaceyHuang/Gen3R "$GEN3R_DIR" || exit 1
else
    echo "⏭️  repo exists"
fi

if [ "$INSTALL_DEPS" != "1" ]; then
    echo "🎉 Done (repo only)."
    exit 0
fi

# ── 2) 依赖（官方 requirements.txt 去掉 flash-attn / pytorch3d：
#       前者代码有 try/except fallback，后者只被 eval_utils 用于评估指标）──
echo "📦 installing deps into env '$CONDA_ENV'"
pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    diffusers==0.33.1 transformers==4.46.3 accelerate==1.0.1 safetensors \
    einops==0.8.0 imageio==2.35.1 imageio-ffmpeg==0.5.1 numpy==1.26.0 \
    open3d==0.18.0 easydict==1.13 pillow opencv-python-headless==4.10.0.84 || exit 1

# ── 3) 验证 ──
cd "$GEN3R_DIR" || exit 1
python - <<'EOF'
import torch
print(f"✅ torch {torch.__version__} cuda={torch.cuda.is_available()}")
from gen3r.pipeline import Gen3RPipeline
print("✅ pipeline import ok")
EOF
[ $? -ne 0 ] && { echo "❌ import check failed" >&2; exit 1; }

echo "🎉 Done. 下一步：bash gen3r/01_download_models.sh"
