#!/usr/bin/env bash
# 00a_setup_env.sh — WSL 建 vggt_omega env + 验证官方仓
# 本目录的推理/训练链路整体转发给 vggt_human（长期验证过的 VGGT-Ω + 3DGS 链路），
# 本 env 只装 vggt-omega 官方包本体，供独立调用其 API（demo / 自定义脚本）用。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [00a] vggt_omega 环境准备"

# 0) 官方仓
if [ ! -d "$VGGT_OMEGA_REPO/vggt_omega" ]; then
    echo "📦 clone 官方仓: github.com/facebookresearch/vggt-omega -> $VGGT_OMEGA_REPO"
    git clone --depth 1 https://github.com/facebookresearch/vggt-omega.git "$VGGT_OMEGA_REPO"
    if [ $? -ne 0 ]; then echo "❌ clone 失败（网络问题先看 proxy.env）" >&2; exit 1; fi
else
    echo "⏭️ 官方仓已存在: $VGGT_OMEGA_REPO"
fi

# 1) env：clone gen3r（python 3.11 + torch 2.8.0+cu128），已存在则跳过
if conda env list | grep -qw vggt_omega; then
    echo "⏭️ env vggt_omega 已存在"
else
    if conda env list | grep -qw gen3r; then
        echo "📦 conda create -n vggt_omega --clone gen3r"
        conda create -n vggt_omega --clone gen3r -y
    else
        conda create -n vggt_omega python=3.11 -y
    fi
    if [ $? -ne 0 ]; then echo "❌ conda create 失败" >&2; exit 1; fi
fi
conda activate vggt_omega

# 2) 官方包本体（numpy<2 与 gen3r 兼容；einops/safetensors 补齐）
echo "📦 pip install -e $VGGT_OMEGA_REPO"
pip install -i https://mirrors.aliyun.com/pypi/simple/ --find-links /mnt/d/wheel \
    einops safetensors "numpy<2"
pip install -i https://mirrors.aliyun.com/pypi/simple/ --find-links /mnt/d/wheel \
    --no-build-isolation -e "$VGGT_OMEGA_REPO"
if [ $? -ne 0 ]; then echo "❌ 安装失败" >&2; exit 1; fi

echo "🔍 验证 import"
python -c "from vggt_omega.models import VGGTOmega; import torch; print(f'✅ vggt_omega import OK, torch {torch.__version__}')"
if [ $? -ne 0 ]; then echo "❌ import 验证失败" >&2; exit 1; fi
echo "🎉 Done. 下一步: bash vggt_omega/01_download_models.sh（gated 权重需先申请）"
