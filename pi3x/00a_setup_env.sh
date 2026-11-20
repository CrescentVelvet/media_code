#!/usr/bin/env bash
# 00a_setup_env.sh — WSL 变体：从零建 pi3x env + clone 官方仓 + 装依赖 + 验证。
#
# 用法:
#   bash pi3x/00a_setup_env.sh            # 幂等：已建好的步骤自动跳过
#   INSTALL_DEPS=0 bash pi3x/00a_setup_env.sh   # 只 clone + 验证
#
# 依赖策略（WSL 实测网络）：
#   - torch 2.5.1+cu124 优先用本地 wheel /mnt/d/wheel/torch-2.5.1+cu124-cp310-*.whl，
#     缺了再从 download.pytorch.org/whl/cu124 拉（pytorch.org 直连可用，勿走代理）。
#   - nvidia-* 运行库走 tuna pypi（2026-11-20 实测：download.pytorch.org 的 nvidia wheel
#     会卡死 0 字节；PyPI 上同版本 wheel 一样，tuna 满速）。版本按 torch 2.5.1 元数据钉死。
#   - 纯 python 小包走 tuna pypi 镜像（proxy.env 的 no_proxy 已排除国内源）。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

INSTALL_DEPS="${INSTALL_DEPS:-1}"
LOCAL_TORCH_WHEEL="${LOCAL_TORCH_WHEEL:-/mnt/d/wheel/torch-2.5.1+cu124-cp310-cp310-linux_x86_64.whl}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "🚀 [00a] pi3x 环境准备"
echo "  🐍 env: $CONDA_ENV (python 3.10)"
echo "  📦 官方仓: $PI3_DIR"

# --- 1. conda env ---
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "⏭️  env '$CONDA_ENV' 已存在，跳过创建"
else
    echo "📦 创建 env '$CONDA_ENV' (python=3.10) ..."
    conda create -n "$CONDA_ENV" python=3.10 -y || { echo "❌ conda create 失败" >&2; exit 1; }
fi
conda activate "$CONDA_ENV" || { echo "❌ conda activate $CONDA_ENV 失败" >&2; exit 1; }
python -c "import platform,sys; assert platform.python_implementation()=='CPython', 'not CPython!'; print('  🐍', platform.python_version())"

# --- 2. clone 官方仓 ---
if [ -d "$PI3_DIR/pi3" ]; then
    echo "⏭️  $PI3_DIR 已存在，跳过 clone"
else
    mkdir -p "$(dirname "$PI3_DIR")"
    git clone --depth 1 "$PI3_REPO" "$PI3_DIR" || { echo "❌ git clone 失败" >&2; exit 1; }
fi

if [ "$INSTALL_DEPS" = "1" ]; then
    # --- 3. torch 栈（本地 wheel 优先）---
    if python -c "import torch" 2>/dev/null; then
        echo "⏭️  torch 已装: $(python -c 'import torch; print(torch.__version__)')"
    else
        echo "📦 安装 torch 2.5.1+cu124 ..."
        if [ -f "$LOCAL_TORCH_WHEEL" ]; then
            echo "  🎯 用本地 wheel: $LOCAL_TORCH_WHEEL"
            # 本地 wheel 是 cu124 构建，依赖的 nvidia-* 运行库仍需从 cu124 index 补齐；
            # 先装 wheel 本体（--no-deps 防 pip 把 torch 换成 PyPI 版），再补依赖。
            pip install --no-deps "$LOCAL_TORCH_WHEEL" || { echo "❌ torch wheel 安装失败" >&2; exit 1; }
            # nvidia 依赖 + triton：tuna 优先（版本 = torch 2.5.1 元数据钉版），pytorch index 兜底。
            # nvidia-cufile-cu12==1.9.1.3 tuna 上没有（最低 1.11），且仅 GPUDirect 文件 IO 用，
            # 前馈推理不需要 → 单列可选安装，失败忽略。
            pip install --index-url "$PYPI_INDEX" \
                "triton==3.1.0" \
                "nvidia-cublas-cu12==12.4.5.8" "nvidia-cuda-cupti-cu12==12.4.127" \
                "nvidia-cuda-nvrtc-cu12==12.4.127" "nvidia-cuda-runtime-cu12==12.4.127" \
                "nvidia-cudnn-cu12==9.1.0.70" "nvidia-cufft-cu12==11.2.1.3" \
                "nvidia-curand-cu12==10.3.5.147" \
                "nvidia-cusolver-cu12==11.6.1.9" "nvidia-cusparse-cu12==12.3.1.170" \
                "nvidia-cusparselt-cu12==0.6.2" "nvidia-nccl-cu12==2.21.5" \
                "nvidia-nvjitlink-cu12==12.4.127" "nvidia-nvtx-cu12==12.4.127" \
                || pip install --index-url "$PYTORCH_INDEX" \
                "triton==3.1.0" \
                nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
                nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
                nvidia-curand-cu12 nvidia-cusolver-cu12 \
                nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
                nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 \
                || { echo "❌ nvidia 依赖安装失败" >&2; exit 1; }
            pip install --index-url "$PYPI_INDEX" nvidia-cufile-cu12 2>/dev/null || true
        else
            pip install --index-url "$PYTORCH_INDEX" torch==2.5.1 \
                || { echo "❌ torch 安装失败" >&2; exit 1; }
        fi
    fi
    if python -c "import torchvision" 2>/dev/null; then
        echo "⏭️  torchvision 已装: $(python -c 'import torchvision; print(torchvision.__version__)')"
    else
        # --no-deps：官方 pin 是 torch==2.5.1（本地 cu124 wheel 版本串带后缀，严格 == 不匹配会触发重装）
        pip install --no-deps --index-url "$PYTORCH_INDEX" torchvision==0.20.1 \
            || { echo "❌ torchvision 安装失败" >&2; exit 1; }
    fi

    # --- 4. 其余依赖（tuna 镜像）---
    echo "📦 安装其余依赖 ..."
    pip install --index-url "$PYPI_INDEX" \
        "numpy==1.26.4" pillow opencv-python plyfile safetensors huggingface_hub pillow-heif \
        || { echo "❌ 依赖安装失败" >&2; exit 1; }
fi

# --- 5. 验证 ---
echo "🔍 验证 imports ..."
python - "$PI3_DIR" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
import torch, torchvision, numpy, cv2, plyfile, safetensors, PIL, pillow_heif
print("  torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| dev:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
from pi3.models.pi3x import Pi3X
from pi3.utils.basic import load_multimodal_data, write_ply
from pi3.utils.geometry import depth_normal_edge, recover_intrinsic_from_rays_d
print("  ✅ pi3 (Pi3X) imports OK")
PYEOF
if [ $? -ne 0 ]; then
    echo "❌ FAILED: import 验证没过" >&2
    exit 1
fi
echo "🎉 [00a] Done."
