#!/usr/bin/env bash
# 00_setup_env.sh —— 建 conda env + 装依赖（WSL 本机，优先复用 D:\wheel 的离线 wheel）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

PY_VER="${PY_VER:-3.11}"          # 必须 3.11：D:\wheel 里的 torch/nvidia/flash-attn 全是 cp311
TORCH_VER="${TORCH_VER:-2.8.0}"    # 满足 >=2.5；与本地 nvidia-cu12.8 全套 wheel 对齐
TV_VER="${TV_VER:-0.23.0}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"

echo "🚀 [00] 建环境 idol (python $PY_VER)"
echo "  📦 wheel 目录: $WHEEL_DIR"
echo "  📁 HF 缓存:    $HF_HOME"
echo "  🌐 代理:       ${https_proxy:-<无>}"

# ---------- 0) 官方代码 clone 到 WSL Linux fs ----------
if [ ! -d "$IDOL_REPO/idol" ]; then
    echo "📥 clone 官方仓 -> $IDOL_REPO"
    mkdir -p "$(dirname "$IDOL_REPO")"
    git clone --depth 1 https://github.com/speridlabs/idol-sdk.git "$IDOL_REPO" \
        || { echo "❌ clone 失败（github 是否需要代理？）" >&2; exit 1; }
else
    echo "⏭️  官方仓已存在: $IDOL_REPO"
fi

# ---------- 1) conda env ----------
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "⏭️  env '$CONDA_ENV' 已存在，跳过创建（重建用 RECREATE=1）"
    if [ "${RECREATE:-0}" = "1" ]; then
        conda env remove -n "$CONDA_ENV" -y >/dev/null 2>&1
        conda create -n "$CONDA_ENV" "python=$PY_VER" -y
    fi
else
    conda create -n "$CONDA_ENV" "python=$PY_VER" -y || { echo "❌ env 创建失败" >&2; exit 1; }
fi
conda activate "$CONDA_ENV" || { echo "❌ conda activate 失败" >&2; exit 1; }
python -c "import sys; print('  🐍', sys.version.split()[0], sys.executable)"

# ---------- 2) torch 栈：优先离线 wheel ----------
if python -c "import torch" 2>/dev/null; then
    echo "⏭️  torch 已装: $(python -c 'import torch;print(torch.__version__)')"
else
    echo "📦 装 torch $TORCH_VER + torchvision $TV_VER（强制走 $WHEEL_DIR 本地 wheel）"
    # 两步法，别偷懒合成一条：
    # ① torch/torchvision 的纯 python 依赖（filelock/sympy/networkx/numpy/pillow...）
    #    D:\wheel 里没有，先联网补齐；
    # ② 再用 --no-index，逼 pip 只用本地 wheel —— 否则 pip 会无视 --find-links，
    #    从 index 重下 888MB torch + ~3GB nvidia-* 运行库（实测浪费 10+ 分钟）。
    # ① 的清单是实测缺啥补啥：缺 filelock、缺 numpy 都会让 ② 直接 "No matching distribution"。
    pip install "numpy<2" pillow filelock sympy networkx jinja2 fsspec \
                typing-extensions setuptools packaging pyyaml requests
    pip install --no-index --find-links "$WHEEL_DIR" "torch==$TORCH_VER" "torchvision==$TV_VER"
    if [ $? -ne 0 ]; then
        echo "⚠️  本地 wheel 不完整，退回联网装"
        pip install "torch==$TORCH_VER" "torchvision==$TV_VER"
    fi
fi

# ---------- 3) python 层依赖（全部钉版本，别放开区间） ----------
echo "📦 装 idol 运行依赖"
# 为什么全部钉死：实测放开 "transformers>=4.56" + "tensordict>=0.8" 会被解析到
# transformers 5.22 / tensordict 新版，后者硬要求 torch>=2.11，直接把刚装的
# torch 2.8 顶成 torch 2.15（多下 540MB + 与本机 CUDA wheel 全不对齐）。
TRANSFORMERS_VER="${TRANSFORMERS_VER:-4.57.6}"   # D:\wheel 有本地 wheel
DIFFUSERS_VER="${DIFFUSERS_VER:-0.36.0}"          # AutoencoderKLFlux2 需 >=0.36
HUB_VER="${HUB_VER:-0.36.2}"                      # D:\wheel 有本地 wheel
TENSORDICT_VER="${TENSORDICT_VER:-0.8.3}"        # 只用 TensorDict 容器，0.8.x 与 torch 2.8 配套

pip install "huggingface_hub==$HUB_VER" "transformers==$TRANSFORMERS_VER" \
            "diffusers==$DIFFUSERS_VER" "safetensors>=0.4" \
            "numpy<2" "pillow>=10.0" "plyfile>=1.0" "tqdm>=4.66" "hf_transfer" "einops"
if [ $? -ne 0 ]; then
    echo "❌ 依赖安装失败" >&2; exit 1
fi

# tensordict 必须 --no-deps：否则它会把 torch 拖到 >=2.11
if ! python -c "import tensordict" 2>/dev/null; then
    pip install --no-deps "tensordict==$TENSORDICT_VER" \
        || pip install --no-deps "tensordict" \
        || { echo "❌ tensordict 安装失败" >&2; exit 1; }
fi
# --no-deps 的代价：tensordict 自己的纯 python 依赖要手工补（实测漏 cloudpickle，
# 报 "No module named 'cloudpickle'" 挂在 import idol 那一步）
pip install cloudpickle orjson

# 装完复查 torch 没被顶掉
python - <<PY
import torch
v = torch.__version__
print(f"  🔎 torch 复查: {v}")
if not v.startswith("$TORCH_VER"):
    raise SystemExit(f"❌ torch 被依赖顶成了 {v}，期望 $TORCH_VER —— 检查本脚本第 3 步的版本钉")
PY
if [ $? -ne 0 ]; then
    echo "❌ torch 版本被污染" >&2; exit 1
fi

# ---------- 4) flash-attn（可选，失败不阻塞） ----------
if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
    if python -c "import flash_attn" 2>/dev/null; then
        echo "⏭️  flash-attn 已装"
    else
        FA_WHL="$(ls "$WHEEL_DIR"/flash_attn-*cp311*.whl 2>/dev/null | head -1)"
        if [ -n "$FA_WHL" ]; then
            echo "📦 尝试离线装 $(basename "$FA_WHL")"
            pip install --no-deps "$FA_WHL" || echo "⚠️  flash-attn ABI 不匹配，退回 sdpa（结果一致，仅变慢）"
        else
            echo "⏭️  无本地 flash-attn wheel，用 sdpa"
        fi
    fi
fi

# ---------- 5) 官方代码以 editable 方式装（源码留在 repo，不改动） ----------
if [ ! -d "$IDOL_REPO/idol" ]; then
    echo "❌ ERROR: 官方代码不在 $IDOL_REPO" >&2
    exit 1
fi
echo "📦 pip install -e $IDOL_REPO（--no-deps，避免重解析 torch）"
pip install -e "$IDOL_REPO" --no-deps || { echo "❌ editable 安装失败" >&2; exit 1; }

# ---------- 6) 验证 ----------
echo "🔍 验证环境"
python - <<'PY'
import sys
ok = True
import torch
print(f"  torch {torch.__version__} | cuda_available={torch.cuda.is_available()}"
      + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
if not torch.cuda.is_available():
    print("  ⚠️  CUDA 不可用，推理会退到 CPU（极慢）")
try:
    import transformers
    from transformers import AutoModel
    print(f"  transformers {transformers.__version__}")
except Exception as e:
    ok = False; print(f"  ❌ transformers: {e}")
try:
    import diffusers
    from diffusers.models import AutoencoderKLFlux2
    print(f"  diffusers {diffusers.__version__} | AutoencoderKLFlux2 OK")
except Exception as e:
    ok = False; print(f"  ❌ diffusers/AutoencoderKLFlux2: {e}")
try:
    import idol
    print(f"  idol import OK -> {idol.__file__}")
except Exception as e:
    ok = False; print(f"  ❌ idol import: {e}")
sys.exit(0 if ok else 1)
PY
if [ $? -ne 0 ]; then
    echo "❌ 验证未通过" >&2
    exit 1
fi

echo "🎉 [00] Done. 下一步: bash 01_download_models.sh"
