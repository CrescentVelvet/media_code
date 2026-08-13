#!/usr/bin/env bash
# 00_setup_env.sh — clone World-R1 官方仓, 建 conda env, 装全部依赖, 验证。
#
# First time:
#   INSTALL_DEPS=1 bash world_r1/00_setup_env.sh
# After that (verify only):
#   bash world_r1/00_setup_env.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [00] World-R1 环境搭建 + 验证"
echo "  🤖 conda env: $CONDA_ENV"
echo "  📁 官方代码:  $WORLD_R1_DIR"
echo "  🏋️ 权重根:    $MODEL_DIR"
echo ""

# _env.sh tolerated a missing env; create it now if needed.
# NB: Python 3.10 (CPython) — World-R1 要求 3.10+; gxx 装 CUDA 扩展也要 cp310。
if ! conda env list 2>/dev/null | grep -qw "$CONDA_ENV"; then
    echo "📦 conda env '$CONDA_ENV' not found; creating python=3.10 (CPython)"
    conda create -n "$CONDA_ENV" python=3.10 -y
    conda activate "$CONDA_ENV"
    # 防御：conda-forge 偶尔把 python 实现掉包成 GraalPy, cp310 wheel 全坏。
    impl="$(python -c 'import platform; print(platform.python_implementation())')"
    if [ "$impl" != "CPython" ]; then
        echo "❌ env python 实现是 $impl（应为 CPython），重建..." >&2
        conda env remove -n "$CONDA_ENV"
        conda create -n "$CONDA_ENV" python=3.10 -y --override-channels -c defaults
        conda activate "$CONDA_ENV"
    fi
    echo "  ✅ python=$(python --version 2>&1 | cut -d' ' -f2) ($(python -c 'import platform; print(platform.python_implementation())'))"
fi

# --- 0. Install deps (first time) ---
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)
    _proxy="${https_proxy:-${http_proxy:-}}"
    if [ -n "$_proxy" ]; then
        PIP_FLAGS+=(--proxy "$_proxy")
        echo "  🌐 using proxy: $_proxy"
    else
        echo "  ⚠️ no proxy set (http_proxy/https_proxy); pip 可能失败" >&2
        echo "     在仓根建 proxy.env: 见 proxy.env.example" >&2
    fi

    # 0a. PyTorch (cu124)
    #     公司代理封 download.pytorch.org (403) → 用 PyPI 默认源 (torch 自带 cu12x)。
    #     若本地有 cp310 torch wheel, 优先本地装 (参考 wan22_rotate)。
    echo "📦 installing PyTorch (cu124)"
    _local_torch=0
    for w in "$MODEL_DIR"/torch-2.*cu124*cp310*.whl "$MODEL_DIR"/torchvision-0.*cu124*cp310*.whl; do
        if [ -f "$w" ]; then
            echo "  found local wheel: $w"
            pip install --force-reinstall --no-deps "$w"
            _local_torch=1
        fi
    done
    if [ "$_local_torch" = "0" ]; then
        echo "  no local torch wheel; installing from PyPI (torch 自带 CUDA)"
        pip install "${PIP_FLAGS[@]}" torch torchvision
    fi
    echo "  ✅ torch=$(python -c 'import torch; print(torch.__version__)')  cuda=$(python -c 'import torch; print(torch.version.cuda)')"

    # 0b. gxx 12 (编 gsplat 等 CUDA 扩展要的; python=3.10 pin 防 GraalPy 掉包)
    echo "📦 installing gxx_linux-64=12 (for gsplat CUDA ext compile)"
    conda install -y -c conda-forge --no-update-deps gxx_linux-64=12 python=3.10 || \
        echo "  ⚠️ gxx install failed; gsplat 可能编不了" >&2
    impl2="$(python -c 'import platform; print(platform.python_implementation())' 2>/dev/null || echo unknown)"
    if [ "$impl2" != "CPython" ]; then
        echo "  ⚠️ gxx 把 python 掉包成 $impl2, 修复..." >&2
        conda install -y -c defaults python=3.10 --force-reinstall
    fi
    export CC="${CC:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc}"
    export CXX="${CXX:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++}"
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    # auto-detect CUDA 12.x (/usr/local/cuda 可能指向 11.8)
    if [ -x "$CUDA_HOME/bin/nvcc" ]; then
        _nvcc_ver="$($CUDA_HOME/bin/nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' || echo '')"
        _torch_ver="$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo '')"
        if [ -n "$_nvcc_ver" ] && [ -n "$_torch_ver" ]; then
            if [ "${_nvcc_ver%%.*}" != "${_torch_ver%%.*}" ]; then
                echo "  ⚠️ CUDA_HOME=$CUDA_HOME 是 $_nvcc_ver 但 torch 是 $_torch_ver"
                for _d in /usr/local/cuda-${_torch_ver%%.*}*; do
                    if [ -x "$_d/bin/nvcc" ]; then
                        export CUDA_HOME="$_d"
                        echo "  → switched CUDA_HOME to $CUDA_HOME"
                        break
                    fi
                done
            fi
        fi
        export PATH="$CUDA_HOME/bin:$PATH"
        echo "  ✅ nvcc: $($CUDA_HOME/bin/nvcc --version | tail -1 | xargs)"
    else
        echo "  ⚠️ nvcc not found at $CUDA_HOME/bin/nvcc — gsplat CUDA ext NOT built." >&2
    fi

    # 0c. clone World-R1 官方仓
    echo "📦 cloning World-R1 -> $WORLD_R1_DIR"
    if [ ! -d "$WORLD_R1_DIR/.git" ]; then
        mkdir -p "$(dirname "$WORLD_R1_DIR")"
        LD_LIBRARY_PATH= git clone https://github.com/microsoft/World-R1.git "$WORLD_R1_DIR" || \
            LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/microsoft/World-R1.git "$WORLD_R1_DIR"
    fi
    echo "  ✅ World-R1 cloned"

    # 0d. pip install -e World-R1 (flow_grpo + reward_server 包)
    echo "📦 installing World-R1 (flow_grpo + reward_server, editable)"
    pip install "${PIP_FLAGS[@]}" -e "$WORLD_R1_DIR"

    # 0e. 核心运行时依赖 (训练 + 推理)
    echo "📦 installing core runtime deps"
    pip install "${PIP_FLAGS[@]}" \
        accelerate diffusers transformers peft wandb absl-py ml-collections \
        numpy pillow imageio imageio-ffmpeg tqdm requests httpx flask addict \
        omegaconf einops ftfy sentencepiece protobuf scipy opencv-python \
        huggingface_hub

    # 0f. 3D reward 栈 + 可视化依赖
    #     gsplat: DA3 的 GS renderer 用 (需 CUDA 编译, 已装 gxx + nvcc)
    #     hpsv2: general reward server 用 (自动下权重)
    #     qwen-vl-utils: Qwen3-VL scorer 用
    #     pycolmap: DA3 可能用 (需 COLMAP 库, 装失败给警告)
    echo "📦 installing 3D reward + visualization deps"
    pip install "${PIP_FLAGS[@]}" \
        lpips trimesh plyfile moviepy gsplat evo e3nn hpsv2 qwen-vl-utils || \
        echo "  ⚠️ 部分 3D reward 依赖装失败 (上面有 WARNING), 继续装其余" >&2
    # pycolmap 单独装 (可能需要系统 COLMAP 库)
    pip install "${PIP_FLAGS[@]}" pycolmap 2>/dev/null || \
        echo "  ⚠️ pycolmap 装失败 (需要系统 COLMAP 库; DA3 可能不用它)" >&2

    # 0g. 可选加速包
    echo "📦 installing optional acceleration packages"
    pip install "${PIP_FLAGS[@]}" xformers bitsandbytes 2>/dev/null || \
        echo "  ⚠️ xformers/bitsandbytes 装失败 (可选, 不致命)" >&2

    echo "🎉 deps installed"
    echo ""
fi

# --- 1. World-R1 官方代码 ---
echo "🔍 [1/3] World-R1 官方代码: $WORLD_R1_DIR"
if [ ! -d "$WORLD_R1_DIR" ]; then
    echo "  ❌ [MISS] Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi
echo "  ✅ [OK]"

# --- 2. verify imports ---
echo "🔍 [2/3] verify imports"
python -c "import torch, diffusers, transformers, peft, flask, lpips; print('  ✅ env ok')" 2>/dev/null || \
    echo "  ❌ [MISS] imports failed — Run: INSTALL_DEPS=1 bash $0" >&2
python -c "import gsplat; print('  ✅ gsplat')" 2>/dev/null || \
    echo "  ⚠️ [MISS] gsplat (3D reward GS renderer 需要; INSTALL_DEPS=1 重装)" >&2
python -c "import hpsv2; print('  ✅ hpsv2')" 2>/dev/null || \
    echo "  ⚠️ [MISS] hpsv2 (general reward 需要; INSTALL_DEPS=1 重装)" >&2
python -c "import qwen_vl_utils; print('  ✅ qwen_vl_utils')" 2>/dev/null || \
    echo "  ⚠️ [MISS] qwen_vl_utils (Qwen3-VL scorer 需要; INSTALL_DEPS=1 重装)" >&2
python -c "from flow_grpo import train_diffusion; print('  ✅ flow_grpo')" 2>/dev/null || \
    echo "  ⚠️ [MISS] flow_grpo (训练需要; pip install -e $WORLD_R1_DIR)" >&2
python -c "from reward_server.reward_3d import MultiGPUReward3DManager; print('  ✅ reward_server')" 2>/dev/null || \
    echo "  ⚠️ [MISS] reward_server (reward server 需要; pip install -e $WORLD_R1_DIR)" >&2

# --- 3. model weights ---
echo "🔍 [3/3] 模型权重"
if [ -d "$WAN_MODEL_PATH" ]; then
    echo "  ✅ Wan2.1-T2V-14B-Diffusers: $WAN_MODEL_PATH"
else
    echo "  ❌ [MISS] Wan2.1-T2V-14B-Diffusers: $WAN_MODEL_PATH" >&2
    echo "     Run: bash $SCRIPT_DIR/01_download_models.sh" >&2
fi
_da3_cache="$HUGGINGFACE_HUB_CACHE/models--depth-anything--DA3-GIANT"
if [ -d "$_da3_cache" ]; then
    echo "  ✅ DA3-GIANT: $_da3_cache"
else
    echo "  ❌ [MISS] DA3-GIANT (3D reward 重建模型)" >&2
    echo "     Run: bash $SCRIPT_DIR/01_download_models.sh" >&2
fi
_qwen_cache="$HUGGINGFACE_HUB_CACHE/models--Qwen--Qwen3-VL-4B-Instruct"
if [ -d "$_qwen_cache" ]; then
    echo "  ✅ Qwen3-VL-4B-Instruct: $_qwen_cache"
else
    echo "  ❌ [MISS] Qwen3-VL-4B-Instruct (reward scorer)" >&2
    echo "     Run: bash $SCRIPT_DIR/01_download_models.sh" >&2
fi

echo ""
echo "🎉 [00] Done. 下一步: bash $SCRIPT_DIR/01_download_models.sh"
