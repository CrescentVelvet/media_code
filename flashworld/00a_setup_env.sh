#!/usr/bin/env bash
# 00a_setup_env.sh — WSL Ubuntu 24.04 环境安装（FlashWorld 推理链路）。
#
# 做什么：
#   1. **clone 现成 env** `minimax_h3`（python 3.11 + torch 2.6.0+cu124 + triton 3.2.0，
#      与官方 requirements.txt 完全对得上）→ 新 env `flashworld`
#      —— 省掉 torch+nvidia 依赖 ~4 GB 下载；clone 源可用 BASE_ENV 覆盖
#   2. conda 装 cuda-nvcc 12.4 + gxx（**不需要 sudo**，全进 env 前缀）
#      —— gsplat 是源码编译（CUDAExtension），install 时就要 nvcc + g++
#   3. clone 官方仓 FlashWorld + 三个 pin 住的 git 依赖（gsplat / spz / diffusers）
#      —— 分开 clone 再本地 pip install，避免 `pip install git+https://` 反复拉
#   4. pip 装其余依赖（transformers 4.57.0 / numpy 2.2.6 严格对齐官方）
#   5. 编译 gsplat + spz（3090 → TORCH_CUDA_ARCH_LIST=8.6）
#   6. 把 FLASHWORLD_* 路径追加进 proxy.env（若尚未存在）
#
# 用法：
#   bash flashworld/00a_setup_env.sh
#   SKIP_ENV_CLONE=1 bash flashworld/00a_setup_env.sh    # 不复用 minimax_h3，从零建
#   BASE_ENV=4danyone bash flashworld/00a_setup_env.sh   # 换 clone 源（要求 torch>=2.5）
#   SKIP_BUILD=1 bash flashworld/00a_setup_env.sh        # 只装 python 依赖，不编译 gsplat
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

WHEELS_DIR="${WHEELS_DIR:-/mnt/d/wheel}"
BASE_ENV="${BASE_ENV:-minimax_h3}"
PIP_FLAGS=(-i https://mirrors.aliyun.com/pypi/simple --find-links "$WHEELS_DIR" --timeout 600 --retries 5)

# ── 1. conda ───────────────────────────────────────────────────────────────
CONDA_BASE="$HOME/miniconda3"
if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "❌ ERROR: miniconda3 not found at $CONDA_BASE" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ── 2. WSL 路径默认值（Linux fs：仓 ~/repos，权重 D 盘，输出 ~/output）─────
export FLASHWORLD_DIR="${FLASHWORLD_DIR:-$HOME/repos/FlashWorld}"
export FLASHWORLD_DEPS="${FLASHWORLD_DEPS:-$HOME/repos/_flashworld_deps}"
export FLASHWORLD_MODEL_DIR="${FLASHWORLD_MODEL_DIR:-/mnt/d/wheel/flashworld_ms}"
export FLASHWORLD_RESULTS_DIR="${FLASHWORLD_RESULTS_DIR:-$HOME/output/flashworld_results}"

# ── 3. 追加 FLASHWORLD_* 路径到 proxy.env（已有则不动）─────────────────────
PROXY_ENV="$REPO_DIR/proxy.env"
if ! grep -q "FLASHWORLD_MODEL_DIR" "$PROXY_ENV" 2>/dev/null; then
    echo "📦 appending FLASHWORLD_* path overrides -> $PROXY_ENV"
    cat >> "$PROXY_ENV" <<'ENVEOF'

# ── flashworld（FlashWorld 单图→3DGS；仅 flashworld/_env.sh 读取）──────────
# 官方仓 + 三个 pin 住的 git 依赖在 Linux fs（编译快）；权重在 D 盘（读一次即可）
export FLASHWORLD_DIR="$HOME/repos/FlashWorld"
export FLASHWORLD_DEPS="$HOME/repos/_flashworld_deps"
export FLASHWORLD_MODEL_DIR="/mnt/d/wheel/flashworld_ms"
export FLASHWORLD_RESULTS_DIR="${FLASHWORLD_RESULTS_DIR:-$HOME/output/flashworld_results}"
ENVEOF
    echo "  ✅ appended"
else
    echo "⏭️  proxy.env 已有 FLASHWORLD_* 配置: $PROXY_ENV"
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "=== [00a] Setup FlashWorld env on WSL ==="
echo "  🤖 官方仓:     $FLASHWORLD_DIR"
echo "  🧷 git 依赖:   $FLASHWORLD_DEPS"
echo "  🏋️ 权重根:     $MODEL_DIR"
echo "  💾 输出:       $RESULTS_DIR"
echo "  🏠 conda env:  $CONDA_ENV"
echo "  🔧 CUDA_HOME:  ${CUDA_HOME:-<unset>}  arch=${TORCH_CUDA_ARCH_LIST}"
echo ""

# ── 4. 建 conda env（优先 clone，省 4 GB torch 下载）───────────────────────
ENV_PREFIX="$(conda info --base)/envs/$CONDA_ENV"
if [ -d "$ENV_PREFIX/conda-meta" ]; then
    echo "--- conda env '$CONDA_ENV' already exists at $ENV_PREFIX ---"
else
    if [ "${SKIP_ENV_CLONE:-0}" != "1" ] && [ -d "$(conda info --base)/envs/$BASE_ENV/conda-meta" ]; then
        echo "--- cloning env '$BASE_ENV' -> '$CONDA_ENV' (复用 torch 2.6.0+cu124) ---"
        conda create -y -n "$CONDA_ENV" --clone "$BASE_ENV"
    else
        echo "--- creating fresh env '$CONDA_ENV' (python=3.11) ---"
        conda create -y -n "$CONDA_ENV" python=3.11
        # 官方 pin：torch 2.6.0 / torchvision 0.21.0 / triton 3.2.0（PyPI 默认 = cu124）
        conda run -n "$CONDA_ENV" pip install "${PIP_FLAGS[@]}" \
            torch==2.6.0 torchvision==0.21.0 triton==3.2.0
    fi
fi

# ── 5. 编译工具链：nvcc 12.4 + gxx（进 env 前缀，无需 sudo）────────────────
# ⚠️ 必须显式 pin python，否则 conda 可能把 CPython 换成 GraalPy（AGENTS.md 第 6 节）
if "$ENV_PREFIX/bin/nvcc" --version >/dev/null 2>&1; then
    echo "--- nvcc already in env ---"
else
    echo "📦 installing cuda-nvcc 12.4 + gxx into env"
    conda install -y -n "$CONDA_ENV" -c "nvidia/label/cuda-12.4.0" cuda-nvcc cuda-cudart-dev "python=3.11"
    conda install -y -n "$CONDA_ENV" -c conda-forge "gxx_linux-64=12" "python=3.11"
fi
if ! "$ENV_PREFIX/bin/python" -c "import platform,sys; sys.exit(0 if platform.python_implementation()=='CPython' else 1)"; then
    echo "❌ ERROR: python implementation 不是 CPython（conda 把 python 换成了 GraalPy？）" >&2
    exit 1
fi
echo "  ✅ python: $("$ENV_PREFIX/bin/python" -V 2>&1) / $("$ENV_PREFIX/bin/nvcc" --version | tail -1)"

# ── 6. clone 官方仓 + pin 住的 git 依赖 ────────────────────────────────────
# ⚠️ conda 的 libffi 会和系统 libp11-kit 冲突导致 git 崩 → 前置 LD_LIBRARY_PATH=
_clone() {
    local url="$1" dir="$2" name="$3"
    if [ -d "$dir/.git" ]; then
        echo "⏭️  $name already cloned: $dir"
        return 0
    fi
    echo "📦 cloning $name -> $dir"
    mkdir -p "$(dirname "$dir")"
    LD_LIBRARY_PATH= git clone --depth 1 "$url" "$dir" || {
        echo "❌ ERROR: clone $url failed" >&2
        return 1
    }
}

if [ "${SKIP_CLONE:-0}" != "1" ]; then
    _clone "$FLASHWORLD_REPO" "$FLASHWORLD_DIR" "FlashWorld" || exit 1

    mkdir -p "$FLASHWORLD_DEPS"
    # gsplat / spz 需要指定 commit；--depth 1 拿不到历史，故用 init+fetch 定点拉取
    _clone_pinned() {
        local url="$1" dir="$2" commit="$3" name="$4"
        if [ -d "$dir/.git" ]; then
            echo "⏭️  $name already cloned: $dir"
            return 0
        fi
        echo "📦 cloning $name @ ${commit:0:8}"
        mkdir -p "$dir"
        (
            cd "$dir" || exit 1
            LD_LIBRARY_PATH= git init -q . &&
            LD_LIBRARY_PATH= git remote add origin "$url" &&
            LD_LIBRARY_PATH= git fetch -q --depth 1 origin "$commit" &&
            LD_LIBRARY_PATH= git checkout -q FETCH_HEAD &&
            # ⚠️ gsplat 的 GLM 在 third_party/glm 子模块里；不 init 会在编译时报
            #    `glm/gtc/type_ptr.hpp: No such file or directory`（已踩）
            LD_LIBRARY_PATH= git submodule update --init --recursive
        ) || { echo "❌ ERROR: clone $name failed" >&2; return 1; }
    }

    _clone_pinned https://github.com/nerfstudio-project/gsplat.git \
        "$FLASHWORLD_DEPS/gsplat" 32f2a54d21c7ecb135320bb02b136b7407ae5712 "gsplat" || exit 1
    _clone_pinned https://github.com/nianticlabs/spz.git \
        "$FLASHWORLD_DEPS/spz" a4fc69e7948c7152e807e6501d73ddc9c149ce37 "spz" || exit 1
    _clone_pinned https://github.com/huggingface/diffusers.git \
        "$FLASHWORLD_DEPS/diffusers" 447e8322f76efea55d4769cd67c372edbf0715b8 "diffusers" || exit 1
fi

# ── 7. pip 依赖 ────────────────────────────────────────────────────────────
# 官方 requirements.txt 去掉三条 git 依赖（已本地 clone），其余照抄版本。
# transformers 必须 4.57.0：WanTransformer3DModel 在 5.x 上 API 变了（clone 源是 5.15）。
INSTALL_DEPS="${INSTALL_DEPS:-1}"
if [ "$INSTALL_DEPS" = "1" ]; then
    echo "📦 installing python deps"
    conda run -n "$CONDA_ENV" pip install "${PIP_FLAGS[@]}" \
        "transformers==4.57.0" "numpy==2.2.6" "omegaconf==2.3.0" "ninja==1.13.0" \
        "einops==0.8.1" "moviepy==1.0.3" "opencv-python==4.12.0.88" "av==15.1.0" \
        "plyfile==1.1.2" "ftfy==6.3.1" "gradio==5.49.1" "accelerate==1.10.1" \
        "nanobind==2.9.2" "uvicorn>=0.37" imageio tqdm jaxtyping rich pandas \
        scikit-build-core "cmake<4" \
        fastapi python-multipart "huggingface_hub==0.35.3"
    [ $? -ne 0 ] && { echo "❌ ERROR: pip deps failed" >&2; exit 1; }

    # ⚠️ clone 源自带的 torchao 0.16.0 与 torch 2.6 不兼容：diffusers 0.36.0.dev0 的
    #    torchao_quantizer.py 在 import 期调用 _update_torch_safe_globals()，
    #    torchao 缺 dtypes.uintx.uint4_layout 会走进一个引用了未定义 logger 的 except，
    #    变成 `NameError: name 'logger' is not defined`，整个 diffusers 挂掉。
    #    本项目不用 torchao → 直接卸掉，让 is_torchao_available() 走 False 分支。
    if conda run -n "$CONDA_ENV" python -c "import torchao" >/dev/null 2>&1; then
        echo "🧹 removing torchao (incompatible with torch 2.6 + diffusers 0.36.dev)"
        conda run -n "$CONDA_ENV" pip uninstall -y torchao >/dev/null 2>&1
    fi
fi

# ── 8. 编译 gsplat / spz（本地源码，不走网络）──────────────────────────────
if [ "${SKIP_BUILD:-0}" != "1" ]; then
    # 编译要在 Linux fs 上做（/mnt/d 是 9p，symlink 坏 + 慢 5-10x）
    if "$ENV_PREFIX/bin/python" -c "import gsplat" >/dev/null 2>&1; then
        echo "⏭️  gsplat already importable"
    else
        echo "🔨 building gsplat @32f2a54 (arch=$TORCH_CUDA_ARCH_LIST, 约 5-10 min)"
        ( cd "$FLASHWORLD_DEPS/gsplat" && \
          CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
          conda run -n "$CONDA_ENV" pip install --no-build-isolation --no-deps . ) || {
            echo "❌ ERROR: gsplat build failed" >&2; exit 1; }
    fi
    if "$ENV_PREFIX/bin/python" -c "import spz" >/dev/null 2>&1; then
        echo "⏭️  spz already importable"
    else
        echo "🔨 building spz @a4fc69e"
        ( cd "$FLASHWORLD_DEPS/spz" && \
          conda run -n "$CONDA_ENV" pip install --no-build-isolation . ) || {
            echo "❌ ERROR: spz build failed" >&2; exit 1; }
    fi
    if "$ENV_PREFIX/bin/python" -c "import diffusers" >/dev/null 2>&1; then
        echo "⏭️  diffusers already importable"
    else
        echo "🔨 installing diffusers @447e832 (wan-5Bi2v)"
        ( cd "$FLASHWORLD_DEPS/diffusers" && \
          conda run -n "$CONDA_ENV" pip install --no-build-isolation --no-deps -e . ) || {
            echo "❌ ERROR: diffusers install failed" >&2; exit 1; }
    fi
fi

# ── 9. 验证 ────────────────────────────────────────────────────────────────
echo ""
echo "=== verify ==="
conda run -n "$CONDA_ENV" python - <<'PY'
import importlib, sys
ok = True
for mod in ("torch", "triton", "transformers", "diffusers", "gsplat", "spz", "gradio", "av"):
    try:
        m = importlib.import_module(mod)
        print(f"  ✅ {mod:14s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        ok = False
        print(f"  ❌ {mod:14s} {type(e).__name__}: {e}")
import torch
print(f"  🎮 cuda available: {torch.cuda.is_available()}  "
      f"capability={torch.cuda.get_device_capability() if torch.cuda.is_available() else '-'}")
sys.exit(0 if ok else 1)
PY
if [ $? -ne 0 ]; then
    echo "❌ ERROR: verification failed" >&2
    exit 1
fi

# ── 10. 依赖补丁（幂等）─────────────────────────────────────────────────────
# 离线模式下 from_pretrained 加载分片权重会去调 model_info() 联网，必须打补丁。
# 详见 00b_patch_deps.py 顶部注释。
echo ""
echo "🩹 [00b] 给 vendored diffusers 打离线补丁"
conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/00b_patch_deps.py" || {
    echo "⚠️ WARNING: 00b 补丁未应用，离线推理会在 from_pretrained 处报 OfflineModeIsEnabled" >&2
}
echo ""
echo "🎉 Done. 下一步：bash flashworld/01_download_models.sh"
