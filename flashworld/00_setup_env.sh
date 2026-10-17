#!/usr/bin/env bash
# 00_setup_env.sh — 服务器环境安装（FlashWorld 推理链路）。
#
# 与 00a_setup_env.sh 的差异：服务器上 conda 在 PATH、有系统 CUDA toolkit 和 gcc，
# 所以这里直接建 env + 装依赖，不 clone 现成 env、不额外装 nvcc/gxx。
#
# 用法：
#   INSTALL_DEPS=1 bash flashworld/00_setup_env.sh
#   SKIP_BUILD=1 bash flashworld/00_setup_env.sh     # 只 clone + 装 python 依赖
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "=== [00] Setup FlashWorld env (server) ==="
echo "  🤖 官方仓:    $FLASHWORLD_DIR"
echo "  🧷 git 依赖:  $FLASHWORLD_DEPS"
echo "  🏋️ 权重根:    $MODEL_DIR"
echo "  💾 输出:      $RESULTS_DIR"
echo "  🏠 conda env: $CONDA_ENV"
echo ""

# ── 1. clone 官方仓 + 三个 pin 住的 git 依赖 ───────────────────────────────
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
        # gsplat 的 GLM 在 third_party/glm 子模块；不 init 编译会报找不到 glm/gtc/type_ptr.hpp
        LD_LIBRARY_PATH= git submodule update --init --recursive
    ) || { echo "❌ ERROR: clone $name failed" >&2; return 1; }
}

if [ "${SKIP_CLONE:-0}" != "1" ]; then
    if [ -d "$FLASHWORLD_DIR/.git" ]; then
        echo "⏭️  FlashWorld already cloned: $FLASHWORLD_DIR"
    else
        echo "📦 cloning FlashWorld -> $FLASHWORLD_DIR"
        mkdir -p "$(dirname "$FLASHWORLD_DIR")"
        LD_LIBRARY_PATH= git clone --depth 1 "$FLASHWORLD_REPO" "$FLASHWORLD_DIR" || {
            echo "❌ ERROR: clone FlashWorld failed" >&2; exit 1; }
    fi
    mkdir -p "$FLASHWORLD_DEPS"
    _clone_pinned https://github.com/nerfstudio-project/gsplat.git \
        "$FLASHWORLD_DEPS/gsplat" 32f2a54d21c7ecb135320bb02b136b7407ae5712 "gsplat" || exit 1
    _clone_pinned https://github.com/nianticlabs/spz.git \
        "$FLASHWORLD_DEPS/spz" a4fc69e7948c7152e807e6501d73ddc9c149ce37 "spz" || exit 1
    _clone_pinned https://github.com/huggingface/diffusers.git \
        "$FLASHWORLD_DEPS/diffusers" 447e8322f76efea55d4769cd67c372edbf0715b8 "diffusers" || exit 1
fi

# ── 2. 建 env ──────────────────────────────────────────────────────────────
ENV_PREFIX="$(conda info --base)/envs/$CONDA_ENV"
if [ -d "$ENV_PREFIX/conda-meta" ]; then
    echo "--- conda env '$CONDA_ENV' already exists ---"
else
    # 官方 requirements 是 python 3.11 + torch 2.6.0（cu124）
    conda create -y -n "$CONDA_ENV" python=3.11
fi

# ── 3. python 依赖 ─────────────────────────────────────────────────────────
# ⚠️ transformers 必须 4.57.0：WanTransformer3DModel 在 5.x 上 API 变了。
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    echo "📦 installing python deps"
    conda run -n "$CONDA_ENV" pip install -i https://mirrors.aliyun.com/pypi/simple --timeout 600 \
        "torch==2.6.0" "torchvision==0.21.0" "triton==3.2.0" \
        "transformers==4.57.0" "numpy==2.2.6" "omegaconf==2.3.0" "ninja==1.13.0" \
        "einops==0.8.1" "moviepy==1.0.3" "opencv-python==4.12.0.88" "av==15.1.0" \
        "plyfile==1.1.2" "ftfy==6.3.1" "gradio==5.49.1" "accelerate==1.10.1" \
        "nanobind==2.9.2" "uvicorn>=0.37" imageio tqdm jaxtyping rich pandas fastapi \
        scikit-build-core "cmake<4" \
        "huggingface_hub==0.35.3" || { echo "❌ ERROR: pip deps failed" >&2; exit 1; }
fi

# ── 4. 编译 gsplat / spz / diffusers ───────────────────────────────────────
if [ "${SKIP_BUILD:-0}" != "1" ]; then
    if conda run -n "$CONDA_ENV" python -c "import gsplat" >/dev/null 2>&1; then
        echo "⏭️  gsplat already importable"
    else
        echo "🔨 building gsplat @32f2a54 (arch=$TORCH_CUDA_ARCH_LIST)"
        ( cd "$FLASHWORLD_DEPS/gsplat" && \
          conda run -n "$CONDA_ENV" pip install --no-build-isolation --no-deps . ) || {
            echo "❌ ERROR: gsplat build failed" >&2; exit 1; }
    fi
    if conda run -n "$CONDA_ENV" python -c "import spz" >/dev/null 2>&1; then
        echo "⏭️  spz already importable"
    else
        echo "🔨 building spz @a4fc69e"
        ( cd "$FLASHWORLD_DEPS/spz" && \
          conda run -n "$CONDA_ENV" pip install --no-build-isolation . ) || {
            echo "❌ ERROR: spz build failed" >&2; exit 1; }
    fi
    if conda run -n "$CONDA_ENV" python -c "import diffusers" >/dev/null 2>&1; then
        echo "⏭️  diffusers already importable"
    else
        ( cd "$FLASHWORLD_DEPS/diffusers" && \
          conda run -n "$CONDA_ENV" pip install --no-build-isolation --no-deps -e . )
    fi
fi

echo ""
echo "=== verify ==="
conda run -n "$CONDA_ENV" python - <<'PY'
import importlib, sys
ok = True
for mod in ("torch", "triton", "transformers", "diffusers", "gsplat", "spz"):
    try:
        m = importlib.import_module(mod)
        print(f"  ✅ {mod:14s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        ok = False
        print(f"  ❌ {mod:14s} {type(e).__name__}: {e}")
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && { echo "❌ ERROR: verification failed" >&2; exit 1; }

echo ""
echo "🎉 Done. 下一步：bash flashworld/01_download_models.sh"
