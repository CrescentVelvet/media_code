#!/usr/bin/env bash
# 00_setup_env.sh — activate conda env & install SGLang (with diffusion support).
#
# MiniMax-H3 本地复现走 SGLang Diffusion 路径（官方推荐的三种推理框架之一）。
# SGLang 自带 MiniMax-H3 的模型实现 + 扩散运行时，所以只需要：
#   1) 一个带 CUDA torch 的 conda env；
#   2) `pip install "sglang[all]"`（含 diffusion 支持，近版本已把 MiniMax-H3
#      的 transformer/vae/scheduler 打包进去）。
#
# Set INSTALL_DEPS=1 to install SGLang (first time).
#
# SGLang pulls its own torch + flashinfer + CUDA kernels; its version pins may
# CONFLICT with other algos in this repo (hunyuanvideo wants diffusers 0.35,
# hypir pins diffusers 0.32 / transformers 4.49). Use a DEDICATED env:
#   conda create -n minimax_h3 --clone doll -y   # 克隆现有 doll，本地复制不走 conda 通道，绕开 TUNA 镜像 403 + 代理 SSL
#   conda activate minimax_h3
#   CONDA_ENV=minimax_h3 INSTALL_DEPS=1 bash minimax_h3/00_setup_env.sh
# doll 不在时新建见 README「可能遇到的问题」第 2 条（--override-channels + conda config ssl_verify）。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [00] Verify torch in conda env '$CONDA_ENV'"
python - <<'PY'
import sys
try:
    import torch
    print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: torch.cuda not available — install a CUDA-enabled torch or check GPU visibility.")
except ModuleNotFoundError:
    raise SystemExit("ERROR: torch not installed in this env. Install with: pip install torch --index-url https://download.pytorch.org/whl/cu124")
print(f"python: {sys.version.split()[0]}")
PY

# Install SGLang on demand. sglang[all] bundles the SRT serving runtime + the
# diffusion extras (diffusers/transformers/einops...) that MiniMax-H3 needs.
# If your sglang build splits diffusion into a separate extra and [all] didn't
# bring it, additionally run: pip install "sglang[diffusion]"
# PIP_FLAGS 提到块外，自检段（git fallback）也能用。
PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org --timeout 600 --retries 10)
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    echo "📦 installing SGLang (pip install -U \"sglang[all]\") ---"
    python -m pip install --upgrade pip "${PIP_FLAGS[@]}"
    # -U 强制升级：SGLang Diffusion（--model-variant / --performance-mode 等参数）
    # 是 2025/11 后新增，旧版 sglang 不认识这些参数会报 unrecognized arguments。
    python -m pip install "${PIP_FLAGS[@]}" -U "sglang[all]"
    # Diffusion extra is explicit in the sglang docker image
    # (python -m pip install -e ...python[diffusion]); install it too as a
    # no-op-ish safety net in case [all] didn't pin the diffusion deps.
    python -m pip install "${PIP_FLAGS[@]}" -U "sglang[diffusion]" || \
        echo "    (sglang[diffusion] extra not separately installable — assume [all] covers it)"
    # sglang diffusion 链路 diffusers→peft→transformers 三者版本须一致。克隆 doll
    # 这类已有 env 时，pip 可能不升级里面旧的 transformers（满足 sglang 下界就跳过），
    # 但新 peft 要 transformers>=4.42 的 HybridCache，于是 import sglang 报
    # `cannot import name 'HybridCache' from 'transformers'`。显式 -U 让 pip 在
    # sglang 声明的约束内重解到一致版本组（pip 新 resolver 会尊重 sglang 的上界）。
    echo "📦 aligning diffusers/peft/transformers (fix HybridCache mismatch from cloned-stale deps) ---"
    python -m pip install "${PIP_FLAGS[@]}" -U diffusers peft transformers
    # 自检 HybridCache（peft 新版要 transformers>=4.42）；pip -U 受 sglang 上界
    # 约束可能没升到 4.42，自检失败则带版本下界强制升级，再不行用 --no-deps 绕开 resolver。
    if ! python -c "from transformers import HybridCache" 2>/dev/null; then
        echo "📦 HybridCache still missing — force-upgrading 'transformers>=4.42' ---"
        python -m pip install "${PIP_FLAGS[@]}" -U "transformers>=4.42"
        if ! python -c "from transformers import HybridCache" 2>/dev/null; then
            echo "📦 still missing — retry with --no-deps (bypass resolver upper-bound) ---"
            python -m pip install "${PIP_FLAGS[@]}" -U --no-deps "transformers>=4.42"
        fi
    fi
    echo "📦 installed. Verify with: python -c 'import sglang; from transformers import HybridCache; print(\"ok\")'"
fi

echo "🔍 [00] Checking SGLang availability ==="
if python -c "import sglang; from transformers import HybridCache" 2>/dev/null; then
    python - <<'PY'
import sglang
from transformers import HybridCache
print(f"sglang: {getattr(sglang, '__version__', 'unknown')}  HybridCache: ok")
PY
else
    echo "⚠️ WARNING: sglang/HybridCache not importable in env '$CONDA_ENV'. Run INSTALL_DEPS=1 bash minimax_h3/00_setup_env.sh." >&2
fi

# 自检 sglang serve 是否支持 MiniMax-H3 的 diffusion 参数（--model-variant / --performance-mode）。
# SGLang Diffusion（2025/11 后新增）的 --model-variant 等参数，PyPI 的 sglang[all]
# 可能不带（cookbook 的 docker 命令是从源码 pip install -e ".../python[diffusion]"）。
# 自检失败则从 git clone sglang 源码 + editable 安装 [diffusion] extra。
echo "🔍 [00] Checking SGLang Diffusion args (--model-variant) ==="
if sglang serve --help 2>&1 | grep -q -- '--model-variant'; then
    echo "✅ sglang serve supports --model-variant (Diffusion ok)"
else
    echo "⚠️ PyPI sglang 不带 --model-variant — 从 git 源码装 [diffusion] extra (cookbook 做法) ---" >&2
    SGLANG_SRC="${SGLANG_SRC:-/tmp/sglang-src}"
    if [ ! -d "$SGLANG_SRC/python" ]; then
        echo "📦 cloning sglang repo -> $SGLANG_SRC"
        LD_LIBRARY_PATH= git clone --depth 1 https://github.com/sgl-project/sglang.git "$SGLANG_SRC" || \
            LD_LIBRARY_PATH= git -c http.sslVerify=false clone --depth 1 https://github.com/sgl-project/sglang.git "$SGLANG_SRC"
    fi
    if [ -d "$SGLANG_SRC/python" ]; then
        echo "📦 pip install -e $SGLANG_SRC/python[diffusion] --no-deps (跳过 torch 2.13/cuda 13 pin + Rust 扩展) ---"
        # git main 的 pyproject.toml pin torch==2.13.0 / cuda-python>=13.0（CUDA 13），
        # A100 是 CUDA 12.4 装不上 cuda 13 wheel。--no-deps 只装 sglang 本体
        # （带 --model-variant / --performance-mode 等 diffusion serving 参数），
        # torch/flashinfer 用 env 现有版本；[diffusion] extra 依赖单独装。
        # SGLANG_BUILD_RUST_EXTS=none 跳过 Rust 扩展（cargo 不在时；运行时报错再装 rustup）。
        SGLANG_BUILD_RUST_EXTS=none python -m pip install "${PIP_FLAGS[@]}" -e "$SGLANG_SRC/python[diffusion]" --no-deps || \
            SGLANG_BUILD_RUST_EXTS=none python -m pip install "${PIP_FLAGS[@]}" -e "$SGLANG_SRC/python" --no-deps
        # --no-deps 跳过了 [diffusion] extra 的依赖，单独对齐
        python -m pip install "${PIP_FLAGS[@]}" -U diffusers peft transformers
    else
        echo "❌ ERROR: clone sglang repo failed. Manual: LD_LIBRARY_PATH= git clone https://github.com/sgl-project/sglang.git $SGLANG_SRC && SGLANG_BUILD_RUST_EXTS=none pip install -e \"$SGLANG_SRC/python[diffusion]\" --no-deps" >&2
    fi
    # 再自检
    if sglang serve --help 2>&1 | grep -q -- '--model-variant'; then
        echo "✅ sglang serve now supports --model-variant (Diffusion ok after git install)"
    else
        echo "❌ ERROR: sglang serve still does NOT recognize --model-variant." >&2
        echo "   Manual fallback: pip install -e \"$SGLANG_SRC/python[diffusion]\" then check deps." >&2
    fi
fi

echo "🎉 [00] Done. Env '$CONDA_ENV' ready. (Missing sglang? INSTALL_DEPS=1 bash this)"
