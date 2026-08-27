#!/usr/bin/env bash
# 00a_setup_env.sh — WSL Ubuntu 24.04 variant of 00_setup_env.sh.
#
# Differences from 00_setup_env.sh (server):
# - Uses the Tsinghua PyPI mirror (default PyPI is slow from China).
# - Does NOT git clone 4DAnyone/GVHMR: WSL cannot reach github.com directly
#   (codeload.github.com works via curl --insecure). The user must pre-fetch
#   the source zips and assemble them at $FDANYONE_DIR before running this.
#   See README_wsl.md for the one-time fetch steps.
# - Inherits HF_ENDPOINT=https://hf-mirror.com from proxy.env (so step 01
#   downloads weights through the mirror without extra config).
# - No CUDA toolkit / nvcc: 4DAnyone has no CUDA extensions to compile
#   (kornia, ultralytics, smplx, pytorch-lightning, transformers, timm are
#   pure Python). DPVO (GVHMR's optional submodule) is NOT used — 4DAnyone
#   sets use_dpvo=false.
#
# Usage:
#   bash 4danyone/00a_setup_env.sh              # verify only (env exists)
#   INSTALL_DEPS=1 bash 4danyone/00a_setup_env.sh  # first time: build env + deps
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "=== [00a] Setup 4DAnyone env on WSL ==="
echo "  🤖 4DAnyone:  $FDANYONE_DIR"
echo "  🔗 GVHMR:     $GVHMR_DIR"
echo "  🏋️ weights:   $MODEL_DIR  (FDANYONE_MODEL_DIR override)"
echo "  💾 output:    $RESULTS_DIR"
echo "  🏠 conda env: $CONDA_ENV"
echo "  🌐 HF:        ${HF_ENDPOINT:-https://huggingface.co}"
echo ""

# ---------------------------------------------------------------------------
# 1. Verify the source code is already in place (WSL can't git clone github).
#    Fetch once via codeload.github.com + curl --insecure (see README_wsl.md).
# ---------------------------------------------------------------------------
echo "--- [1] verify source checkout ---"
if [ ! -f "$FDANYONE_DIR/inference.py" ]; then
    echo "❌ ERROR: 4DAnyone source missing at $FDANYONE_DIR" >&2
    echo "       WSL cannot git clone github.com directly. Fetch via codeload:" >&2
    echo "         curl.exe -sL --insecure --retry 3 --max-time 300 \\" >&2
    echo "           'https://codeload.github.com/ant-research/4DAnyone/zip/refs/heads/main' -o 4d.zip" >&2
    echo "         # extract -> C:\\code\\4DAnyone  (see README_wsl.md for full steps)" >&2
    exit 1
fi
if [ ! -f "$GVHMR_DIR/hmr4d/__init__.py" ]; then
    echo "❌ ERROR: GVHMR submodule missing at $GVHMR_DIR" >&2
    echo "       Fetch GVHMR zip from codeload.github.com/zju3dv/GVHMR and extract" >&2
    echo "       to $FDANYONE_DIR/third_party/GVHMR (see README_wsl.md)" >&2
    exit 1
fi
echo "  ✅ 4DAnyone source present (inference.py + GVHMR hmr4d/__init__.py)"

# ---------------------------------------------------------------------------
# 2. Create conda env from scratch (CPython 3.11; matches 4DAnyone install).
# ---------------------------------------------------------------------------
_new_prefix="$(conda info --base 2>/dev/null)/envs/$CONDA_ENV"
if [ -d "$_new_prefix/conda-meta" ] && [ -f "$_new_prefix/conda-meta/history" ]; then
    echo "--- conda env '$CONDA_ENV' already exists at $_new_prefix ---"
else
    echo "📦 creating conda env '$CONDA_ENV' (python=3.11, CPython)..."
    conda create -y -n "$CONDA_ENV" python=3.11 || {
        echo "❌ conda create failed" >&2
        exit 1
    }
    echo "  ✅ created"
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "  🐍 python: $(python --version 2>&1) at $(which python)"

# Verify CPython (gxx_linux-64 can swap it to GraalPy — not used here, but check).
_impl="$(python -c "import platform; print(platform.python_implementation())" 2>/dev/null)"
if [ "$_impl" != "CPython" ]; then
    echo "❌ ERROR: python is $_impl, not CPython. Recreate:" >&2
    echo "       conda env remove -n $CONDA_ENV && bash $0" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Install deps (first time or INSTALL_DEPS=1). Tsinghua mirror for speed.
# ---------------------------------------------------------------------------
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_MIRROR="${PIP_MIRROR:-https://mirrors.aliyun.com/pypi/simple}"
    # Local wheel cache: manual downloads land in D:\wheel (WSL: /mnt/d/wheel).
    # pip prefers these local wheels; falls back to the mirror for missing ones.
    # See download_urls.txt — torch/nvidia wheels are pre-fetched because the
    # PyPI CDN is rate-limited to ~8 KB/s on this network.
    WHEELS_DIR="${WHEELS_DIR:-/mnt/d/wheel}"
    PIP_FLAGS=(-i "$PIP_MIRROR" --find-links "$WHEELS_DIR" --timeout 600 --retries 10)
    echo "📦 PyPI mirror: $PIP_MIRROR  | local wheels: $WHEELS_DIR"
    if [ -d "$WHEELS_DIR" ]; then
        echo "  📁 found $(ls "$WHEELS_DIR"/*.whl 2>/dev/null | wc -l) local wheels"
    else
        echo "  ⚠️ $WHEELS_DIR not found — will download everything (slow). Pre-fetch wheels per download_urls.txt"
    fi

    # --- torch install is SPLIT into stages. pip with `-i INDEX --find-links`
    #     still downloads the 888 MB torch wheel from the index instead of using
    #     the local one (verified). So use `--no-index` to force local wheels for
    #     the big ones (torch/triton/torchvision/nvidia-*), and the aliyun mirror
    #     only for the small pure-Python deps that aren't in D:\wheel. ---

    # 3a. torch's small pure-Python deps (filelock/sympy/...; not in D:\wheel).
    if ! python -c "import torch" 2>/dev/null; then
        echo "📦 [3a] torch's small deps (filelock/sympy/networkx/jinja2/fsspec) from aliyun..."
        pip install "${PIP_FLAGS[@]}" filelock typing-extensions sympy networkx jinja2 fsspec || {
            echo "❌ small deps install failed" >&2; exit 1
        }
    fi

    # 3b. torch + triton from LOCAL wheels (--no-index; 14 nvidia-cu12 auto).
    if ! python -c "import torch" 2>/dev/null; then
        echo "📦 [3b] torch 2.8 + triton 3.4 from local wheels (--no-index, no download)..."
        pip install --no-index --find-links "$WHEELS_DIR" "torch==2.8.0" "triton==3.4.0" || {
            echo "❌ torch install failed" >&2
            echo "  Ensure torch-2.8.0-cp311 + nvidia_*.whl + triton are in $WHEELS_DIR" >&2
            exit 1
        }
    fi
    python -c "import torch; print(f'  ✅ torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}')"

    # 3c. numpy + pillow (torchvision deps, not in D:\wheel) from aliyun.
    if ! python -c "import torchvision" 2>/dev/null; then
        echo "📦 [3c] numpy + pillow from aliyun (torchvision deps)..."
        pip install "${PIP_FLAGS[@]}" "numpy>=1.26,<2" "pillow>=10,<13" || true
    fi

    # 3d. torchvision from LOCAL wheel (--no-index).
    if ! python -c "import torchvision" 2>/dev/null; then
        echo "📦 [3d] torchvision 0.23 from local wheel (--no-index)..."
        pip install --no-index --find-links "$WHEELS_DIR" "torchvision==0.23.0" || \
            echo "  ⚠️ torchvision local install failed — try: pip install ${PIP_FLAGS[@]} torchvision==0.23.0" >&2
    fi

    # 3e. 4DAnyone deps (filter torch/torchvision/numpy — handled above) from
    #     aliyun; D:\wheel wheels reused via --find-links.
    echo "📦 [3e] 4DAnyone deps from requirements.txt (aliyun + local wheels)..."
    if [ -f "$FDANYONE_DIR/requirements.txt" ]; then
        TMP_REQ="$(mktemp).txt"
        grep -v -iE '^[[:space:]]*(torch|torchvision|numpy)([=<>!~]|$|[[:space:]])' \
            "$FDANYONE_DIR/requirements.txt" > "$TMP_REQ"
        pip install "${PIP_FLAGS[@]}" -r "$TMP_REQ"
        rm -f "$TMP_REQ"
    else
        echo "  ⚠️ requirements.txt not found; installing known deps directly"
        pip install "${PIP_FLAGS[@]}" \
            "av>=16,<17" colorlog einops ffmpeg-python fire ftfy \
            "huggingface-hub>=0.36,<1" hydra-core hydra-zen imageio \
            kornia lapx opencv-python \
            "pytorch-lightning>=2.3,<3" regex safetensors scipy \
            sentencepiece "smplx==0.1.28" "timm==0.9.12" tqdm \
            "transformers>=4.57,<5" "ultralytics==8.2.42" yacs
    fi

    # 3f. numba — lapx's runtime dep (lapx's setup.py doesn't declare it).
    pip install "${PIP_FLAGS[@]}" numba || echo "  ⚠️ numba (lapx runtime dep) install failed" >&2

    # 3g. Pin numpy (1.26.x; <2 per requirements.txt, avoids 2.x ABI issues).
    pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps "numpy>=1.26,<2" || true
fi

# ---------------------------------------------------------------------------
# 4. conda init bash (so interactive shells have conda on PATH).
# ---------------------------------------------------------------------------
if ! grep -q "miniconda3" "$HOME/.bashrc" 2>/dev/null; then
    echo "📦 running 'conda init bash'..."
    conda init bash 2>/dev/null || true
    echo "  ✅ conda initialized in ~/.bashrc (run 'source ~/.bashrc' or restart shell)"
else
    echo "⏭️  conda already initialized in ~/.bashrc"
fi

# ---------------------------------------------------------------------------
# 5. Verify torch + key imports.
# ---------------------------------------------------------------------------
echo ""
echo "--- [verify] torch + imports ---"
python - <<'PY'
import sys
ok = True
try:
    import torch
    print(f"  ✅ torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  ⚠️  torch.cuda not available — check NVIDIA driver / WSL GPU passthrough.")
except Exception as e:
    print(f"  ❌ torch: {e}"); ok = False
for mod in ["numpy", "transformers", "pytorch_lightning", "ultralytics", "smplx", "kornia", "huggingface_hub"]:
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "ok")
        print(f"  ✅ {mod} {v}")
    except Exception as e:
        print(f"  ❌ {mod}: {e}"); ok = False
sys.exit(0 if ok else 1)
PY

echo ""
echo "🎉 [00a] Done. Env '$CONDA_ENV' ready on WSL."
echo "  ⚠️  Inference needs ~43 GiB VRAM — WSL's RTX 3090 (24 GiB) can't run it."
echo "      Use this env to download weights (step 01), then run step 02 on the server."
echo "  Next: bash $SCRIPT_DIR/01_download_models.sh  (HF via hf-mirror.com, + SMPL-X)"
