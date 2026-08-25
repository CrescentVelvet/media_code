#!/usr/bin/env bash
# 00_setup_env.sh — create the `4danyone` conda env (CPython 3.11), clone the
# 4DAnyone repo + GVHMR submodule, and install the Python deps.
#
# 4DAnyone (ant-research) turns a monocular video into synchronized multi-view
# videos for downstream 4DGS reconstruction. It depends on GVHMR (zju3dv) as a
# git submodule under third_party/GVHMR.
#
# ⚠️  Inference needs ~43 GiB peak VRAM (6-view minimum). This setup runs on
#    any machine, but inference itself MUST run on a >=48 GiB GPU on the server.
#
# First time:
#   INSTALL_DEPS=1 bash 4danyone/00_setup_env.sh
# After that (verify only):
#   bash 4danyone/00_setup_env.sh
#
# Env:
#   CONDA_ENV=4danyone   (dedicated env; python=3.11, torch 2.8 from PyPI)
#
# Notes:
# - torch>=2.8,<2.9 is installed from PyPI (the PyPI wheel ships cu12x by
#   default, so no download.pytorch.org index is needed — the corporate proxy
#   blocks that host anyway).
# - No CUDA extensions need compiling; all deps (kornia, ultralytics, smplx,
#   pytorch-lightning, transformers, timm) are pure Python.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "=== [00] Setup 4DAnyone env ==="
echo "  🤖 4DAnyone:  $FDANYONE_DIR"
echo "  🔗 GVHMR:     $GVHMR_DIR"
echo "  🏋️ weights:   $MODEL_DIR"
echo "  💾 output:    $RESULTS_DIR"
echo "  🏠 conda env: $CONDA_ENV"
echo ""

# ---------------------------------------------------------------------------
# 1. Create conda env if missing (CPython 3.11 — matches 4DAnyone's install).
# ---------------------------------------------------------------------------
if ! conda env list 2>/dev/null | grep -qw "$CONDA_ENV"; then
    echo "📦 creating conda env '$CONDA_ENV' (python=3.11, CPython)..."
    conda create -n "$CONDA_ENV" python=3.11 -y || {
        echo "❌ conda create failed" >&2
        exit 1
    }
    conda activate "$CONDA_ENV"
    impl="$(python -c 'import platform; print(platform.python_implementation())')"
    if [ "$impl" != "CPython" ]; then
        echo "❌ ERROR: python implementation is $impl (should be CPython)." >&2
        echo "       conda-forge may have substituted GraalPy. Recreate:" >&2
        echo "         conda env remove -n $CONDA_ENV && conda create -n $CONDA_ENV python=3.11 -y --override-channels -c defaults" >&2
        exit 1
    fi
    echo "  ✅ created: python=$(python --version 2>&1 | cut -d' ' -f2) ($impl)"
else
    conda activate "$CONDA_ENV" 2>/dev/null || true
    echo "⏭️  conda env '$CONDA_ENV' already exists"
fi
echo "  🐍 python: $(python --version 2>&1) at $(which python)"

# ---------------------------------------------------------------------------
# 2. Clone 4DAnyone official repo + init GVHMR submodule.
# ---------------------------------------------------------------------------
FDANYONE_REPO="${FDANYONE_REPO:-https://github.com/ant-research/4DAnyone.git}"

if [ ! -d "$FDANYONE_DIR/.git" ]; then
    echo "📦 cloning 4DAnyone -> $FDANYONE_DIR"
    mkdir -p "$(dirname "$FDANYONE_DIR")"
    LD_LIBRARY_PATH= git clone "$FDANYONE_REPO" "$FDANYONE_DIR" || \
        LD_LIBRARY_PATH= git -c http.sslVerify=false clone "$FDANYONE_REPO" "$FDANYONE_DIR"
else
    echo "⏭️  4DAnyone repo present: $FDANYONE_DIR"
fi

# GVHMR submodule (zju3dv/GVHMR) — required for motion recovery.
if [ ! -f "$GVHMR_DIR/hmr4d/__init__.py" ]; then
    echo "📦 initializing GVHMR submodule (third_party/GVHMR)"
    ( cd "$FDANYONE_DIR" && git submodule update --init third_party/GVHMR ) || \
        ( cd "$FDANYONE_DIR" && git -c http.sslVerify=false submodule update --init third_party/GVHMR )
else
    echo "⏭️  GVHMR submodule present: $GVHMR_DIR"
fi
# Sanity: GVHMR must be checked out (not an empty dir).
if [ ! -f "$GVHMR_DIR/hmr4d/__init__.py" ]; then
    echo "❌ ERROR: GVHMR not initialized at $GVHMR_DIR" >&2
    echo "       Run inside $FDANYONE_DIR:" >&2
    echo "         git submodule update --init third_party/GVHMR" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Install Python deps (first time or when INSTALL_DEPS=1).
# ---------------------------------------------------------------------------
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)
    _proxy="${https_proxy:-${http_proxy:-}}"
    if [ -n "$_proxy" ]; then
        PIP_FLAGS+=(--proxy "$_proxy")
        echo "  using proxy: $_proxy"
    fi

    # 3a. torch 2.8 from PyPI (PyPI wheel ships cu12x; no pytorch.org index).
    #     requirements.txt pins torch>=2.8,<2.9 and torchvision>=0.23,<0.24.
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "📦 installing torch 2.8 + torchvision 0.23 (from PyPI, cu12x)..."
        pip install "${PIP_FLAGS[@]}" "torch>=2.8,<2.9" "torchvision>=0.23,<0.24" || {
            echo "❌ torch install failed" >&2
            echo "  If PyPI is blocked, download wheels manually to \$MODEL_DIR and:" >&2
            echo "    pip install --force-reinstall --no-deps <wheel>" >&2
            exit 1
        }
    fi
    python -c "import torch; print(f'  ✅ torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}')"

    # 3b. 4DAnyone deps (filter out torch/torchvision/numpy pins — handled above).
    echo "📦 installing 4DAnyone deps (from requirements.txt, torch/numpy filtered)..."
    if [ -f "$FDANYONE_DIR/requirements.txt" ]; then
        TMP_REQ="$(mktemp).txt"
        grep -v -iE '^[[:space:]]*(torch|torchvision|numpy)([=<>!~]|$|[[:space:]])' \
            "$FDANYONE_DIR/requirements.txt" > "$TMP_REQ"
        pip install "${PIP_FLAGS[@]}" -r "$TMP_REQ"
        rm -f "$TMP_REQ"
    else
        echo "  ⚠️ requirements.txt not found at $FDANYONE_DIR; installing known deps directly"
        pip install "${PIP_FLAGS[@]}" \
            "av>=16,<17" colorlog einops ffmpeg-python fire ftfy \
            "huggingface-hub>=0.36,<1" hydra-core hydra-zen imageio \
            kornia lapx "numpy>=1.26,<2" opencv-python pillow \
            "pytorch-lightning>=2.3,<3" regex safetensors scipy \
            sentencepiece "smplx==0.1.28" "timm==0.9.12" tqdm \
            "transformers>=4.57,<5" typing-extensions "ultralytics==8.2.42" yacs
    fi

    # 3c. Pin numpy (1.26.x is the sweet spot; <2 per requirements.txt).
    pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps "numpy>=1.26,<2" || true

    # 3d. Optional: FlashAttention-3 / SageAttention for faster denoising.
    #     Only on Hopper (H100/H200); skip on Ampere (RTX 30xx/A100-40GB).
    #     Left commented — install manually if you have a Hopper card:
    #   pip install flash-attn --no-build-isolation
    #   pip install sageattention
    :
fi

# ---------------------------------------------------------------------------
# 4. Verify torch + key imports.
# ---------------------------------------------------------------------------
echo ""
echo "--- [verify] torch ---"
python - <<'PY'
import sys
ok = True
try:
    import torch
    print(f"  ✅ torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  ⚠️  torch.cuda not available — check GPU visibility / driver.")
except Exception as e:
    print(f"  ❌ torch: {e}"); ok = False
try:
    import numpy; print(f"  ✅ numpy {numpy.__version__}")
except Exception as e:
    print(f"  ❌ numpy: {e}"); ok = False
try:
    import transformers; print(f"  ✅ transformers {transformers.__version__}")
except Exception as e:
    print(f"  ❌ transformers: {e}"); ok = False
try:
    import pytorch_lightning; print(f"  ✅ pytorch_lightning {pytorch_lightning.__version__}")
except Exception as e:
    print(f"  ❌ pytorch_lightning: {e}"); ok = False
try:
    import ultralytics; print(f"  ✅ ultralytics {ultralytics.__version__}")
except Exception as e:
    print(f"  ❌ ultralytics: {e}"); ok = False
try:
    import smplx; print(f"  ✅ smplx")
except Exception as e:
    print(f"  ❌ smplx: {e}"); ok = False
try:
    import kornia; print(f"  ✅ kornia {kornia.__version__}")
except Exception as e:
    print(f"  ❌ kornia: {e}"); ok = False
sys.exit(0 if ok else 1)
PY

# ---------------------------------------------------------------------------
# 5. Verify repo + submodule layout.
# ---------------------------------------------------------------------------
echo "--- [verify] repo layout ---"
[ -f "$FDANYONE_DIR/inference.py" ] && echo "  ✅ 4DAnyone inference.py" || echo "  [MISS] 4DAnyone inference.py: $FDANYONE_DIR"
[ -f "$GVHMR_DIR/hmr4d/__init__.py" ] && echo "  ✅ GVHMR submodule" || echo "  [MISS] GVHMR submodule: $GVHMR_DIR"

echo ""
echo "🎉 [00] Done. Env '$CONDA_ENV' ready."
echo "    Next: bash $SCRIPT_DIR/01_download_models.sh  (download HF ckpts + SMPL-X)"
echo "    ⚠️  Inference needs ~43 GiB VRAM — run 02 on a >=48 GiB GPU on the server."
