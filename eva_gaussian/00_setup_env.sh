#!/usr/bin/env bash
# 00_setup_env.sh — clone EVA-Gaussian, create conda env (CPython 3.10),
# install torch + deps, build the feature-splatting CUDA rasterizer, and verify.
#
# First time:
#   INSTALL_DEPS=1 BUILD_CUDA=1 bash eva_gaussian/00_setup_env.sh
# After that (verify only):
#   bash eva_gaussian/00_setup_env.sh
#
# Torch: official pins 2.5.0+cu118. Company proxy blocks download.pytorch.org,
# so we try LOCAL cu118 wheels at $MODEL_DIR first. If no local cu118 wheels,
# falls back to pip (may 403). To reuse the wan22_rotate/pi3_3dgs cu124 wheels:
#   CUDA_TAG=cu124 TORCH_VERSION=2.6.0 TORCHVISION_VERSION=0.21.0 \
#     INSTALL_DEPS=1 BUILD_CUDA=1 bash eva_gaussian/00_setup_env.sh
#
# Env:
#   CONDA_ENV=eva_gaussian        (dedicated env)
#   CUDA_TAG=cu118                (torch CUDA build tag; cu124 to reuse wheels)
#   TORCH_VERSION=2.5.0           (torch version; 2.6.0 for cu124)
#   TORCHVISION_VERSION=0.20.0    (matching torchvision)
#   CUDA_HOME=/usr/local/cuda     (toolkit root for nvcc; must match cu major)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

CUDA_TAG="${CUDA_TAG:-cu118}"
TORCH_VERSION="${TORCH_VERSION:-2.5.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.0}"

# Map torch version -> triton version (requirements.txt pins 3.1.0 for torch 2.5).
case "$TORCH_VERSION" in
    2.5.*) TRITON_VERSION="${TRITON_VERSION:-3.1.0}" ;;
    2.6.*) TRITON_VERSION="${TRITON_VERSION:-3.2.0}" ;;
    *)     TRITON_VERSION="${TRITON_VERSION:-3.1.0}" ;;
esac

# ---------------------------------------------------------------------------
# 0. Create conda env if missing (CPython 3.10).
# ---------------------------------------------------------------------------
if ! conda env list 2>/dev/null | grep -qw "$CONDA_ENV"; then
    echo "📦 [00] conda env '$CONDA_ENV' not found; creating python=3.10 (CPython)"
    conda create -n "$CONDA_ENV" python=3.10 -y
    conda activate "$CONDA_ENV"
    impl="$(python -c 'import platform; print(platform.python_implementation())')"
    if [ "$impl" != "CPython" ]; then
        echo "❌ ERROR: env '$CONDA_ENV' python is $impl (should be CPython)." >&2
        echo "       conda-forge may have substituted GraalPy. Recreate:" >&2
        echo "         conda env remove -n $CONDA_ENV && conda create -n $CONDA_ENV python=3.10 -y --override-channels -c defaults" >&2
        exit 1
    fi
    echo "  ✅ python=$(python --version 2>&1 | cut -d' ' -f2) ($impl)"
else
    echo "📦 [00] conda env '$CONDA_ENV' present"
fi

# ---------------------------------------------------------------------------
# 1. Clone EVA-Gaussian official repo (sibling of media_code).
# ---------------------------------------------------------------------------
EVA_REPO="${EVA_REPO:-https://github.com/zhenliuZJU/EVA-Gaussian.git}"

if [ ! -d "$EVA_DIR/.git" ]; then
    echo "📦 [00] cloning EVA-Gaussian -> $EVA_DIR"
    mkdir -p "$(dirname "$EVA_DIR")"
    LD_LIBRARY_PATH= git clone "$EVA_REPO" "$EVA_DIR" || \
        LD_LIBRARY_PATH= git -c http.sslVerify=false clone "$EVA_REPO" "$EVA_DIR"
else
    echo "📦 [00] EVA-Gaussian repo present: $EVA_DIR"
fi

# ---------------------------------------------------------------------------
# 2. Install deps (first time or when INSTALL_DEPS=1).
# ---------------------------------------------------------------------------
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org \
        --timeout 600 --retries 10)
    _proxy="${https_proxy:-${http_proxy:-}}"
    if [ -n "$_proxy" ]; then
        PIP_FLAGS+=(--proxy "$_proxy")
        echo "  using proxy: $_proxy"
    else
        echo "⚠️ WARNING: no proxy set; pip may fail if no direct internet." >&2
    fi

    # 2a. Install PyTorch — try local wheels at $MODEL_DIR first (company proxy
    #     blocks download.pytorch.org). Falls back to pip if no local wheels.
    echo "🏋️ [00] installing PyTorch ${TORCH_VERSION}+${CUDA_TAG}"
    LOCAL_WHEELS=()
    for w in \
        "$MODEL_DIR/torch-${TORCH_VERSION}+${CUDA_TAG}-cp310-cp310-linux_x86_64.whl" \
        "$MODEL_DIR/torch-${TORCH_VERSION}%2B${CUDA_TAG}-cp310-cp310-linux_x86_64.whl" \
        "$MODEL_DIR/torchvision-${TORCHVISION_VERSION}+${CUDA_TAG}-cp310-cp310-linux_x86_64.whl" \
        "$MODEL_DIR/torchvision-${TORCHVISION_VERSION}%2B${CUDA_TAG}-cp310-cp310-linux_x86_64.whl" ; do
        [ -f "$w" ] && LOCAL_WHEELS+=("$w")
    done
    if [ ${#LOCAL_WHEELS[@]} -ge 1 ]; then
        echo "  installing local wheels: ${LOCAL_WHEELS[*]}"
        pip install --force-reinstall --no-deps "${LOCAL_WHEELS[@]}"
    else
        echo "  ⚠️ no local ${CUDA_TAG} wheels found at $MODEL_DIR"
        echo "     falling back to pip (download.pytorch.org may 403 behind proxy)"
        pip install "${PIP_FLAGS[@]}" \
            "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
            --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
    fi

    # 2b. Install EVA-Gaussian Python deps (filter torch/numpy — installed separately).
    echo "📦 [00] installing EVA-Gaussian Python deps (from requirements.txt)"
    if [ -f "$EVA_DIR/requirements.txt" ]; then
        TMP_REQ="$(mktemp).txt"
        # Filter torch, torchvision, numpy (we install/pin them separately).
        grep -v -iE '^[[:space:]]*(torch|torchvision|torchaudio|numpy|triton)([=<>!~]|$|[[:space:]])' \
            "$EVA_DIR/requirements.txt" > "$TMP_REQ"
        pip install "${PIP_FLAGS[@]}" -r "$TMP_REQ"
        rm -f "$TMP_REQ"
    else
        echo "❌ ERROR: requirements.txt not found at $EVA_DIR/requirements.txt" >&2
        exit 1
    fi

    # 2c. Install triton matching torch version (requirements.txt pins 3.1.0 for 2.5).
    echo "🏋️ [00] installing triton==${TRITON_VERSION} (matches torch ${TORCH_VERSION})"
    pip install "${PIP_FLAGS[@]}" "triton==${TRITON_VERSION}"

    # 2d. Pin numpy + setuptools (numpy 2.x breaks older C extensions; setuptools<70
    #     keeps pkg_resources which some deps use).
    echo "📦 [00] pinning numpy==1.26.4 + setuptools<70"
    pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps numpy==1.26.4 "setuptools<70"
    echo "  ✅ numpy=$(python -c 'import numpy; print(numpy.__version__)')  setuptools=$(python -c 'import setuptools; print(setuptools.__version__)')"
    echo "  ✅ torch=$(python -c 'import torch; print(torch.__version__)')"
fi

# ---------------------------------------------------------------------------
# 3. Build feature-splatting CUDA extension (modified diff-gaussian-rasterization).
#    Needed by lib/GaussianRender.py — it imports diff_gaussian_rasterization.
# ---------------------------------------------------------------------------
if [ "${BUILD_CUDA:-0}" = "1" ]; then
    echo "🎮 [00] building feature-splatting CUDA rasterizer"
    if [ -x "$CUDA_HOME/bin/nvcc" ]; then
        echo "  nvcc: $($CUDA_HOME/bin/nvcc --version | tail -1 | xargs)"
    else
        echo "⚠️ WARNING: nvcc not found at $CUDA_HOME/bin/nvcc — CUDA build will fail." >&2
        echo "         Install CUDA toolkit (major ver == torch.cuda, e.g. 11.8 for" >&2
        echo "         torch==2.5.0+cu118) and set CUDA_HOME=/path/to/cuda." >&2
    fi

    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)

    echo "📦 [00] pip install -e feature-splatting (no-build-isolation, needs nvcc)"
    pip install "${PIP_FLAGS[@]}" --no-build-isolation -e "$EVA_DIR/feature-splatting"
    if [ $? -ne 0 ]; then
        echo "❌ FAILED: feature-splatting build failed" >&2
        exit 1
    fi
    echo "  ✅ feature-splatting built"
fi

# ---------------------------------------------------------------------------
# 4. Verify imports.
# ---------------------------------------------------------------------------
echo "🔍 [00] verifying Python imports"
python - <<'PY'
import sys
ok = True
try:
    import torch; print(f"  ✅ torch {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  ⚠️ torch.cuda not available — check GPU visibility")
except Exception as e:
    print(f"  ❌ torch: {e}"); ok = False
try:
    import numpy; print(f"  ✅ numpy {numpy.__version__}")
except Exception as e:
    print(f"  ❌ numpy: {e}"); ok = False
try:
    import cv2; print(f"  ✅ cv2 {cv2.__version__}")
except Exception as e:
    print(f"  ❌ cv2: {e}"); ok = False
try:
    import open3d; print(f"  ✅ open3d {open3d.__version__}")
except Exception as e:
    print(f"  ❌ open3d: {e}"); ok = False
try:
    import einops; print("  ✅ einops")
except Exception as e:
    print(f"  ❌ einops: {e}"); ok = False
try:
    from yacs.config import CfgNode; print("  ✅ yacs")
except Exception as e:
    print(f"  ❌ yacs: {e}"); ok = False
try:
    import taichi; print(f"  ✅ taichi {taichi.__version__}")
except Exception as e:
    print(f"  ❌ taichi: {e}"); ok = False
try:
    import diff_gaussian_rasterization; print("  ✅ diff_gaussian_rasterization (feature-splatting CUDA ext)")
except Exception as e:
    print(f"  ❌ diff_gaussian_rasterization: {e}")
    print("     (rerun with BUILD_CUDA=1)"); ok = False
sys.exit(0 if ok else 1)
PY

if [ $? -ne 0 ]; then
    echo "❌ [00] some imports failed — fix above before training." >&2
    exit 1
fi

echo ""
echo "🎉 [00] Done. Env '$CONDA_ENV' ready."
echo "    Next: DATA_ROOT=/path/to/dataset GPU=0 bash $SCRIPT_DIR/run_all.sh"
