#!/usr/bin/env bash
# 00_setup_env.sh — create the pi3_3dgs conda env (CPython 3.10), install
# Pi3 + 2D Gaussian Splatting deps, and build 2DGS's two CUDA extensions:
#   submodules/diff-surfel-rasterization  (the 2DGS surfel rasterizer)
#   submodules/simple-knn                  (kNN for init point cloud)
#
# First time:
#   INSTALL_DEPS=1 BUILD_CUDA=1 bash pi3_3dgs/00_setup_env.sh
# After that (verify only):
#   bash pi3_3dgs/00_setup_env.sh
#
# We reuse the local cp310 torch 2.6.0+cu124 wheels (same as wan22_rotate) to
# avoid re-downloading ~3GB of nvidia deps. 2DGS's environment.yml pins
# python 3.8 + torch 2.0.0, but the diff-surfel-rasterization builds fine
# against torch 2.6 + CUDA 12.4 toolkit (same major cu version).
#
# Env:
#   CONDA_ENV=pi3_3dgs            (dedicated env; both Pi3 + 2DGS deps coexist)
#   CUDA_HOME=/usr/local/cuda      (must match torch's cu major: 12.4 for cu124)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# Create env if missing (CPython 3.10; matches local cp310 torch wheels).
if ! conda env list 2>/dev/null | grep -qw "$CONDA_ENV"; then
    echo "--- conda env '$CONDA_ENV' not found; creating python=3.10 (CPython) ---"
    conda create -n "$CONDA_ENV" python=3.10 -y
    conda activate "$CONDA_ENV"
    impl="$(python -c 'import platform; print(platform.python_implementation())')"
    if [ "$impl" != "CPython" ]; then
        echo "ERROR: env '$CONDA_ENV' python implementation is $impl (should be CPython)." >&2
        echo "       conda-forge may have substituted GraalPy. Recreate:" >&2
        echo "         conda env remove -n $CONDA_ENV && conda create -n $CONDA_ENV python=3.10 -y --override-channels -c defaults" >&2
        exit 1
    fi
    echo "  [OK] python=$(python --version 2>&1 | cut -d' ' -f2) ($impl)"
fi

echo "=== [00] Verify torch in conda env '$CONDA_ENV' ==="
python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda not available — install a CUDA torch or check GPU visibility.")
PY

# ---------------------------------------------------------------------------
# 0. Clone Pi3 + 2DGS official repos (recursive, w/ submodules for 2DGS).
# ---------------------------------------------------------------------------
PI3_REPO="${PI3_REPO:-https://github.com/yyfz/Pi3.git}"
GS2D_REPO="${GS2D_REPO:-https://github.com/hbb1/2d-gaussian-splatting.git}"

if [ ! -d "$PI3_DIR/.git" ]; then
    echo "--- cloning Pi3 -> $PI3_DIR ---"
    mkdir -p "$(dirname "$PI3_DIR")"
    git clone "$PI3_REPO" "$PI3_DIR" || \
        git -c http.sslVerify=false clone "$PI3_REPO" "$PI3_DIR"
else
    echo "--- Pi3 repo present: $PI3_DIR ---"
fi

if [ ! -d "$GS2D_DIR/.git" ]; then
    echo "--- cloning 2D Gaussian Splatting (recursive) -> $GS2D_DIR ---"
    mkdir -p "$(dirname "$GS2D_DIR")"
    git clone --recursive "$GS2D_REPO" "$GS2D_DIR" || \
        git -c http.sslVerify=false clone --recursive "$GS2D_REPO" "$GS2D_DIR"
else
    echo "--- 2DGS repo present: $GS2D_DIR ---"
fi
# Ensure 2DGS submodules are checked out (simple-knn lives on gitlab.inria.fr,
# diff-surfel-rasterization on github; a plain clone leaves submodules/ empty).
if [ ! -f "$GS2D_DIR/submodules/simple-knn/setup.py" ] || \
   [ ! -f "$GS2D_DIR/submodules/diff-surfel-rasterization/setup.py" ]; then
    echo "--- ensuring 2DGS submodules are initialized ---"
    ( cd "$GS2D_DIR" && git submodule update --init --recursive ) || \
        ( cd "$GS2D_DIR" && git -c http.sslVerify=false submodule update --init --recursive )
fi

# ---------------------------------------------------------------------------
# 1. Install deps (first time or when INSTALL_DEPS=1).
# ---------------------------------------------------------------------------
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org \
        --timeout 600 --retries 10)
    _proxy="${https_proxy:-${http_proxy:-}}"
    if [ -n "$_proxy" ]; then
        PIP_FLAGS+=(--proxy "$_proxy")
        echo "--- using proxy: $_proxy ---"
    else
        echo "WARNING: no proxy set; pip may fail if no direct internet." >&2
    fi

    # 1a. Uninstall torchaudio (version conflict) + install PyTorch cu124 via
    #     LOCAL wheels at $WAN_MODEL_DIR (same set as wan22_rotate).
    echo "--- uninstalling torchaudio (version conflict) ---"
    pip uninstall -y torchaudio 2>/dev/null || true

    echo "--- installing PyTorch cu124 local wheels (if present at $WAN_MODEL_DIR) ---"
    LOCAL_WHEELS=()
    for w in \
        "$WAN_MODEL_DIR/nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl" \
        "$WAN_MODEL_DIR/nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl" \
        "$WAN_MODEL_DIR/nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl" \
        "$WAN_MODEL_DIR/triton-3.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
        "$WAN_MODEL_DIR/nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl" \
        "$WAN_MODEL_DIR/nvidia_cusparselt_cu12-0.6.2-py3-none-manylinux2014_x86_64.whl" \
        "$WAN_MODEL_DIR/nvidia_nccl_cu12-2.21.5-py3-none-manylinux2014_x86_64.whl" \
        "$WAN_MODEL_DIR/torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl" \
        "$WAN_MODEL_DIR/torchvision-0.21.0+cu124-cp310-cp310-linux_x86_64.whl" \
        ; do
        [ -f "$w" ] && LOCAL_WHEELS+=("$w")
    done
    if [ ${#LOCAL_WHEELS[@]} -ge 1 ]; then
        echo "  installing local wheels: ${LOCAL_WHEELS[*]}"
        pip install --force-reinstall --no-deps "${LOCAL_WHEELS[@]}"
    else
        echo "  no local wheels found; falling back to pip install torch 2.6.0+cu124 (will DL ~3GB)"
        pip install "${PIP_FLAGS[@]}" torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
            --index-url https://download.pytorch.org/whl/cu124
    fi

    echo "--- installing nvidia deps + sympy + triton (torch 2.6.0+cu124 pins) ---"
    pip install "${PIP_FLAGS[@]}" \
        nvidia-cuda-nvrtc-cu12==12.4.127 \
        nvidia-cuda-runtime-cu12==12.4.127 \
        nvidia-cuda-cupti-cu12==12.4.127 \
        nvidia-cudnn-cu12==9.1.0.70 \
        nvidia-cublas-cu12==12.4.5.8 \
        nvidia-cufft-cu12==11.2.1.3 \
        nvidia-curand-cu12==10.3.5.147 \
        nvidia-cusolver-cu12==11.6.1.9 \
        nvidia-cusparse-cu12==12.3.1.170 \
        nvidia-cusparselt-cu12==0.6.2 \
        nvidia-nccl-cu12==2.21.5 \
        nvidia-nvtx-cu12==12.4.127 \
        nvidia-nvjitlink-cu12==12.4.127 \
        sympy==1.13.1 \
        triton==3.2.0

    # numpy 1.26.4 is the sweet spot: Pi3 requirements.txt pins 1.26.4,
    # 2DGS doesn't pin (works with 1.26.x), and detectron2-free env avoids ABI issues.
    echo "--- pinning numpy==1.26.4 (Pi3 + 2DGS both compatible) ---"
    pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps numpy==1.26.4

    # 1b. Pi3 deps (Pi3 is imported via sys.path, no `pip install -e .` needed).
    echo "--- installing Pi3 Python deps (from requirements.txt) ---"
    if [ -f "$PI3_DIR/requirements.txt" ]; then
        # Filter out torch / numpy pins (we just installed the right versions).
        TMP_REQ="$(mktemp).txt"
        grep -v -iE '^[[:space:]]*(torch|torchvision|numpy)([=<>!~]|$|[[:space:]])' \
            "$PI3_DIR/requirements.txt" > "$TMP_REQ"
        cat "$TMP_REQ"
        pip install "${PIP_FLAGS[@]}" -r "$TMP_REQ"
        rm -f "$TMP_REQ"
    else
        # Fallback: install the known deps directly (Pi3 README list).
        pip install "${PIP_FLAGS[@]}" pillow opencv-python plyfile \
            huggingface_hub safetensors einops
    fi

    # 1c. 2DGS Python deps (filter out the two submodule local-path lines,
    #     which would trigger their CUDA build mid-install; built separately).
    echo "--- installing 2DGS Python deps (submodules/ filtered out) ---"
    # 2DGS has no requirements.txt; its deps live in environment.yml's pip: block.
    pip install "${PIP_FLAGS[@]}" \
        open3d==0.18.0 mediapy==1.1.2 lpips==0.1.4 \
        scikit-image==0.21.0 tqdm==4.66.2 trimesh==4.3.2 \
        plyfile opencv-python "setuptools<70"

    # Pin numpy + setuptools again — the above installs may have bumped them.
    echo "--- re-pinning numpy==1.26.4 + setuptools<70 (post-install) ---"
    pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps numpy==1.26.4 "setuptools<70"
    echo "  numpy=$(python -c 'import numpy; print(numpy.__version__)')  setuptools=$(python -c 'import setuptools; print(setuptools.__version__)')"
    echo "  torch=$(python -c 'import torch; print(torch.__version__)')"
fi

# ---------------------------------------------------------------------------
# 2. Build 2DGS's two CUDA extensions (needed by train.py AND render.py —
#    gaussian_renderer imports diff_surfel_rasterization, and Scene init
#    uses simple_knn for the initial point cloud).
# ---------------------------------------------------------------------------
if [ "${BUILD_CUDA:-0}" = "1" ]; then
    if [ ! -f "$GS2D_DIR/submodules/simple-knn/setup.py" ] || \
       [ ! -f "$GS2D_DIR/submodules/diff-surfel-rasterization/setup.py" ]; then
        echo "--- 2DGS submodules missing -> git submodule update --init --recursive ---"
        ( cd "$GS2D_DIR" && git submodule update --init --recursive ) || \
            ( cd "$GS2D_DIR" && git -c http.sslVerify=false submodule update --init --recursive )
    fi

    # diff-surfel-rasterization's setup.py needs nvcc. CUDA_HOME must point at a
    # toolkit whose major version matches torch's CUDA build (cu124 -> 12.4).
    if [ -x "$CUDA_HOME/bin/nvcc" ]; then
        echo "--- nvcc: $($CUDA_HOME/bin/nvcc --version | tail -1 | xargs) ---"
    else
        echo "WARNING: nvcc not found at $CUDA_HOME/bin/nvcc — the CUDA build will fail." >&2
        echo "         Install CUDA toolkit (major ver == torch.cuda, e.g. 12.4 for" >&2
        echo "         torch==2.6.0+cu124) and set CUDA_HOME=/path/to/cuda." >&2
    fi

    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)

    echo "--- building CUDA ext: simple-knn ---"
    pip install "${PIP_FLAGS[@]}" --no-build-isolation "$GS2D_DIR/submodules/simple-knn"

    echo "--- building CUDA ext: diff-surfel-rasterization (2DGS surfel rasterizer) ---"
    pip install "${PIP_FLAGS[@]}" --no-build-isolation "$GS2D_DIR/submodules/diff-surfel-rasterization"

    echo "--- CUDA extensions built. Verify: python -c \"import simple_knn, diff_surfel_rasterization\" ---"
fi

# ---------------------------------------------------------------------------
# 3. Verify Pi3 checkpoint is at $PI3_CKPT.
# ---------------------------------------------------------------------------
echo "--- [verify] Pi3 checkpoint ---"
if [ -f "$PI3_CKPT" ]; then
    echo "  [OK] $PI3_CKPT"
else
    echo "  [MISS] $PI3_CKPT" >&2
    echo "  Download from https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors" >&2
    echo "    -> place at $PI3_CKPT" >&2
fi

# ---------------------------------------------------------------------------
# 4. Verify imports.
# ---------------------------------------------------------------------------
echo "--- [verify] Python imports ---"
python - <<'PY'
import sys
ok = True
try:
    import torch; print(f"  [OK] torch {torch.__version__}")
except Exception as e:
    print(f"  [MISS] torch: {e}"); ok = False
try:
    import numpy; print(f"  [OK] numpy {numpy.__version__}")
except Exception as e:
    print(f"  [MISS] numpy: {e}"); ok = False
try:
    import cv2; print(f"  [OK] cv2 {cv2.__version__}")
except Exception as e:
    print(f"  [MISS] cv2: {e}"); ok = False
try:
    import safetensors; print("  [OK] safetensors")
except Exception as e:
    print(f"  [MISS] safetensors: {e}"); ok = False
try:
    import plyfile; print("  [OK] plyfile")
except Exception as e:
    print(f"  [MISS] plyfile: {e}"); ok = False
try:
    import simple_knn; print("  [OK] simple_knn (CUDA ext for 2DGS)")
except Exception as e:
    print(f"  [MISS] simple_knn: {e} (rerun with BUILD_CUDA=1)"); ok = False
try:
    import diff_surfel_rasterization; print("  [OK] diff_surfel_rasterization (2DGS surfel rasterizer)")
except Exception as e:
    print(f"  [MISS] diff_surfel_rasterization: {e} (rerun with BUILD_CUDA=1)"); ok = False
sys.exit(0 if ok else 1)
PY

echo ""
echo "=== [00] Done. Env '$CONDA_ENV' ready. ==="
echo "    Next: INPUT=/path/to/rotate_360.mp4 GPU=0 bash $SCRIPT_DIR/run_all.sh"
