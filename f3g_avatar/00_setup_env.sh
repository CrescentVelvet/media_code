#!/usr/bin/env bash
# 00_setup_env.sh — activate conda env (torch preinstalled) & verify torch+CUDA.
# Set INSTALL_DEPS=1 to install the official requirements.txt.
# Set BUILD_CUDA=1 to build the two CUDA extensions the avatar network needs:
#   gaussians/diff_gaussian_rasterization_depth_alpha  (depth+alpha rasterizer)
#   network/styleunet                                  (DualStyleUNet from StyleAvatar)
#
# F3G-Avatar pins torch==2.1.0+cu121, pytorch3d==0.7.8, torch-scatter
# ==2.1.2+pt21cu121, triton==2.1.0 — these VERSION pins CONFLICT with other
# algos in this repo (hypir wants torch==2.6.0, retouchformer wants
# torch==1.13.1 cu117 / python3.8, hunyuanvideo wants diffusers==0.35). Use a
# DEDICATED env:
#   conda create -n f3g_avatar python=3.10 -y
#   pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
#   CONDA_ENV=f3g_avatar INSTALL_DEPS=1 bash f3g_avatar/00_setup_env.sh
#
# SKIP_TORCH=1 keeps the already-installed torch (filters the torch/torchvision
# pins out of requirements.txt) — useful if your env already has a CUDA torch
# you don't want to disturb.
#
# NOTE on pytorch3d / torch-scatter: plain `pip install` cannot build these
# from the bare PyPI pins. INSTALL_DEPS=1 installs the rest of requirements.txt
# then attempts a best-effort install of pytorch3d (conda) + torch-scatter
# (pyG wheels). See README "可能遇到的问题" #2 if these fail.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

F3G_DIR="${F3G_DIR:-$REPO_DIR/../F3G-avatar}"

echo "=== [00] Verify torch in conda env '$CONDA_ENV' ==="
python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda not available in this env — install a CUDA-enabled torch or check GPU visibility.")
PY

# Install the official deps on demand. requirements.txt pins torch==2.1.0+cu121
# (CUDA 12.1 — works on A100/RTX-3090/4090; H100 sm90 wants cu118+, see README).
# Set SKIP_TORCH=1 to keep your existing torch.
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org \
        --timeout 600 --retries 10)
    REQ="$F3G_DIR/requirements.txt"
    if [ ! -f "$REQ" ]; then
        echo "ERROR: $REQ not found. Run run_all.sh first (it clones the official repo), or set F3G_DIR." >&2
        exit 1
    fi
    if [ "${SKIP_TORCH:-0}" = "1" ]; then
        TMP_REQ="$(mktemp).txt"
        # Drop the torch / torchvision / triton version pins; keep everything else.
        grep -v -iE '^[[:space:]]*(torch|torchvision|triton)([=<>!~]|$|[[:space:]])' "$REQ" > "$TMP_REQ"
        echo "--- installing requirements.txt (SKIP_TORCH=1 -> keeping existing torch) ---"
        pip install "${PIP_FLAGS[@]}" -r "$TMP_REQ"
        rm -f "$TMP_REQ"
    else
        echo "--- installing full requirements.txt (pulls torch==2.1.0+cu121) ---"
        pip install "${PIP_FLAGS[@]}" -r "$REQ"
    fi

    # pytorch3d + torch-scatter need special handling (not buildable from bare pip).
    echo "--- best-effort: pytorch3d + torch-scatter (match torch==2.1.0+cu121) ---"
    python - <<'PY' || echo "    (pytorch3d/torch-scatter install skipped — see README #2)"
import torch, sys
try:
    import pytorch3d
    print("pytorch3d already installed:", pytorch3d.__version__)
except Exception:
    print("pytorch3d missing -> will try conda then pip")
PY
    if ! python -c "import pytorch3d" 2>/dev/null; then
        conda install -n "$CONDA_ENV" -y -c pytorch3d pytorch3d=0.7.8 || \
            pip install "${PIP_FLAGS[@]}" "pytorch3d==0.7.8" --no-build-isolation || \
            echo "    ! pytorch3d install failed — install manually (see README #2)"
    fi
    if ! python -c "import torch_scatter" 2>/dev/null; then
        pip install "${PIP_FLAGS[@]}" torch-scatter==2.1.2 \
            --no-index --find-links "https://data.pyg.org/whl/torch-2.1.0+cu121.html" || \
            pip install "${PIP_FLAGS[@]}" torch-scatter || \
            echo "    ! torch-scatter install failed — install manually (see README #2)"
    fi
    echo "--- deps installed. Tip: verify with 'python -c \"import pytorch3d,torch_scatter,smplx,cv2,trimesh\"' ---"
fi

# Build the two CUDA extensions (required by AvatarNet for train AND inference).
if [ "${BUILD_CUDA:-0}" = "1" ]; then
    echo "--- building CUDA ext: diff-gaussian-rasterization (depth+alpha) ---"
    pushd "$F3G_DIR/gaussians/diff_gaussian_rasterization_depth_alpha" >/dev/null
    python setup.py install
    popd >/dev/null
    echo "--- building CUDA ext: StyleUNet (from StyleAvatar) ---"
    pushd "$F3G_DIR/network/styleunet" >/dev/null
    python setup.py install
    popd >/dev/null
    echo "--- CUDA extensions built. Verify: python -c \"import diff_gaussian_rasterization; import styleunet\" ---"
fi

echo "=== [00] Done. Env '$CONDA_ENV' ready. (Missing a dep? INSTALL_DEPS=1 bash this, or pip install it) ==="
