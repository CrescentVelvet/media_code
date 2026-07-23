#!/usr/bin/env bash
# 00_setup_env.sh — activate conda env (torch preinstalled) & verify torch+CUDA.
# Set INSTALL_DEPS=1 to install the official requirements.txt (Python deps only;
# the two CUDA submodules are built separately by BUILD_CUDA=1).
# Set BUILD_CUDA=1 to build the two CUDA extensions Deformable-GS needs:
#   submodules/depth-diff-gaussian-rasterization  (depth+alpha rasterizer, branch filter-norm)
#   submodules/simple-knn                          (knn for init point cloud)
#
# Deformable-3D-Gaussians pins torch==1.13.1+cu116 / torchvision==0.14.1
# (python 3.7) in its README — these VERSION pins CONFLICT with other algos in
# this repo (hypir wants torch==2.6.0, f3g_avatar wants torch==2.1.0+cu121,
# retouchformer wants torch==1.13.1 cu117 / python3.8). Use a DEDICATED env:
#   conda create -n deformable_gaussians python=3.7 -y
#   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
#   CONDA_ENV=deformable_gaussians INSTALL_DEPS=1 BUILD_CUDA=1 bash deformable_gaussians/00_setup_env.sh
#
# SKIP_TORCH=1 keeps the already-installed torch (filters the torch/torchvision
# pins out of any pin list) — useful if your env already has a CUDA torch you
# don't want to disturb. (requirements.txt has no torch pin; this is a no-op
# here unless you point it at a file that does.)
#
# NOTE on the CUDA build: diff-gaussian-rasterization's setup.py uses
# torch.utils.cpp_extension, which needs `nvcc` (CUDA toolkit). The toolkit's
# major version MUST match torch's CUDA build (torch==1.13.1+cu116 -> CUDA 11.6
# toolkit). Set CUDA_HOME if nvcc isn't auto-detected (default /usr/local/cuda).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

DG_DIR="${DG_DIR:-$REPO_DIR/../Deformable-3D-Gaussians}"

echo "=== [00] Verify torch in conda env '$CONDA_ENV' ==="
python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda not available in this env — install a CUDA-enabled torch or check GPU visibility.")
PY

# Install the Python deps on demand. requirements.txt lists the two submodules
# as local paths (which would trigger their CUDA build mid-install); we filter
# those out here and build them separately (BUILD_CUDA=1) so a missing CUDA
# toolkit doesn't abort the whole pip install.
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org \
        --timeout 600 --retries 10)
    REQ="$DG_DIR/requirements.txt"
    if [ ! -f "$REQ" ]; then
        echo "ERROR: $REQ not found. Run run_all.sh first (it clones the official repo), or set DG_DIR." >&2
        exit 1
    fi
    TMP_REQ="$(mktemp).txt"
    # Drop the submodule local-path lines (built via BUILD_CUDA) and any
    # torch/torchvision pins (keep existing torch when SKIP_TORCH=1).
    grep -v -iE '^[[:space:]]*submodules/' "$REQ" \
        | grep -v -iE '^[[:space:]]*(torch|torchvision)([=<>!~]|$|[[:space:]])' > "$TMP_REQ"
    echo "--- installing requirements.txt (submodules/ + torch pins filtered out) ---"
    cat "$TMP_REQ"
    pip install "${PIP_FLAGS[@]}" -r "$TMP_REQ"
    rm -f "$TMP_REQ"
    echo "--- deps installed. Tip: verify with 'python -c \"import plyfile,imageio,cv2,scipy,lpips,dearpygui\"' ---"
fi

# Build the two CUDA extensions (required by train.py AND render.py — the
# gaussian_renderer imports diff_gaussian_rasterization, and Scene init uses
# simple_knn for the initial point cloud).
if [ "${BUILD_CUDA:-0}" = "1" ]; then
    # Make sure the submodules are checked out (run_all clones --recursive, but
    # a plain clone or a failed recursive fetch leaves them empty).
    if [ ! -f "$DG_DIR/submodules/simple-knn/setup.py" ] || \
       [ ! -f "$DG_DIR/submodules/depth-diff-gaussian-rasterization/setup.py" ]; then
        echo "--- submodules missing -> git submodule update --init --recursive ---"
        ( cd "$DG_DIR" && git submodule update --init --recursive ) || \
        ( cd "$DG_DIR" && git -c http.sslVerify=false submodule update --init --recursive )
    fi

    # diff-gaussian-rasterization's setup.py needs nvcc. Point CUDA_HOME at the
    # toolkit whose major version matches torch's CUDA build (cu116 -> 11.6).
    : "${CUDA_HOME:=/usr/local/cuda}"
    export CUDA_HOME
    if [ -x "$CUDA_HOME/bin/nvcc" ]; then
        echo "--- nvcc: $($CUDA_HOME/bin/nvcc --version | tail -1 | xargs) ---"
    else
        echo "WARNING: nvcc not found at $CUDA_HOME/bin/nvcc — the CUDA build will fail." >&2
        echo "         Install the CUDA toolkit (major ver == torch.cuda, e.g. 11.6 for" >&2
        echo "         torch==1.13.1+cu116) and set CUDA_HOME=/path/to/cuda." >&2
    fi

    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)

    echo "--- building CUDA ext: simple-knn ---"
    pip install "${PIP_FLAGS[@]}" --no-build-isolation "$DG_DIR/submodules/simple-knn"

    echo "--- building CUDA ext: depth-diff-gaussian-rasterization (filter-norm) ---"
    pip install "${PIP_FLAGS[@]}" --no-build-isolation "$DG_DIR/submodules/depth-diff-gaussian-rasterization"

    echo "--- CUDA extensions built. Verify: python -c \"import simple_knn, diff_gaussian_rasterization\" ---"
fi

echo "=== [00] Done. Env '$CONDA_ENV' ready. (Missing a dep? INSTALL_DEPS=1 bash this; missing CUDA ext? BUILD_CUDA=1) ==="
