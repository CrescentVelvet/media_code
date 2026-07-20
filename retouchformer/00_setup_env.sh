#!/usr/bin/env bash
# 00_setup_env.sh — activate conda env (torch preinstalled) & verify torch+CUDA.
# Set INSTALL_DEPS=1 to install the inference-only deps
# (requirements_inference.txt in this dir — a trimmed subset of the official
# requirements.txt; the official one has an uninstallable `transformers==4.43.0.dev0`
# dev pin and many training-only heavies).
#
# RetouchFormer pins python==3.8 / torch==1.13.1 (cu117). These pins CONFLICT
# with other algos in this repo — use a DEDICATED env:
#   conda create -n retouchformer python=3.8 -y
#   pip install torch==1.13.1 torchvision==0.14.1
#   CONDA_ENV=retouchformer INSTALL_DEPS=1 bash retouchformer/00_setup_env.sh
#
# SKIP_TORCH=1 keeps the already-installed torch (this file already excludes
# torch from requirements, so it's mostly a no-op; kept for parity with hypir).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

RETOUCH_DIR="${RETOUCH_DIR:-$REPO_DIR/../RetouchFormer}"

echo "=== [00] Verify torch in conda env '$CONDA_ENV' ==="
python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("WARNING: torch.cuda not available — RetouchFormer's op/ CUDA kernels will use the "
          "pure-PyTorch fallback (slower). On a GPU box, install a CUDA-enabled torch.", file=__import__('sys').stderr)
PY

# Install the inference-only deps on demand.
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)
    REQ="$SCRIPT_DIR/requirements_inference.txt"
    echo "--- installing inference-only deps (SKIP_TORCH=1 -> keeping existing torch) ---"
    # requirements_inference.txt already excludes torch/torchvision; SKIP_TORCH
    # is a no-op here but accepted for parity with hypir.
    if [ "${SKIP_TORCH:-0}" = "1" ]; then
        echo "(SKIP_TORCH=1: requirements_inference.txt has no torch pins, nothing to skip)"
    fi
    pip install "${PIP_FLAGS[@]}" -r "$REQ"
    echo "--- deps installed. Verify: python -c \"import kornia,timm,einops,cv2,skimage; print('ok')\" ---"
fi

# The custom CUDA ops in op/ (fused_act, upfirdn2d) are JIT-compiled by
# torch.utils.cpp_extension.load on first import (Linux+CUDA). Needs ninja
# (installed above) + a working nvcc (CUDA toolkit). Warn if nvcc is absent —
# otherwise the first inference run will raise a BuildExtension error.
if ! command -v nvcc >/dev/null 2>&1; then
    echo "WARNING: 'nvcc' not found on PATH. RetouchFormer's op/ ships stylegan2 CUDA kernels" >&2
    echo "         (fused_bias_act, upfirdn2d) that are JIT-compiled on first import (Linux+CUDA)." >&2
    echo "         Install the CUDA toolkit (version matching your torch, e.g. CUDA 11.7 for" >&2
    echo "         torch==1.13.1) or inference will fall back to the slow pure-PyTorch path / fail." >&2
else
    echo "--- nvcc found: $(nvcc --version | tail -1) ---"
fi

echo "=== [00] Done. Env '$CONDA_ENV' ready. (Missing a dep? INSTALL_DEPS=1 bash this, or pip install it) ==="
