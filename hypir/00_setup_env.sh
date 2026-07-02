#!/usr/bin/env bash
# 00_setup_env.sh — activate conda env (torch preinstalled) & verify torch+CUDA.
# Set INSTALL_DEPS=1 to install the official requirements.txt.
#
# HYPIR pins torch==2.6.0, diffusers==0.32.2, transformers==4.49.0, peft==0.14.0
# — these VERSION pins CONFLICT with other algos in this repo (e.g. hunyuanvideo
# wants diffusers==0.35.0 / transformers==4.57.1). Use a DEDICATED env:
#   conda create -n hypir python=3.10 -y
#   CONDA_ENV=hypir INSTALL_DEPS=1 bash hypir/00_setup_env.sh
# Default CONDA_ENV=doll matches the repo, but installing HYPIR requirements into
# doll will overwrite those pins for every other algo — prefer a dedicated env.
#
# SKIP_TORCH=1 keeps the already-installed torch (filters the torch/torchvision
# pins out of requirements.txt) — useful if your env already has a CUDA torch you
# don't want to disturb.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

HYPIR_DIR="${HYPIR_DIR:-$REPO_DIR/../HYPIR}"

echo "=== [00] Verify torch in conda env '$CONDA_ENV' ==="
python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda not available in this env — install a CUDA-enabled torch or check GPU visibility.")
PY

# Install the official deps on demand. The full requirements.txt pulls
# torch==2.6.0 from PyPI (the default manylinux wheel is CUDA 12.x — works on
# A100/H100/RTX-4090). Set SKIP_TORCH=1 to keep your existing torch.
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org \
        --timeout 600 --retries 10)
    REQ="$HYPIR_DIR/requirements.txt"
    if [ ! -f "$REQ" ]; then
        echo "ERROR: $REQ not found. Run run_all.sh first (it clones the official repo), or set HYPIR_DIR." >&2
        exit 1
    fi
    if [ "${SKIP_TORCH:-0}" = "1" ]; then
        TMP_REQ="$(mktemp).txt"
        # Drop the torch / torchvision version pins; keep everything else.
        grep -v -iE '^[[:space:]]*(torch|torchvision)([=<>!~]|$|[[:space:]])' "$REQ" > "$TMP_REQ"
        echo "--- installing requirements.txt (SKIP_TORCH=1 -> keeping existing torch) ---"
        cat "$TMP_REQ"
        pip install "${PIP_FLAGS[@]}" -r "$TMP_REQ"
        rm -f "$TMP_REQ"
    else
        echo "--- installing full requirements.txt (pulls torch==2.6.0, CUDA 12.x) ---"
        pip install "${PIP_FLAGS[@]}" -r "$REQ"
    fi
    echo "--- deps installed. Tip: verify with 'python -c \"import diffusers,transformers,peft,accelerate\"' ---"
fi

echo "=== [00] Done. Env '$CONDA_ENV' ready. (Missing a dep? INSTALL_DEPS=1 bash this, or pip install it) ==="
