#!/usr/bin/env bash
# 00_setup_env.sh — create venv and install TripoSplat runtime deps.
# Run on the Ubuntu A100 server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_DIR="$SCRIPT_DIR"                          # .../media_code/triposplat
REPO_DIR="$(dirname "$ALGO_DIR")"               # .../media_code

# Optional proxy (gitignored proxy.env at repo root; see proxy.env.example).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

# ---- Config (override via env vars if needed) -------------------------------
VENV_DIR="${VENV_DIR:-$ALGO_DIR/.venv}"         # per-algorithm venv
CUDA_TAG="${CUDA_TAG:-cu124}"                   # cu118 / cu121 / cu124 / cu126
INSTALL_GRADIO="${INSTALL_GRADIO:-1}"           # 1 = install gradio (web demo)
# ----------------------------------------------------------------------------

echo "=== [00] Setting up Python environment (triposplat) ==="
echo "  venv:     $VENV_DIR"
echo "  cuda tag: $CUDA_TAG"
echo "  gradio:   $INSTALL_GRADIO"

if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY=python
fi

if [ ! -d "$VENV_DIR" ]; then
    "$PY" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Hosts to trust even when a corporate proxy's TLS cert can't be verified
# (SSL:CERTIFICATE_VERIFY_FAILED). Harmless when not needed.
TH=(--trusted-host pypi.org --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org)

python -m pip install "${TH[@]}" --upgrade pip

# torch + torchvision from the CUDA-specific index (wheels bundle their own
# CUDA runtime; only the NVIDIA driver needs to be recent enough).
echo "--- installing torch (CUDA $CUDA_TAG) ---"
pip install "${TH[@]}" --index-url "https://download.pytorch.org/whl/$CUDA_TAG" torch torchvision

# Runtime deps used by triposplat.py / model.py / run_example.py.
echo "--- installing runtime deps ---"
pip install "${TH[@]}" numpy safetensors pillow tqdm huggingface_hub

# Optional: faster HF downloads. NOTE: hf_transfer (Rust) may not honor a
# corporate HTTP proxy, so it is OFF by default. Enable with
# HF_HUB_ENABLE_HF_TRANSFER=1 only if you don't need the proxy.
pip install "${TH[@]}" hf_transfer || true
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

# Optional: Gradio web demo (run_gradio.py).
if [ "$INSTALL_GRADIO" = "1" ]; then
    echo "--- installing gradio ---"
    pip install "${TH[@]}" gradio
fi

echo "--- verifying CUDA visibility ---"
python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
PY

echo "=== [00] Done. Activate later with: source $VENV_DIR/bin/activate ==="
