#!/usr/bin/env bash
# 00_setup_env.sh — activate conda env (torch preinstalled) & verify torch+CUDA.
# Reuses an existing env to avoid re-downloading torch. No venv is created.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "=== [00] Verify torch in conda env '$CONDA_ENV' ==="
python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda not available in this env — install a CUDA-enabled torch or check GPU visibility.")
PY

# Deps are installed ON DEMAND by default. Set INSTALL_DEPS=1 to install the
# known runtime set in one shot (small packages; torch is NOT reinstalled).
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org \
        --timeout 600 --retries 10)
    echo "--- installing known runtime deps ---"
    pip install "${PIP_FLAGS[@]}" numpy safetensors pillow tqdm huggingface_hub
fi

echo "=== [00] Done. Env '$CONDA_ENV' ready. (Missing a dep? pip install it, or rerun with INSTALL_DEPS=1) ==="
