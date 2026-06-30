#!/usr/bin/env bash
# 00_setup_env.sh — activate conda env (torch preinstalled) & verify torch+CUDA.
# Reuses an existing env to avoid re-downloading torch from a torch-specific
# index. Set INSTALL_DEPS=1 to install the official requirements.txt (this
# WILL pull/upgrade torch to >=2.6 + CUDA 12.x as the repo requires).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

HYVIDEO_DIR="${HYVIDEO_DIR:-$REPO_DIR/../HunyuanVideo-1.5}"

echo "=== [00] Verify torch in conda env '$CONDA_ENV' ==="
python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda not available in this env — install a CUDA-enabled torch or check GPU visibility.")
PY

# Install the official deps on demand. HunyuanVideo-1.5 pins torch>=2.6,
# diffusers==0.35.0, transformers==4.57.1, etc. — these go into the shared env.
# torch is pulled from PyPI (CUDA wheel, works on A100/H100/RTX-4090).
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org \
        --timeout 600 --retries 10)
    if [ ! -f "$HYVIDEO_DIR/requirements.txt" ]; then
        echo "ERROR: $HYVIDEO_DIR/requirements.txt not found. Run run_all.sh first (it clones the official repo)." >&2
        exit 1
    fi
    echo "--- installing official requirements.txt ---"
    pip install "${PIP_FLAGS[@]}" -r "$HYVIDEO_DIR/requirements.txt"
    echo "--- installing tencentcloud-sdk-python (prompt-rewrite client) ---"
    pip install "${PIP_FLAGS[@]}" -i https://mirrors.tencent.com/pypi/simple/ --upgrade tencentcloud-sdk-python || \
        pip install "${PIP_FLAGS[@]}" --upgrade tencentcloud-sdk-python
    echo "--- done. Optional kernels (install separately if needed): ---"
    echo "    flash-attn:   pip install flash-attn --no-build-isolation"
    echo "    sage-attn:    see https://github.com/cooper1637/SageAttention (build)"
    echo "    flex-block:   see https://github.com/Tencent-Hunyuan/flex-block-attn (sparse attn, 720p)"
fi

echo "=== [00] Done. Env '$CONDA_ENV' ready. (Missing a dep? INSTALL_DEPS=1 bash this, or pip install it) ==="
