#!/usr/bin/env bash
# 00_setup_env.sh — activate conda env (torch preinstalled) & verify torch+CUDA.
# Set INSTALL_DEPS=1 to install the FLUX.1 runtime deps (no official repo to
# clone — FLUX.1 runs purely via diffusers' FluxPipeline).
#
# FLUX.1 needs: diffusers (recent, >=0.30 — Flux support landed there),
# transformers (T5 + CLIP text encoders), accelerate (CPU offload for VRAM),
# Pillow (save PNG), sentencepiece (T5 tokenizer). The shared 'doll' env
# (hunyuanvideo-1.5 pulls diffusers==0.35.0 / transformers==4.57.1) already
# satisfies this. If your env has an OLDER diffusers, use a dedicated env:
#   conda create -n flux1 python=3.10 -y && CONDA_ENV=flux1 INSTALL_DEPS=1 bash ...
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

# Install deps on demand. We don't ship a requirements.txt (no cloned repo), so
# install the minimal set directly. diffusers is upgraded so FluxPipeline exists.
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org \
        --timeout 600 --retries 10)
    echo "--- installing FLUX.1 runtime deps ---"
    pip install "${PIP_FLAGS[@]}" -U "diffusers>=0.30" transformers accelerate Pillow sentencepiece
    echo "--- optional: faster attention (lower VRAM + speed) ---"
    echo "    pip install flash-attn --no-build-isolation    # then ATTN_IMPL=flash_attention_2 (advanced; see README)"
    echo "--- deps installed. Verify with: python -c \"import diffusers,transformers,accelerate; print(diffusers.__version__)\" ---"
fi

echo "=== [00] Done. Env '$CONDA_ENV' ready. (Missing a dep? INSTALL_DEPS=1 bash this, or pip install it) ==="
