#!/usr/bin/env bash
# 00_setup_env.sh — clone the shared 'doll' conda env into 'flux2', then
# upgrade diffusers (to 0.39 — where Flux2Pipeline / Flux2KleinPipeline
# landed) + transformers + accelerate. FLUX.2 runs purely via diffusers, so
# there is NO official code repo to clone (the HF snapshot is everything).
#
# Why clone 'doll'? The shared 'doll' env already has a CUDA torch +
# transformers + accelerate + Pillow. We clone it (so torch is reused, no
# re-download) into a DEDICATED 'flux2' env, then bump diffusers 0.35 -> 0.39
# there. This leaves 'doll' untouched — other algos that pin diffusers 0.35
# (hunyuanvideo-1.5, wan22, ...) keep working.
#
# FLUX.2 needs:
#   - diffusers >= 0.39   (Flux2Pipeline + Flux2KleinPipeline; 0.35 lacks them)
#   - transformers (recent — Mistral3 [dev text encoder] + Qwen3 [klein text
#     encoder] landed in 4.50+/4.57; doll's 4.57.1 is fine, upgrade to be safe)
#   - accelerate (model CPU offload for VRAM)
#   - Pillow (save PNG; already in doll)
#   - bitsandbytes  (OPTIONAL: only for QUANT=4bit/8bit dev on consumer GPUs)
#   - sentencepiece (NOT needed — FLUX.2 dropped T5; uses Mistral3/Qwen3)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# 00 always operates on the dedicated 'flux2' env (it creates it if missing).
# Override any inherited CONDA_ENV so we never accidentally upgrade 'doll'.
CONDA_ENV="flux2"

# 1. Clone 'doll' -> 'flux2' once (reuse its CUDA torch). Skip if flux2 exists.
if [ "${CONDA_ENV}" = "flux2" ] && ! conda env list 2>/dev/null | grep -qE "(^| )flux2( |$)"; then
    if ! conda env list 2>/dev/null | grep -qE "(^| )doll( |$)"; then
        echo "❌ ERROR: source env 'doll' not found." >&2
        echo "   Create it first (e.g. conda create -n doll python=3.10 -y && pip install torch ...) or" >&2
        echo "   clone a different base: BASE_ENV=hunyuanvideo-1.5 bash $SCRIPT_DIR/00_setup_env.sh" >&2
        exit 1
    fi
    BASE_ENV="${BASE_ENV:-doll}"
    echo "📦 cloning conda env '$BASE_ENV' -> 'flux2' (reuses its CUDA torch) ..."
    conda create -n flux2 --clone "$BASE_ENV" -y
fi

# (Re)activate to make sure we're in the right env after the clone above.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "🚀 [00] Verify torch in conda env '$CONDA_ENV'"
python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("❌ ERROR: torch.cuda not available in this env — install a CUDA-enabled torch or check GPU visibility.")
PY

# 2. Install / upgrade deps on demand.
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org \
        --timeout 600 --retries 10)
    echo "📦 upgrading FLUX.2 runtime deps (diffusers 0.39 has Flux2 pipelines) ..."
    pip install "${PIP_FLAGS[@]}" -U "diffusers==0.39.0" transformers accelerate Pillow
    echo "📦 optional: bitsandbytes (4-bit/8-bit quant for FLUX.2-dev on consumer GPUs)"
    echo "    install with:  pip install bitsandbytes   # then QUANT=4bit bash 02_run_inference.sh"
    echo "    (skip if you run FLUX.2-klein-9B, or dev on an H100/H200/B200 with OFFLOAD=model)"
fi

# 3. Verify the Flux2 pipelines import (the whole point of upgrading diffusers).
echo "🔍 verifying Flux2Pipeline / Flux2KleinPipeline import ..."
python - <<'PY'
try:
    from diffusers import Flux2Pipeline, Flux2KleinPipeline  # noqa: F401
    import diffusers
    print(f"✅ diffusers {diffusers.__version__}: Flux2Pipeline + Flux2KleinPipeline OK")
except ImportError as e:
    raise SystemExit(
        f"❌ Flux2 pipelines not importable ({e}).\n"
        f"   diffusers >= 0.39 required (doll has 0.35). Run: INSTALL_DEPS=1 bash flux2/00_setup_env.sh"
    )
PY

echo "🎉 [00] Done. Env '$CONDA_ENV' ready."
echo "    Next: bash $SCRIPT_DIR/01_download_models.sh   # (or skip if you already have the weights)"
echo "          GPU=0 bash $SCRIPT_DIR/02_run_inference.sh"
