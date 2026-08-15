#!/usr/bin/env bash
# 00_setup_env.sh — clone the shared 'doll' conda env into 'flux_human', then
# install the Flux1 + ControlNet + IP-Adapter + pyrender stack. Flux1 runs
# purely via diffusers (FLUX.1-dev = FluxPipeline; ControlNet = FluxControlNet
# Pipeline), so there is NO official code repo to clone — the HF snapshots are
# everything.
#
# Why clone 'doll'? The shared 'doll' env already has a CUDA torch +
# transformers + accelerate + Pillow. We clone it (so torch is reused, no
# re-download) into a DEDICATED 'flux_human' env, then bump diffusers there.
# This leaves 'doll' untouched — other algos keep working.
#
# flux_human needs:
#   - diffusers >= 0.31   (FluxPipeline + FluxControlNetPipeline + FluxControlNetModel)
#   - transformers        (T5 text encoder for FLUX.1-dev; doll's is fine, upgrade to be safe)
#   - accelerate          (model CPU offload for VRAM)
#   - sentencepiece       (T5 tokenizer; NOT in doll by default — installed here)
#   - protobuf            (T5 tokenizer dep)
#   - Pillow              (save PNG; already in doll)
#   - pyrender + trimesh  (03_render_depth: render SMPL mesh -> depth maps)
#   - opencv-python       (image I/O for depth/condition maps)
#   - einops              (ControlNet tensor ops)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# 00 always operates on the dedicated 'flux_human' env (it creates it if missing).
# Override any inherited CONDA_ENV so we never accidentally upgrade 'doll'.
CONDA_ENV="flux_human"

# 1. Clone 'doll' -> 'flux_human' once (reuse its CUDA torch). Skip if flux_human exists.
if ! conda env list 2>/dev/null | grep -qE "(^| )flux_human( |$)"; then
    if ! conda env list 2>/dev/null | grep -qE "(^| )doll( |$)"; then
        echo "❌ ERROR: source env 'doll' not found." >&2
        echo "   Create it first (e.g. conda create -n doll python=3.10 -y && pip install torch ...) or" >&2
        echo "   clone a different base: BASE_ENV=hunyuanvideo-1.5 bash $SCRIPT_DIR/00_setup_env.sh" >&2
        exit 1
    fi
    BASE_ENV="${BASE_ENV:-doll}"
    echo "📦 cloning conda env '$BASE_ENV' -> 'flux_human' (reuses its CUDA torch) ..."
    conda create -n flux_human --clone "$BASE_ENV" -y
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
    echo "📦 upgrading flux_human runtime deps (diffusers 0.31+ has Flux + FluxControlNet) ..."
    pip install "${PIP_FLAGS[@]}" -U "diffusers>=0.31.0" transformers accelerate \
        sentencepiece protobuf Pillow opencv-python einops
    echo "📦 installing 3D-render deps (pyrender + trimesh for 03_render_depth) ..."
    pip install "${PIP_FLAGS[@]}" pyrender trimesh
    echo "📦 optional: bitsandbytes (4-bit/8bit quant for FLUX.1-dev on consumer GPUs)"
    echo "    install with:  pip install bitsandbytes   # then QUANT=4bit bash 04_generate_views.sh"
fi

# 3. Verify the Flux pipelines import (the whole point of upgrading diffusers).
echo "🔍 verifying FluxPipeline / FluxControlNetPipeline import ..."
python - <<'PY'
try:
    from diffusers import FluxPipeline, FluxControlNetPipeline, FluxControlNetModel  # noqa: F401
    import diffusers
    print(f"✅ diffusers {diffusers.__version__}: FluxPipeline + FluxControlNetPipeline OK")
except ImportError as e:
    raise SystemExit(
        f"❌ Flux pipelines not importable ({e}).\n"
        f"   diffusers >= 0.31 required. Run: INSTALL_DEPS=1 bash flux_human/00_setup_env.sh"
    )
try:
    import pyrender, trimesh  # noqa: F401
    print("✅ pyrender + trimesh OK (for 03_render_depth)")
except ImportError as e:
    print(f"⚠️ pyrender/trimesh not importable ({e}) — 03_render_depth needs them. Run INSTALL_DEPS=1.")
PY

echo "🎉 [00] Done. Env '$CONDA_ENV' ready."
echo "    Next: bash $SCRIPT_DIR/01_download_models.sh   # (or skip if you already have the weights)"
echo "          GPU=0 bash $SCRIPT_DIR/04_generate_views.sh"
