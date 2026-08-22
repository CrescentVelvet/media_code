#!/usr/bin/env bash
# 00_setup_env.sh — clone ArtiFixer (+ 3DGRUT submodule), install deps, verify imports.
#
# ArtiFixer needs: torch>=2.1, diffusers==0.37.1, transformers==5.5.0, accelerate==1.13.0,
# 3DGRUT-ArtiFixer (slangc + slangtorch), MoGe, and various utils.
# On A100 the code auto-selects cuDNN SDPA attention — NO flash-attn 3/4 needed.
#
# Usage:
#   conda create -n artifixer python=3.12 -y && conda activate artifixer
#   pip install torch torchvision  # CUDA build from PyPI (or download.pytorch.org if not behind proxy)
#   INSTALL_DEPS=1 bash artifixer/00_setup_env.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

ARTIFIXER_REPO="${ARTIFIXER_REPO:-https://github.com/nv-tlabs/ArtiFixer.git}"

echo "🚀 [00] Setup ArtiFixer environment"
echo "  📁 conda env:  $CONDA_ENV"
echo "  📁 code dir:   $ARTIFIXER_DIR"
echo "  📁 model dir:  $MODEL_DIR"
echo "  🤖 variant:    $MODEL_VARIANT  (model_id: $WAN_MODEL_ID)"

# --- verify torch ---
echo "--- verify torch ---"
python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda not available — install a CUDA-enabled torch or check GPU visibility.")
cap = torch.cuda.get_device_capability()
print(f"GPU capability: sm_{cap[0]}{cap[1]}")
PY

# --- clone ArtiFixer with submodules ---
if [ ! -d "$ARTIFIXER_DIR" ]; then
    mkdir -p "$(dirname "$ARTIFIXER_DIR")"
    echo "📦 cloning ArtiFixer (+ 3DGRUT submodule) -> $ARTIFIXER_DIR"
    LD_LIBRARY_PATH= git clone --recurse-submodules "$ARTIFIXER_REPO" "$ARTIFIXER_DIR" || \
        LD_LIBRARY_PATH= git -c http.sslVerify=false clone --recurse-submodules "$ARTIFIXER_REPO" "$ARTIFIXER_DIR"
else
    echo "⏭️  ArtiFixer already present: $ARTIFIXER_DIR"
    # ensure submodule is initialised
    if [ ! -f "$ARTIFIXER_DIR/thirdparty/3DGRUT-ArtiFixer/setup.py" ] && \
       [ ! -f "$ARTIFIXER_DIR/thirdparty/3DGRUT-ArtiFixer/pyproject.toml" ]; then
        echo "📦 initialising 3DGRUT submodule..."
        (cd "$ARTIFIXER_DIR" && LD_LIBRARY_PATH= git submodule update --init --recursive)
    fi
fi

# --- install dependencies ---
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)

    echo "📦 installing ArtiFixer Python deps..."
    pip install "${PIP_FLAGS[@]}" \
        "diffusers==0.37.1" "transformers==5.5.0" "accelerate==1.13.0" ftfy \
        einops scipy wandb tqdm Pillow matplotlib pyyaml \
        torchmetrics imageio-ffmpeg h5py av torch-fidelity

    echo "📦 installing 3DGRUT-ArtiFixer (slangc + slangtorch + deps)..."
    # install slangc compiler
    (cd "$ARTIFIXER_DIR/thirdparty/3DGRUT-ArtiFixer" && bash scripts/install_slangc.sh /usr/local)
    # install 3DGRUT requirements + package
    pip install "${PIP_FLAGS[@]}" -r "$ARTIFIXER_DIR/thirdparty/3DGRUT-ArtiFixer/requirements.txt"
    pip install "${PIP_FLAGS[@]}" -e "$ARTIFIXER_DIR/thirdparty/3DGRUT-ArtiFixer"

    echo "📦 installing MoGe..."
    pip install "${PIP_FLAGS[@]}" "git+https://github.com/microsoft/MoGe.git"

    # version pins (last, overrides dependency upgrades)
    pip install --force-reinstall --no-deps "numpy<2.0"
    pip install --force-reinstall --no-deps "setuptools<72.1.0"

    echo "✅ deps installed."
fi

# --- verify imports ---
echo "--- verify imports ---"
python - <<'PY'
import diffusers; print(f"diffusers {diffusers.__version__}")
import transformers; print(f"transformers {transformers.__version__}")
import accelerate; print(f"accelerate {accelerate.__version__}")
import h5py; print("h5py ok")
import einops; print("einops ok")
try:
    import threedgrut; print("threedgrut ok")
except ImportError as e:
    print(f"⚠️ threedgrut import failed: {e}")
try:
    from moge.model.v2 import MoGeModel; print("MoGe ok")
except Exception as e:
    print(f"⚠️ MoGe import failed: {e}")
PY

echo "🎉 [00] Done. Env '$CONDA_ENV' ready."
echo "    Next: bash $SCRIPT_DIR/01_download_models.sh  (download weights)"
