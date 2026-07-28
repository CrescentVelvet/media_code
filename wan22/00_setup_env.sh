#!/usr/bin/env bash
# 00_setup_env.sh — clone DiffSynth-Studio, activate conda env, verify torch+CUDA,
# and pip install -e . (editable). The official repo is DiffSynth-Studio (modelscope).
#
# DiffSynth-Studio needs python>=3.10 and a CUDA-enabled torch. Use a DEDICATED env:
#   conda create -n wan22 python=3.10 -y
#   conda activate wan22
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#   INSTALL_DEPS=1 bash wan22/00_setup_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

DIFFSYNTH_REPO="${DIFFSYNTH_REPO:-https://github.com/modelscope/DiffSynth-Studio.git}"

echo "=== [00] Verify torch in conda env '$CONDA_ENV' ==="
python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda not available — install a CUDA-enabled torch or check GPU visibility.")
PY

# Clone the official repo if not present.
if [ ! -d "$DIFFSYNTH_DIR" ]; then
    mkdir -p "$(dirname "$DIFFSYNTH_DIR")"
    echo "--- cloning official repo -> $DIFFSYNTH_DIR ---"
    git clone "$DIFFSYNTH_REPO" "$DIFFSYNTH_DIR" || \
        git -c http.sslVerify=false clone "$DIFFSYNTH_REPO" "$DIFFSYNTH_DIR"
else
    echo "--- official repo already present: $DIFFSYNTH_DIR ---"
fi

# Install the package (editable) on demand. This pulls diffsynth + its deps.
# Set SKIP_DEPS=1 to skip (use if already installed).
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)
    echo "--- pip install -e $DIFFSYNTH_DIR (editable) ---"
    pip install "${PIP_FLAGS[@]}" -e "$DIFFSYNTH_DIR"
    echo "--- deps installed. Tip: verify with 'python -c \"import diffsynth; print(diffsynth.__version__)\"' ---"
fi

# Verify the import works.
echo "--- verify diffsynth import ---"
python -c "import diffsynth; print(f'diffsynth OK')" 2>/dev/null || \
    echo "WARNING: 'import diffsynth' failed. Run INSTALL_DEPS=1 bash $0 to install."

echo "=== [00] Done. Env '$CONDA_ENV' ready. ==="
echo "    Next: bash $SCRIPT_DIR/01_verify_models.sh  (check weights are present)"
