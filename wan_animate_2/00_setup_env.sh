#!/usr/bin/env bash
# 00_setup_env.sh — clone Wan-Animate-2 (recursive), verify torch+CUDA, install
# deps (requirements.txt + flash-attn + pip install -e .) when INSTALL_DEPS=1.
#
# Wan-Animate-2 needs python>=3.10 and a CUDA-enabled torch (cu126). Use a
# DEDICATED env:
#   conda create -n wan_animate_2 python=3.11 -y
#   conda activate wan_animate_2
#   pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
#   INSTALL_DEPS=1 bash wan_animate_2/00_setup_env.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

OFFICIAL_REPO="${OFFICIAL_REPO:-https://github.com/Wan-Video/Wan-Animate-2.git}"

echo "=== [00] Verify torch in conda env '$CONDA_ENV' ==="
python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda not available — install a CUDA-enabled torch or check GPU visibility.")
PY

# Clone the official repo (recursive for submodules) if not present.
# LD_LIBRARY_PATH= avoids conda libffi/system libp11-kit clash crashing system git.
if [ ! -d "$OFFICIAL_DIR" ]; then
    mkdir -p "$(dirname "$OFFICIAL_DIR")"
    echo "📦 cloning official repo -> $OFFICIAL_DIR"
    LD_LIBRARY_PATH= git clone --recursive "$OFFICIAL_REPO" "$OFFICIAL_DIR" || \
        LD_LIBRARY_PATH= git -c http.sslVerify=false clone --recursive "$OFFICIAL_REPO" "$OFFICIAL_DIR"
else
    echo "--- official repo already present: $OFFICIAL_DIR ---"
fi

# Install deps on demand. torch/torchvision/torchaudio are pre-installed from the
# pytorch index (download.pytorch.org is 403 behind the corporate proxy), so they
# are filtered out of requirements.txt to avoid re-resolution failures.
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)
    REQ="$OFFICIAL_DIR/requirements.txt"
    if [ -f "$REQ" ]; then
        echo "📦 pip install (non-torch deps from requirements.txt)"
        grep -v -iE '^(torch|torchvision|torchaudio)==' "$REQ" | \
            pip install "${PIP_FLAGS[@]}" -r /dev/stdin
    else
        echo "⚠️ requirements.txt not found at $REQ — skipping"
    fi
    echo "📦 pip install flash-attn --no-build-isolation"
    pip install "${PIP_FLAGS[@]}" flash-attn --no-build-isolation || \
        echo "⚠️ flash-attn install failed — install manually: pip install flash-attn --no-build-isolation"
    echo "📦 pip install -e $OFFICIAL_DIR (editable)"
    pip install "${PIP_FLAGS[@]}" -e "$OFFICIAL_DIR"
    echo "✅ deps installed. Verify: python -c 'import core; print(\"core OK\")'"
fi

# Verify imports.
echo "🔍 verify imports (core / pipelines)"
python -c "import core; print('core OK')" 2>/dev/null || \
    echo "⚠️ 'import core' failed. Run: INSTALL_DEPS=1 bash $0"
python -c "from core import build_object_from_config_file; print('build_object_from_config_file OK')" 2>/dev/null || true

echo "=== [00] Done. Env '$CONDA_ENV' ready. ==="
echo "    Next: bash $SCRIPT_DIR/01_download_models.sh  (download weights)"
