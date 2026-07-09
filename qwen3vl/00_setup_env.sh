#!/usr/bin/env bash
# 00_setup_env.sh — activate conda env (torch preinstalled) & verify torch+CUDA.
# Set INSTALL_DEPS=1 to install the Qwen3-VL runtime deps (no official repo to
# clone — Qwen3-VL runs purely via transformers + qwen-vl-utils).
#
# Qwen3-VL needs: transformers (recent, >=4.55 — Qwen3-VL support landed only
# in very recent versions), qwen-vl-utils (process_vision_info), accelerate
# (device_map="auto" / multi-GPU sharding), Pillow. These are usually
# compatible with the shared 'doll' env (hunyuanvideo-1.5 already pulls
# transformers==4.57.1). If your doll env has an OLDER transformers, either
# `pip install -U transformers` or use a dedicated env (CONDA_ENV=qwen3vl).
#
# SKIP_TORCH=1 is a no-op here (we don't pin torch) but kept for consistency.
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
# install the minimal set directly. transformers is upgraded to the latest
# compatible release so Qwen3-VL model classes are registered.
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org \
        --timeout 600 --retries 10)
    echo "--- installing Qwen3-VL runtime deps ---"
    pip install "${PIP_FLAGS[@]}" -U "transformers>=4.55" accelerate qwen-vl-utils Pillow
    # Optional: bitsandbytes for 4/8-bit loading (LOAD_IN_4BIT/8BIT). Off by
    # default — install separately if you want quantized inference:
    echo "--- optional: bitsandbytes (for 4/8-bit quantized inference) ---"
    echo "    pip install bitsandbytes    # then LOAD_IN_4BIT=1 bash 02_run_inference.sh"
    echo "--- deps installed. Verify with: python -c \"import transformers,qwen_vl_utils,accelerate; print(transformers.__version__)\" ---"
fi

echo "=== [00] Done. Env '$CONDA_ENV' ready. (Missing a dep? INSTALL_DEPS=1 bash this, or pip install it) ==="
