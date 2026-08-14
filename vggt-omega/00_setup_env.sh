#!/usr/bin/env bash
# 00_setup_env.sh — clone official repo + activate conda env + verify torch+CUDA.
# Reuses an existing env (torch preinstalled) to avoid re-downloading torch. No venv.
# The vggt_omega package is imported via sys.path (run_batch.py adds VGGT_DIR),
# so `pip install -e .` is NOT required (but still works if you prefer it).
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "=== [00] Setup VGGT-Omega env '$CONDA_ENV' ==="

# Clone the official repo if not present yet.
if [ ! -d "$VGGT_DIR" ]; then
    echo "--- cloning official repo -> $VGGT_DIR ---"
    mkdir -p "$(dirname "$VGGT_DIR")"
    git clone "$VGGT_REPO" "$VGGT_DIR" || \
        git -c http.sslVerify=false clone "$VGGT_REPO" "$VGGT_DIR"
else
    echo "--- official repo already present: $VGGT_DIR ---"
fi

# Verify torch + CUDA in the conda env.
python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda not available in this env — install a CUDA-enabled torch or check GPU visibility.")
if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 3):
    print(f"WARNING: VGGT-Omega requires torch>=2.3 (have {torch.__version__}).")
PY

# Deps are installed ON DEMAND by default. Set INSTALL_DEPS=1 to install the
# known runtime set in one shot (small packages; torch is NOT reinstalled).
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)
    echo "--- installing known runtime deps ---"
    # core (vggt_omega): numpy<2, Pillow, einops, safetensors, opencv-python
    # point-cloud export (predictions_to_glb): scipy, trimesh, matplotlib, tqdm
    # 03 render (ply -> mp4): plyfile, imageio, imageio-ffmpeg, gsplat
    pip install "${PIP_FLAGS[@]}" "numpy<2" Pillow einops safetensors opencv-python \
        scipy trimesh matplotlib tqdm huggingface_hub \
        plyfile imageio imageio-ffmpeg gsplat
fi

echo "=== [00] Done. Env '$CONDA_ENV' ready, official repo cloned. (Missing a dep? pip install it, or rerun with INSTALL_DEPS=1) ==="
