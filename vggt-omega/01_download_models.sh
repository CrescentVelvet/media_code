#!/usr/bin/env bash
# 01_download_models.sh — download the VGGT-Omega checkpoint from HuggingFace.
#
# IMPORTANT: the HF repo `facebook/VGGT-Omega` is GATED.
#   1. Request access at https://huggingface.co/facebook/VGGT-Omega (auto-reviewed).
#   2. Create a read token: https://huggingface.co/settings/tokens
#   3. Put `export HF_TOKEN=hf_xxx` in proxy.env (repo root, gitignored).
# Only the requested checkpoint file is downloaded (saves bandwidth).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

VGGT_DIR="${VGGT_DIR:-$REPO_DIR/../vggt-omega}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../model/VGGT-Omega}"
HF_REPO_ID="${HF_REPO_ID:-facebook/VGGT-Omega}"

# Which checkpoint to fetch. VARIANT selects the file; 02_run_inference.sh
# reads the same VARIANT to know which one to load.
VARIANT="${VARIANT:-1b_512}"
case "$VARIANT" in
    1b_512)      CKPT_FILE="vggt_omega_1b_512.pt" ;;
    1b_256_text) CKPT_FILE="vggt_omega_1b_256_text.pt" ;;
    all)         CKPT_FILE="" ;;   # download every file
    *) echo "ERROR: VARIANT must be 1b_512 | 1b_256_text | all (got '$VARIANT')" >&2; exit 1 ;;
esac

echo "=== [01] Downloading VGGT-Omega checkpoint ==="
echo "  HF repo:  $HF_REPO_ID  (GATED — needs HF_TOKEN + access granted)"
echo "  variant:  $VARIANT"
echo "  file:     ${CKPT_FILE:-<all>}"
echo "  model:    $MODEL_DIR"

if [ ! -d "$VGGT_DIR" ]; then
    echo "ERROR: VGGT-Omega code dir not found at $VGGT_DIR." >&2
    echo "       Run run_all.sh first (it clones the official repo), or set VGGT_DIR." >&2
    exit 1
fi
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed in env '$CONDA_ENV'. Install: pip install huggingface_hub" >&2
    exit 1
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set — the VGGT-Omega repo is gated." >&2
    echo "       1. Request access: https://huggingface.co/facebook/VGGT-Omega" >&2
    echo "       2. Create a read token: https://huggingface.co/settings/tokens" >&2
    echo "       3. Add 'export HF_TOKEN=hf_xxx' to $(dirname "$REPO_DIR" 2>/dev/null || echo repo-root)/proxy.env" >&2
    echo "          (proxy.env is at the repo root next to this folder; it is gitignored)" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"
export HF_TOKEN   # huggingface_hub / hf CLI read this automatically.

# The CDN endpoints (us.aws.cdn.hf.co) present MITM certs the trust store
# can't verify, so fall back to a downloader with SSL verification disabled.
# Set HF_DISABLE_SSL=1 to skip the normal attempt and go straight to it.
if [ "${HF_DISABLE_SSL:-0}" = "1" ]; then
    echo "--- downloading (SSL verification DISABLED via HF_DISABLE_SSL=1) ---"
    python "$SCRIPT_DIR/_hf_download.py" "$HF_REPO_ID" "$MODEL_DIR" "${CKPT_FILE}"
elif [ -n "$CKPT_FILE" ]; then
    if ! hf download "$HF_REPO_ID" "$CKPT_FILE" --local-dir "$MODEL_DIR"; then
        echo "--- hf download failed; retrying with SSL verification disabled ---"
        python "$SCRIPT_DIR/_hf_download.py" "$HF_REPO_ID" "$MODEL_DIR" "$CKPT_FILE"
    fi
else
    if ! hf download "$HF_REPO_ID" --local-dir "$MODEL_DIR"; then
        echo "--- hf download failed; retrying with SSL verification disabled ---"
        python "$SCRIPT_DIR/_hf_download.py" "$HF_REPO_ID" "$MODEL_DIR"
    fi
fi

echo "--- downloaded files ---"
find "$MODEL_DIR" -maxdepth 2 -type f -printf '  %P\n' | sort

echo "=== [01] Done. Checkpoint at: $MODEL_DIR  (VARIANT=$VARIANT) ==="
