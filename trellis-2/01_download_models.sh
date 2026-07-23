#!/usr/bin/env bash
# 01_download_models.sh — pull TRELLIS.2-4B weights from HuggingFace into the
# shared model store. The pipeline loads them via from_pretrained(<local path>),
# so no symlink into the official repo is needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/TRELLIS.2-4B}"
HF_REPO_ID="${HF_REPO_ID:-microsoft/TRELLIS.2-4B}"

echo "=== [01] Downloading TRELLIS.2-4B weights ==="
echo "  HF repo:  $HF_REPO_ID"
echo "  model:    $MODEL_DIR"

if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed in env '$CONDA_ENV'. Install with: pip install huggingface_hub" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"

# Public MIT model — no token required. The snapshot covers every file the
# pipeline needs (pipeline.json + the 8 model weight dirs + configs). The CDN
# endpoints (us.aws.cdn.hf.co) present MITM certs the trust store can't verify,
# so fall back to a downloader with SSL verification disabled. Set
# HF_DISABLE_SSL=1 to skip the normal attempt and go straight to it.
if [ "${HF_DISABLE_SSL:-0}" = "1" ]; then
    echo "--- downloading (SSL verification DISABLED via HF_DISABLE_SSL=1) ---"
    python "$SCRIPT_DIR/_hf_download.py" "$HF_REPO_ID" "$MODEL_DIR"
elif ! hf download "$HF_REPO_ID" --local-dir "$MODEL_DIR"; then
    echo "--- hf download failed (likely SSL on CDN); retrying with SSL verification disabled ---"
    python "$SCRIPT_DIR/_hf_download.py" "$HF_REPO_ID" "$MODEL_DIR"
fi

echo "--- downloaded files ---"
find "$MODEL_DIR" -maxdepth 2 -type f -printf '  %P\n' | sort

echo "=== [01] Done. Weights at: $MODEL_DIR ==="
echo "    (pipeline loads via Trellis2ImageTo3DPipeline.from_pretrained(\"$MODEL_DIR\"))"
