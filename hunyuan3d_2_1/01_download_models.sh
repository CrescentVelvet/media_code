#!/usr/bin/env bash
# 01_download_models.sh — pull Hunyuan3D weights from HuggingFace into the
# shared model store. Pipelines load via from_pretrained(<local dir>), so no
# symlink into the official repo is needed.
#
#   tencent/Hunyuan3D-2.1   -> shape-v2-1 + paint-v2-1 (single-image shape)
#   tencent/Hunyuan3D-2mv   -> shape-v2-mv  (multi-view shape, 2.0-gen weights,
#                              run by the 2.1 codebase; set SKIP_MV=1 to skip)
#   facebook/dinov2-giant   -> image encoder used by the paint stage
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SKIP_MV="${SKIP_MV:-0}"

download_repo() {
    local repo_id="$1" dst="$2"
    echo "📦 [01] $repo_id -> $dst"
    mkdir -p "$dst"
    # Public repos — no token needed. Corporate-proxy CDN endpoints may present
    # MITM certs the trust store can't verify, so fall back to a downloader
    # with SSL verification disabled (proxy already MITMs everything).
    if [ "${HF_DISABLE_SSL:-0}" = "1" ]; then
        python "$SCRIPT_DIR/_hf_download.py" "$repo_id" "$dst"
    elif ! hf download "$repo_id" --local-dir "$dst"; then
        echo "⚠️ hf download failed (likely SSL on CDN); retrying with SSL verification disabled"
        python "$SCRIPT_DIR/_hf_download.py" "$repo_id" "$dst"
    fi
}

if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "❌ huggingface_hub not installed in env '$CONDA_ENV'." >&2
    echo "   pip install huggingface_hub" >&2
    exit 1
fi

download_repo "${SHAPE_REPO_ID:-tencent/Hunyuan3D-2.1}" "$MODEL_DIR/Hunyuan3D-2.1"
if [ "$SKIP_MV" != "1" ]; then
    download_repo "${MV_REPO_ID:-tencent/Hunyuan3D-2mv}" "$MODEL_DIR/Hunyuan3D-2mv"
fi
download_repo "${DINO_REPO_ID:-facebook/dinov2-giant}" "$MODEL_DIR/dinov2-giant"

echo "=== [01] downloaded files (depth 2) ==="
find "$MODEL_DIR/Hunyuan3D-2.1" -maxdepth 2 -type f -printf '  %P\n' 2>/dev/null | sort

echo "🎉 [01] Done. Weights under: $MODEL_DIR"
