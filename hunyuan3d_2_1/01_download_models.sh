#!/usr/bin/env bash
# 01_download_models.sh — pull Hunyuan3D weights into the shared model store.
# Pipelines load via from_pretrained(<local dir>), so no symlink into the
# official repo is needed.
#
#   tencent/Hunyuan3D-2.1   -> shape-v2-1 + paint-v2-1 (single-image shape)
#   tencent/Hunyuan3D-2mv   -> shape-v2-mv  (multi-view shape, 2.0-gen weights,
#                              run by the 2.1 codebase; set SKIP_MV=1 to skip)
#   facebook/dinov2-giant   -> image encoder used by the paint stage
#
# Two sources: HuggingFace (hf download) and ModelScope (modelscope download).
# HF is flaky/blocked on some networks (WSL 实测 CDN resolve 会挂), so each repo
# falls back HF -> ModelScope automatically. ModelScope repo ids differ only in
# the org name (Tencent-Hunyuan with capital T).
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SKIP_MV="${SKIP_MV:-0}"

hf_ok() {
    [ "${HF_DISABLE_SSL:-0}" = "1" ] && return 1
    curl -s -o /dev/null --connect-timeout 8 -I "https://huggingface.co" 2>/dev/null
}

download_repo() {
    local hf_id="$1" ms_id="$2" dst="$3"
    echo "📦 [01] $hf_id -> $dst"
    mkdir -p "$dst"
    if [ "${HF_DISABLE_SSL:-0}" = "1" ]; then
        python "$SCRIPT_DIR/_hf_download.py" "$hf_id" "$dst" && return 0
    elif hf_ok; then
        # Public repos — no token needed. Corporate-proxy CDN endpoints may
        # present MITM certs, so fall back to the no-verify downloader.
        if hf download "$hf_id" --local-dir "$dst"; then
            return 0
        fi
        echo "⚠️ hf download failed (likely SSL on CDN); retrying no-verify"
        if python "$SCRIPT_DIR/_hf_download.py" "$hf_id" "$dst"; then
            return 0
        fi
    else
        echo "⚠️ huggingface.co unreachable — skipping HF"
    fi
    # Fallback: ModelScope (China-friendly CDN).
    if ! python -c "import modelscope" 2>/dev/null; then
        echo "📦 installing modelscope CLI"
        pip install modelscope || return 1
    fi
    echo "📦 modelscope download $ms_id"
    modelscope download --model "$ms_id" --local_dir "$dst"
}

if ! python -c "import huggingface_hub" 2>/dev/null; then
    pip install huggingface_hub || {
        echo "❌ huggingface_hub install failed" >&2; exit 1; }
fi

download_repo "${SHAPE_REPO_ID:-tencent/Hunyuan3D-2.1}" \
    "${SHAPE_MS_ID:-Tencent-Hunyuan/Hunyuan3D-2.1}" "$MODEL_DIR/Hunyuan3D-2.1" || {
    echo "❌ Hunyuan3D-2.1 download failed" >&2; exit 1; }
if [ "$SKIP_MV" != "1" ]; then
    download_repo "${MV_REPO_ID:-tencent/Hunyuan3D-2mv}" \
        "${MV_MS_ID:-Tencent-Hunyuan/Hunyuan3D-2mv}" "$MODEL_DIR/Hunyuan3D-2mv" || {
        echo "❌ Hunyuan3D-2mv download failed" >&2; exit 1; }
fi
download_repo "${DINO_REPO_ID:-facebook/dinov2-giant}" \
    "${DINO_MS_ID:-facebook/dinov2-giant}" "$MODEL_DIR/dinov2-giant" || {
    echo "❌ dinov2-giant download failed" >&2; exit 1; }

echo "=== [01] downloaded files (depth 2) ==="
find "$MODEL_DIR/Hunyuan3D-2.1" -maxdepth 2 -type f -printf '  %P\n' 2>/dev/null | sort

echo "🎉 [01] Done. Weights under: $MODEL_DIR"
