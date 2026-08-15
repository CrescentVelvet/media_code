#!/usr/bin/env bash
# 01_download_models.sh — pull FLUX.1-dev + ControlNet(depth) + IP-Adapter
# diffusers-format snapshots from HuggingFace into the shared model store.
# flux_human runs purely via diffusers (no official code repo to clone).
#
#   1. black-forest-labs/FLUX.1-dev          -> $MODEL_DIR/flux_human/FLUX.1-dev  (GATED)
#        the base FLUX.1-dev pipeline (FluxPipeline: 12B MMDiT + T5 text encoder)
#   2. InstantX/FLUX.1-dev-Controlnet-Union   -> $MODEL_DIR/flux_human/controlnet  (depth+canny+...)
#        diffusers-format ControlNet (FluxControlNetModel); Union variant supports
#        depth/canny/pose/etc via a control mode flag. Override CONTROLNET_REPO
#        to use a different one (e.g. xlabs/flux-controlnet-collections).
#   3. xlabs/flux-ip-adapter                  -> $MODEL_DIR/flux_human/ip-adapter (OPTIONAL)
#        reference-image conditioning (lock the subject's appearance across views).
#        Skip with SKIP_IPADAPTER=1.
#
# ⚠️  GATED ACCESS — FLUX.1-dev is gated (FLUX.1 Non-Commercial License):
#    `hf download` without a token returns 401 / "not found". You MUST
#      1) accept the license at https://huggingface.co/black-forest-labs/FLUX.1-dev
#      2) pass HF_TOKEN (a read token from https://huggingface.co/settings/tokens)
#    The ControlNet / IP-Adapter repos themselves are public, but they're built on
#    FLUX.1-dev so you still need the base model's access + token for the base.
#
# Override which models: SKIP_CONTROLNET=1 / SKIP_IPADAPTER=1 / MODELS_TO_DOWNLOAD.
# Falls back to an SSL-bypass downloader if the CDN MITM cert can't be verified.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/flux_human}"
FLUX_REPO="${FLUX_REPO:-black-forest-labs/FLUX.1-dev}"
CONTROLNET_REPO="${CONTROLNET_REPO:-InstantX/FLUX.1-dev-Controlnet-Union}"
IPADAPTER_REPO="${IPADAPTER_REPO:-xlabs/flux-ip-adapter}"

# Local dest dirs.
FLUX_DIR="$MODEL_DIR/FLUX.1-dev"
CONTROLNET_DIR="$MODEL_DIR/controlnet"
IPADAPTER_DIR="$MODEL_DIR/ip-adapter"

echo "=== [01] Downloading flux_human weights (FLUX.1-dev + ControlNet + IP-Adapter) ==="
echo "  模型根目录:    $MODEL_DIR"
echo "  base (dev):    $FLUX_REPO -> $FLUX_DIR  (GATED — needs HF_TOKEN)"
echo "  controlnet:     $CONTROLNET_REPO -> $CONTROLNET_DIR"
[ "${SKIP_IPADAPTER:-0}" != "1" ] && echo "  ip-adapter:     $IPADAPTER_REPO -> $IPADAPTER_DIR"
echo "  HF_TOKEN:       ${HF_TOKEN:+已设置}${HF_TOKEN:-❌ 未设置 (dev 是 gated, 需要!)}"

# Gated-repo guard: FLUX.1-dev base is gated.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "❌ ERROR: FLUX.1-dev is GATED and HF_TOKEN is not set." >&2
    echo "       1) accept the license at https://huggingface.co/black-forest-labs/FLUX.1-dev" >&2
    echo "       2) create a read token at https://huggingface.co/settings/tokens" >&2
    echo "       3) HF_TOKEN=<token> bash $SCRIPT_DIR/01_download_models.sh" >&2
    exit 1
fi

if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "❌ ERROR: huggingface_hub not installed in env '$CONDA_ENV'. Install with: pip install huggingface_hub" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"

# --- download one HF repo with SSL-bypass fallback (mirrors flux2 01) ---
# hf_get <repo_id> <local_dir> [include_patterns_csv] [token]
hf_get() {
    local repo="$1" dir="$2" includes="${3:-}" token="${4:-}"
    mkdir -p "$dir"
    local token_args=(); [ -n "$token" ] && token_args=(--token "$token")
    local include_args=()
    if [ -n "$includes" ]; then
        local parts=()
        IFS=',' read -ra parts <<< "$includes"
        for p in "${parts[@]}"; do include_args+=(--include "$p"); done
    fi
    if [ "${HF_DISABLE_SSL:-0}" = "1" ]; then
        echo "--- [$repo] downloading (SSL verification DISABLED via HF_DISABLE_SSL=1) ---"
        python "$SCRIPT_DIR/_hf_download.py" "$repo" "$dir" "$includes" "$token"
    else
        if ! hf download "$repo" "${token_args[@]}" "${include_args[@]}" --local-dir "$dir"; then
            echo "--- [$repo] hf download failed (likely SSL on CDN); retrying with SSL verification disabled ---"
            python "$SCRIPT_DIR/_hf_download.py" "$repo" "$dir" "$includes" "$token"
        fi
    fi
}

# 1. FLUX.1-dev base (gated). Whole diffusers snapshot (model_index.json +
#    transformer + vae + text_encoder + scheduler).
echo "--- [1/3] base pipeline: $FLUX_REPO ---"
hf_get "$FLUX_REPO" "$FLUX_DIR" "" "${HF_TOKEN:-}"
if [ -f "$FLUX_DIR/model_index.json" ]; then
    echo "✅ base pipeline OK: $FLUX_DIR"
else
    echo "⚠️ WARNING: $FLUX_DIR/model_index.json not found — snapshot may be incomplete." >&2
    echo "         Check HF_TOKEN / network and rerun." >&2
fi

# 2. ControlNet (depth). diffusers-format FluxControlNetModel.
if [ "${SKIP_CONTROLNET:-0}" != "1" ]; then
    echo "--- [2/3] controlnet: $CONTROLNET_REPO ---"
    hf_get "$CONTROLNET_REPO" "$CONTROLNET_DIR" "" "${HF_TOKEN:-}"
    if [ -f "$CONTROLNET_DIR/config.json" ] || [ -f "$CONTROLNET_DIR/model_index.json" ]; then
        echo "✅ controlnet OK: $CONTROLNET_DIR"
    else
        echo "⚠️ WARNING: $CONTROLNET_DIR has no config.json/model_index.json — may be wrong format." >&2
        echo "         (Union repo uses subfolders; 04 loads via FluxControlNetModel.from_pretrained)" >&2
    fi
else
    echo "--- [2/3] SKIPPED controlnet (SKIP_CONTROLNET=1) ---"
fi

# 3. IP-Adapter (optional; reference-image appearance lock).
if [ "${SKIP_IPADAPTER:-0}" != "1" ]; then
    echo "--- [3/3] ip-adapter: $IPADAPTER_REPO ---"
    hf_get "$IPADAPTER_REPO" "$IPADAPTER_DIR" "" "${HF_TOKEN:-}"
    echo "✅ ip-adapter downloaded: $IPADAPTER_DIR"
else
    echo "--- [3/3] SKIPPED ip-adapter (SKIP_IPADAPTER=1) ---"
fi

echo "--- downloaded files (depth 1) ---"
find "$MODEL_DIR" -maxdepth 2 -mindepth 1 \( -type f -o -type d \) -printf '  %P\n' 2>/dev/null | sort | head -60

echo "=== [01] Done. Weights under: $MODEL_DIR ==="
echo "    Next: GPU=0 bash $SCRIPT_DIR/02_extract_smpl.sh  (SMPL 提取, 调 sam_3d_body)"
echo "          GPU=0 bash $SCRIPT_DIR/04_generate_views.sh (Flux1 多视角生成)"
echo "    ⚠️ sam_3d_body 权重需单独下: HF_TOKEN=hf_xxx bash sam_3d_body/01_download_models.sh"
