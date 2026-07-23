#!/usr/bin/env bash
# 01_download_models.sh — pull SAM 3D Body weights into the shared model store:
#   1. facebook/sam-3d-body-dinov3  -> $MODEL_DIR/sam-3d-body-dinov3  (GATED)
#        contains: model.ckpt, model_config.yaml, assets/mhr_model.pt
#   2. Ruicheng/moge-2-vitl-normal  -> $MODEL_DIR/moge-2-vitl-normal  (public)
#        the default MoGe2 FOV estimator weights
#
# ⚠️  GATED ACCESS — the SAM 3D Body checkpoint repos require you to:
#    1) Open https://huggingface.co/facebook/sam-3d-body-dinov3 and click
#       "Request access" (usually auto-approved within minutes). Repeat for
#       `facebook/sam-3d-body-vith` if you want the ViT-H backbone too.
#    2) Create a *read* token at https://huggingface.co/settings/tokens and
#       export it: HF_TOKEN=hf_xxx bash sam_3d_body/01_download_models.sh
#    Without access + token, HF returns 401/"repository not found".
#
# Override the backbone with HF_REPO_ID=facebook/sam-3d-body-vith (ViT-H, 631M).
# The default dinov3 backbone (DINOv3-H+, 840M) is the recommended/best one.
# Set SKIP_FOV=1 to skip the MoGe2 FOV download (then inference falls back to
# runtime download, or set FOV_NAME= to disable the FOV estimator entirely).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SAM3D_DIR="${SAM3D_DIR:-$REPO_DIR/../sam-3d-body}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/sam-3d-body}"

HF_REPO_ID="${HF_REPO_ID:-facebook/sam-3d-body-dinov3}"
HF_FOV_REPO="${HF_FOV_REPO:-Ruicheng/moge-2-vitl-normal}"

# Local dirs derived from the repo basename (works for both dinov3 and vith).
CKPT_DIR="$MODEL_DIR/$(basename "$HF_REPO_ID")"
FOV_MODEL_DIR="$MODEL_DIR/$(basename "$HF_FOV_REPO")"

CHECKPOINT_PATH="$CKPT_DIR/model.ckpt"
MHR_PATH="$CKPT_DIR/assets/mhr_model.pt"
MODEL_CONFIG="$CKPT_DIR/model_config.yaml"

echo "=== [01] Downloading SAM 3D Body weights ==="
echo "  model dir:   $MODEL_DIR"
echo "  code dir:    $SAM3D_DIR"
echo "  main ckpt:   $HF_REPO_ID -> $CKPT_DIR  (GATED — needs HF_TOKEN)"
echo "  fov (moge2): $HF_FOV_REPO -> $FOV_MODEL_DIR"

if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed in env '$CONDA_ENV'. Install with: pip install huggingface_hub" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR" "$CKPT_DIR" "$FOV_MODEL_DIR"

# --- download one HF repo with SSL-bypass fallback (mirrors hypir 01) ---
# hf_get <repo_id> <local_dir> [allow_patterns_csv] [token]
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
            echo "--- [$repo] hf download failed (likely SSL on CDN, or 401 on gated repo); retrying with SSL verification disabled ---"
            python "$SCRIPT_DIR/_hf_download.py" "$repo" "$dir" "$includes" "$token"
        fi
    fi
}

# 1. SAM 3D Body main checkpoint (gated). Whole repo is small (model.ckpt +
#    model_config.yaml + assets/mhr_model.pt) — fetch in full.
if [ "${SKIP_MAIN:-0}" != "1" ]; then
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "WARNING: HF_TOKEN is empty. The SAM 3D Body checkpoint is GATED —" >&2
        echo "         a 401 / 'repository not found' error means you have NOT" >&2
        echo "         requested access or not passed a read token. See INSTALL.md." >&2
        echo "         Request access + create a token, then:" >&2
        echo "           HF_TOKEN=hf_xxx bash sam_3d_body/01_download_models.sh" >&2
    fi
    echo "--- [1/2] main checkpoint: $HF_REPO_ID ---"
    hf_get "$HF_REPO_ID" "$CKPT_DIR" "" "${HF_TOKEN:-}"
else
    echo "--- [1/2] SKIPPED main checkpoint (SKIP_MAIN=1) ---"
fi

# 2. MoGe2 FOV estimator (public; auto-downloads at runtime too, but
#    pre-fetching avoids flaky proxy downloads during inference).
if [ "${SKIP_FOV:-0}" != "1" ]; then
    echo "--- [2/2] fov estimator: $HF_FOV_REPO ---"
    hf_get "$HF_FOV_REPO" "$FOV_MODEL_DIR" "" "${HF_TOKEN:-}"
else
    echo "--- [2/2] SKIPPED fov estimator (SKIP_FOV=1) ---"
fi

echo "--- downloaded files ---"
echo "  main ckpt:"; find "$CKPT_DIR" -maxdepth 2 -mindepth 1 -printf '    %P\n' 2>/dev/null | sort
echo "  fov dir:";   find "$FOV_MODEL_DIR" -maxdepth 1 -mindepth 1 -printf '    %P\n' 2>/dev/null | sort

# --- verify the key files exist (warn, don't fail, so a partial run can resume) ---
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "--- main ckpt OK: $CHECKPOINT_PATH ($(du -h "$CHECKPOINT_PATH" | cut -f1)) ---"
else
    echo "WARNING: model.ckpt not found at $CHECKPOINT_PATH." >&2
    echo "         If the download errored with 401/'repository not found'," >&2
    echo "         you must request access on the HF repo page AND pass HF_TOKEN." >&2
fi
if [ -f "$MHR_PATH" ]; then
    echo "--- mhr model OK: $MHR_PATH ($(du -h "$MHR_PATH" | cut -f1)) ---"
else
    echo "WARNING: assets/mhr_model.pt not found at $MHR_PATH." >&2
fi
if [ -f "$MODEL_CONFIG" ]; then
    echo "--- model config OK: $MODEL_CONFIG ---"
else
    echo "WARNING: model_config.yaml not found at $MODEL_CONFIG." >&2
fi

echo "=== [01] Done. Weights at: $MODEL_DIR ==="
echo "    checkpoint: $CHECKPOINT_PATH"
echo "    mhr model:  $MHR_PATH"
echo "    model cfg:  $MODEL_CONFIG"
echo "    fov model:  $FOV_MODEL_DIR"
