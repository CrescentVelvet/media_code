#!/usr/bin/env bash
# 01_download_models.sh — download SD2.1-Base + RAM; stage DAPE.
#
# OSEDiff ships osediff.pkl / osediff_face.pkl inside the repo (clone brings them),
# so we only fetch the three dependent models:
#   * SD2.1-Base (~5GB, HF: Manojb/stable-diffusion-2-1-base)
#   * RAM swin_large_14m (~1.7GB, HF: xinyu1205/recognize-anything)
#   * DAPE — fine-tuned RAM for enhancement captions. The repo ships a copy at
#     preset/models/DAPE.pth; we stage it to $MODEL_DIR. If you want the full
#     Google-Drive version, see README "可能遇到的问题".
#
# Usage:
#   bash osediff/01_download_models.sh
#   SKIP_SD=1 bash osediff/01_download_models.sh     # skip SD2.1
#   SKIP_RAM=1 bash osediff/01_download_models.sh    # skip RAM
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [01] Download OSEDiff dependent models"
echo "  🏋️ weights root: $MODEL_DIR"
echo "  🤖 SD2.1-Base:   $SD21_BASE_DIR"
echo "  🤖 RAM:          $RAM_PATH"
echo "  🤖 DAPE:         $DAPE_PATH"
echo ""

mkdir -p "$MODEL_DIR"

# ── helper: hf download with SSL-bypass curl fallback ─────────────────────
# Usage: hf_get <repo> <local_dir> <comma_separated_includes>
hf_get() {
    local repo="$1" dir="$2" includes="${3:-}"
    mkdir -p "$dir"
    local include_args=()
    if [ -n "$includes" ]; then
        local parts=()
        IFS=',' read -ra parts <<< "$includes"
        for p in "${parts[@]}"; do include_args+=(--include "$p"); done
    fi

    # Prefer `hf` (new CLI); fallback to `huggingface-cli` for older versions.
    local hf_cmd="hf"
    if ! command -v "$hf_cmd" >/dev/null 2>&1; then
        if command -v huggingface-cli >/dev/null 2>&1; then
            hf_cmd="huggingface-cli"
        else
            echo "  ❌ neither 'hf' nor 'huggingface-cli' found — pip install -U huggingface_hub" >&2
            return 1
        fi
    fi

    local token_args=()
    [ -n "${HF_TOKEN:-}" ] && token_args=(--token "$HF_TOKEN")

    echo "  📦 $hf_cmd download $repo -> $dir"
    if "$hf_cmd" download "$repo" "${token_args[@]}" "${include_args[@]}" --local-dir "$dir"; then
        return 0
    fi
    echo "  ⚠️ hf download failed (likely SSL); retrying individual files via curl (--insecure)..."
    local ep="${HF_ENDPOINT:-https://huggingface.co}"
    if [ -n "$includes" ]; then
        local parts=()
        IFS=',' read -ra parts <<< "$includes"
        for p in "${parts[@]}"; do
            [[ "$p" == *'*'* ]] && continue   # skip glob patterns (curl can't expand)
            mkdir -p "$dir/$(dirname "$p")"
            curl --insecure -sL --max-time 600 "$ep/$repo/resolve/main/$p" -o "$dir/$p" \
                || echo "    ⚠️ failed: $p" >&2
        done
    else
        echo "    (no --include list; cannot curl-fallback a whole repo)" >&2
        return 1
    fi
}

# Local download cache (AGENTS.md §6: D:\wheel = /mnt/d/wheel, shared across algos).
# download_urls.md lists files to manually place here; 01 stages them into $MODEL_DIR.
WHEELS_DIR="${WHEELS_DIR:-/mnt/d/wheel}"
LOCAL_CACHE="${LOCAL_CACHE:-$WHEELS_DIR/osediff_ms}"
export WHEELS_DIR LOCAL_CACHE

# ── 1. SD2.1-Base (only the files OSEDiff actually loads — saves ~9GB) ─────
HF_SD21_REPO="${HF_SD21_REPO:-Manojb/stable-diffusion-2-1-base}"
# model_index + scheduler + tokenizer + text_encoder + unet + vae (skip safety_checker etc.)
SD21_INCLUDES="model_index.json,scheduler/*.json,scheduler/*,tokenizer/*,text_encoder/config.json,text_encoder/model.safetensors,unet/config.json,unet/diffusion_pytorch_model.safetensors,vae/config.json,vae/diffusion_pytorch_model.safetensors"

if [ "${SKIP_SD:-0}" != "1" ]; then
    if [ -f "$SD21_BASE_DIR/unet/diffusion_pytorch_model.safetensors" ]; then
        echo "⏭️  SD2.1-Base already present: $SD21_BASE_DIR"
    elif [ -d "$LOCAL_CACHE/sd21_base" ] && [ -f "$LOCAL_CACHE/sd21_base/unet/diffusion_pytorch_model.safetensors" ]; then
        echo "📦 staging SD2.1-Base from local cache: $LOCAL_CACHE/sd21_base -> $SD21_BASE_DIR"
        mkdir -p "$SD21_BASE_DIR"
        cp -rn "$LOCAL_CACHE/sd21_base/." "$SD21_BASE_DIR/"
        echo "  ✅ staged ($(du -sh "$SD21_BASE_DIR" | cut -f1))"
    else
        echo "⚠️  SD2.1 not in local cache ($LOCAL_CACHE/sd21_base); downloading online (~5GB, slow)..."
        echo "    (faster: download to $LOCAL_CACHE/sd21_base/ per osediff/download_urls.md §B, then re-run)"
        hf_get "$HF_SD21_REPO" "$SD21_BASE_DIR" "$SD21_INCLUDES"
    fi
else
    echo "⏭️  SKIP_SD=1 — skipping SD2.1-Base"
fi

# ── 2. RAM (recognize-anything, single file) ──────────────────────────────
HF_RAM_REPO="${HF_RAM_REPO:-xinyu1205/recognize-anything}"
RAM_FILE="${RAM_FILE:-ram_swin_large_14m.pth}"

if [ "${SKIP_RAM:-0}" != "1" ]; then
    if [ -f "$RAM_PATH" ]; then
        echo "⏭️  RAM already present: $RAM_PATH ($(du -h "$RAM_PATH" | cut -f1))"
    elif [ -f "$LOCAL_CACHE/ram_swin_large_14m.pth" ]; then
        echo "📦 staging RAM from local cache -> $RAM_PATH"
        cp -n "$LOCAL_CACHE/ram_swin_large_14m.pth" "$RAM_PATH"
        echo "  ✅ staged ($(du -h "$RAM_PATH" | cut -f1))"
    else
        # RAM lives in a HF *spaces* repo, not a models repo — `hf download` can't
        # fetch it. Use curl on the spaces resolve URL (hf-mirror mirrors spaces).
        echo "⚠️  RAM not in local cache; downloading online from hf-mirror spaces (~1.7GB, slow)..."
        echo "    (faster: download to $LOCAL_CACHE/ram_swin_large_14m.pth per osediff/download_urls.md §C, then re-run)"
        _ram_url="${HF_ENDPOINT:-https://hf-mirror.com}/spaces/$HF_RAM_REPO/resolve/main/$RAM_FILE"
        mkdir -p "$(dirname "$RAM_PATH")"
        curl -L --insecure --max-time 1200 "$_ram_url" -o "$RAM_PATH" \
            || echo "  ❌ curl failed: $_ram_url" >&2
        if [ -f "$RAM_PATH" ] && [ "$(stat -c%s "$RAM_PATH" 2>/dev/null || echo 0)" -gt 1048576 ]; then
            echo "  ✅ downloaded ($(du -h "$RAM_PATH" | cut -f1))"
        else
            echo "  ❌ download incomplete — get it manually, see osediff/download_urls.md §C" >&2
        fi
    fi
else
    echo "⏭️  SKIP_RAM=1 — skipping RAM"
fi

# ── 3. DAPE (repo ships a copy at preset/models/DAPE.pth) ──────────────────
REPO_DAPE="$OSEDIFF_DIR/preset/models/DAPE.pth"
if [ -f "$DAPE_PATH" ]; then
    echo "⏭️  DAPE already staged: $DAPE_PATH ($(du -h "$DAPE_PATH" | cut -f1))"
elif [ -f "$REPO_DAPE" ]; then
    echo "📦 staging DAPE from repo -> $DAPE_PATH"
    cp "$REPO_DAPE" "$DAPE_PATH"
    echo "  ✅ staged ($(du -h "$DAPE_PATH" | cut -f1))"
    echo "  ⚠️ NOTE: this is the repo-shipped copy ($(du -h "$DAPE_PATH" | cut -f1))."
    echo "          If inference complains about DAPE weights, download the full"
    echo "          version from Google Drive (see README '可能遇到的问题')."
else
    echo "⚠️  DAPE not found in repo ($REPO_DAPE)."
    echo "    Trying gdown for the full Google-Drive version..."
    if ! command -v gdown >/dev/null 2>&1; then
        pip install --quiet gdown 2>/dev/null || true
    fi
    # Google Drive file id from the OSEDiff README.
    if command -v gdown >/dev/null 2>&1; then
        gdown "1KIV6VewwO2eDC9g4Gcvgm-a0LDI7Lmwm" -O "$DAPE_PATH" || \
            echo "  ❌ gdown failed — download manually (see README)" >&2
    else
        echo "  ❌ gdown unavailable — download manually (see README)" >&2
    fi
fi

# ── 4. Summary ────────────────────────────────────────────────────────────
echo ""
echo "=== summary ==="
_all_ok=1
for _f in "$SD21_BASE_DIR/unet/diffusion_pytorch_model.safetensors" "$RAM_PATH" "$DAPE_PATH" "$OSEDIFF_PKL"; do
    if [ -f "$_f" ]; then
        echo "  [OK]   $(basename "$_f"): $(du -h "$_f" | cut -f1)  $_f"
    else
        echo "  [MISS] $_f"
        _all_ok=0
    fi
done

if [ "$_all_ok" = "1" ]; then
    echo ""
    echo "🎉 [01] Done. All models ready."
    echo "  Next: GPU=0 bash osediff/02_run_inference.sh"
else
    echo ""
    echo "⚠️  Some models missing. Re-run or download manually." >&2
    exit 1
fi
