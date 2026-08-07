#!/usr/bin/env bash
# 01_verify_models.sh — verify Wan2.2-TI2V-5B weights are present in $MODEL_DIR,
# and fix the directory layout so DiffSynth's ModelConfig can find them.
#
# DiffSynth's ModelConfig (with DIFFSYNTH_SKIP_DOWNLOAD=True) looks for files at:
#   $DIFFSYNTH_MODEL_BASE_PATH/<model_id>/<origin_file_pattern>
# e.g. $MODEL_DIR/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors
#
# But modelscope download --local_dir often creates the dir WITHOUT the org
# prefix (e.g. $MODEL_DIR/Wan2.2-TI2V-5B/ instead of $MODEL_DIR/Wan-AI/Wan2.2-TI2V-5B/).
# This script detects both layouts and creates a symlink $MODEL_DIR/Wan-AI -> .
# so the org-prefixed path resolves either way.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# Model repos and the files each should contain.
# Format: "model_id|file_glob|description"
MODELS=(
    "Wan-AI/Wan2.2-TI2V-5B|diffusion_pytorch_model*.safetensors|DiT (diffusion transformer)"
    "Wan-AI/Wan2.2-TI2V-5B|models_t5_umt5-xxl-enc-bf16.pth|T5 text encoder (UMT5-XXL)"
    "Wan-AI/Wan2.2-TI2V-5B|Wan2.2_VAE.pth|VAE"
    "Wan-AI/Wan2.1-T2V-1.3B|google/umt5-xxl|tokenizer (UMT5-XXL)"
)

echo "=== [01] Verify Wan2.2-TI2V-5B weights in: $MODEL_DIR ==="

# --- ensure the org-prefix symlink exists so both layouts resolve ---
# If $MODEL_DIR/Wan-AI/<model>/ exists, the layout already has the org prefix.
# If only $MODEL_DIR/<model>/ exists (no Wan-AI/), symlink Wan-AI -> . so the
# org-prefixed glob still resolves.
ORG="Wan-AI"
if [ ! -e "$MODEL_DIR/$ORG" ]; then
    # Check whether any model dir exists WITHOUT the org prefix.
    _need_link=0
    for entry in "Wan2.2-TI2V-5B" "Wan2.1-T2V-1.3B"; do
        if [ -d "$MODEL_DIR/$entry" ]; then _need_link=1; break; fi
    done
    if [ "$_need_link" = "1" ]; then
        echo "--- creating symlink $MODEL_DIR/$ORG -> . (so org-prefixed paths resolve) ---"
        ln -sfn . "$MODEL_DIR/$ORG"
    fi
fi

# --- check each model file ---
# For files/globs, use bash glob; for directory paths (contain '/'), check -d.
_all_ok=1
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_id file_glob desc <<< "$entry"
    base="$MODEL_DIR/$model_id"
    found=""
    if [[ "$file_glob" == */ ]]; then
        # Directory pattern (e.g. "google/umt5-xxl/") — check the dir exists.
        if [ -d "$base/${file_glob%/}" ]; then
            found="$model_id/${file_glob%/}"
        fi
    elif [[ "$file_glob" == */* ]]; then
        # Sub-path without trailing slash (e.g. "google/umt5-xxl") — check dir.
        if [ -d "$base/$file_glob" ]; then
            found="$model_id/$file_glob"
        fi
    else
        # File or glob pattern — expand with shell glob.
        shopt -s nullglob
        matches=("$base/$file_glob")
        shopt -u nullglob
        if [ ${#matches[@]} -gt 0 ]; then
            found="$model_id/$(basename "${matches[0]}")"
        fi
    fi
    if [ -n "$found" ]; then
        size=$(du -h "$MODEL_DIR/$found" 2>/dev/null | cut -f1 || echo "?")
        echo "  [OK]   $desc: $found  ($size)"
    else
        echo "  [MISS] $desc: $model_id/$file_glob  -- not found in $MODEL_DIR"
        _all_ok=0
    fi
done

if [ "$_all_ok" = "1" ]; then
    echo "=== [01] All weights present. Ready for inference / training. ==="
else
    echo "" >&2
    echo "WARNING: some weights are missing." >&2
    echo "  Expected layout (either works, this script links them):" >&2
    echo "    $MODEL_DIR/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors" >&2
    echo "    $MODEL_DIR/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth" >&2
    echo "    $MODEL_DIR/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" >&2
    echo "    $MODEL_DIR/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/" >&2
    echo "  Download with:" >&2
    echo "    modelscope download Wan-AI/Wan2.2-TI2V-5B --local_dir $MODEL_DIR/Wan-AI/Wan2.2-TI2V-5B" >&2
    echo "    modelscope download Wan-AI/Wan2.1-T2V-1.3B --local_dir $MODEL_DIR/Wan-AI/Wan2.1-T2V-1.3B --include google/umt5-xxl/" >&2
    echo "  Or if already downloaded without org prefix (e.g. $MODEL_DIR/Wan2.2-TI2V-5B/), this script auto-creates a symlink." >&2
fi
