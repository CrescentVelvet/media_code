#!/usr/bin/env bash
# 01_download_models.sh — fetch the Deformable-GS adjusted D-NeRF dataset into
# the shared model store. For a training-based dynamic-3DGS method there are no
# pretrained weights to download: the "model" is the trained Gaussian point
# cloud, produced by `train.py` (see run_all.sh / README "Train"). So this
# script downloads the **reproduction dataset** instead — the adjusted D-NeRF
# (CVPR 2024 paper's main quantitative benchmark).
#
#   v0.1-pre-released / D-NeRF-Deformable-GS.zip (~258 MB)
#     -> $DATA_DIR/D-NeRF/<scene>/  (transforms_train.json + images/)
#
# Why "adjusted": the vanilla D-NeRF Lego train/test sets are inconsistent (the
# shovel's flip angle differs). ingra14m ships a fixed Lego (val-as-test +
# first val frame added to train) so the paper's numbers are reproducible. The
# other 7 D-NeRF scenes (bouncing, hell, hook, jump, mutant, standup, trex) are
# the standard synthetic Blender set; all use `--is_blender`.
#
# Source: a GitHub *release asset* (not HuggingFace). The download redirects to
# objects.githubusercontent.com, which behind a TLS-intercepting corporate proxy
# can present a cert the trust store rejects -> we retry with --insecure. The
# proxy + CA bundle from _env.sh are forwarded to curl.
#
# NeRF-DS / HyperNeRF (real-world) are NOT auto-downloaded (no scriptable,
# license-clear mirror). See README "Datasets" for manual placement.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

DG_DIR="${DG_DIR:-$REPO_DIR/../Deformable-3D-Gaussians}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/deformable-3d-gaussians}"

# Dataset root. The official layout is data/D-NeRF/<scene>; train.py takes an
# absolute -s path, so data can live anywhere. We keep it in the shared model
# store (one dir above <code-dir>) to avoid polluting the cloned repo.
DATA_DIR="${DATA_DIR:-$MODEL_DIR/data}"
DNERF_DIR="${DNERF_DIR:-$DATA_DIR/D-NeRF}"

DG_REPO="${DG_REPO:-ingra14m/Deformable-3D-Gaussians}"
RELEASE_TAG="${RELEASE_TAG:-v0.1-pre-released}"
ASSET_NAME="${ASSET_NAME:-D-NeRF-Deformable-GS.zip}"
ZIP_URL="${ZIP_URL:-https://github.com/$DG_REPO/releases/download/$RELEASE_TAG/$ASSET_NAME}"

echo "=== [01] Downloading D-NeRF dataset (Deformable-GS adjusted) ==="
echo "  代码路径:  $DG_DIR"
echo "  模型/数据: $MODEL_DIR"
echo "  数据集:    $DNERF_DIR"
echo "  来源:      $ZIP_URL"

if [ ! -d "$DG_DIR" ]; then
    echo "ERROR: Deformable-GS code dir not found at $DG_DIR." >&2
    echo "       Run run_all.sh first (it clones the official repo), or set DG_DIR." >&2
    exit 1
fi

mkdir -p "$DATA_DIR" "$DNERF_DIR"

# --- find scenes already present (a scene dir has transforms_train.json) ---
find_scenes() {
    find "$1" -maxdepth 2 -name transforms_train.json -printf '%h\n' 2>/dev/null \
        | sed "s|$1/||" | sort -u
}

present_count=$(find "$DNERF_DIR" -maxdepth 2 -name transforms_train.json 2>/dev/null | wc -l)
if [ "$present_count" -gt 0 ]; then
    echo "--- dataset already present ($present_count scene(s)): ---"
    find_scenes "$DNERF_DIR" | sed 's/^/    /'
    echo "--- to re-download, remove $DNERF_DIR first ---"
    echo "=== [01] Done. Dataset at: $DNERF_DIR ==="
    exit 0
fi

# --- download one URL with curl (proxy + CA bundle forwarded; SSL-bypass fallback) ---
# dl <url> <out_path>
dl() {
    local url="$1" out="$2"
    local flags=(-L --fail --retry 5 --retry-delay 5 --connect-timeout 30 -o "$out")
    [ -n "${http_proxy:-}" ]      && flags+=(--proxy "$http_proxy")
    [ -n "${CURL_CA_BUNDLE:-}" ] && flags+=(--cacert "$CURL_CA_BUNDLE")
    if [ "${DL_DISABLE_SSL:-0}" = "1" ]; then
        echo "    (SSL verification DISABLED via DL_DISABLE_SSL=1)"
        flags+=(--insecure)
        curl "${flags[@]}" "$url" && return 0
        return 1
    fi
    if curl "${flags[@]}" "$url"; then return 0; fi
    echo "--- curl failed (likely SSL on the CDN redirect); retrying with --insecure ---"
    curl -L --fail --retry 3 --connect-timeout 30 --insecure \
        ${http_proxy:+--proxy "$http_proxy"} -o "$out" "$url"
}

STAGE="$DATA_DIR/.dnerf_stage"
mkdir -p "$STAGE"
ZIP_PATH="$STAGE/$ASSET_NAME"

if [ ! -f "$ZIP_PATH" ]; then
    echo "--- downloading $ASSET_NAME (~258 MB) ---"
    if ! dl "$ZIP_URL" "$ZIP_PATH"; then
        echo "ERROR: download failed. The GitHub release asset may be unreachable" >&2
        echo "       behind your proxy. Retry with DL_DISABLE_SSL=1, or download" >&2
        echo "       manually from:" >&2
        echo "         $ZIP_URL" >&2
        echo "       and place it at: $ZIP_PATH  then rerun this script." >&2
        exit 1
    fi
else
    echo "--- zip already downloaded: $ZIP_PATH ($(du -h "$ZIP_PATH" | cut -f1)) ---"
fi

# --- unzip + organize into $DNERF_DIR/<scene>/ ---
if ! command -v unzip >/dev/null 2>&1; then
    echo "ERROR: unzip not found. Install it (apt install unzip) and rerun." >&2
    exit 1
fi

UNPACK="$STAGE/unpacked"
rm -rf "$UNPACK"; mkdir -p "$UNPACK"
echo "--- unzipping -> $UNPACK ---"
if ! unzip -q -o "$ZIP_PATH" -d "$UNPACK" 2>unzip.err; then
    echo "--- zip unreadable/corrupt (likely a partial download); re-downloading ---"
    cat unzip.err >&2 2>/dev/null || true
    rm -f unzip.err "$ZIP_PATH"
    dl "$ZIP_URL" "$ZIP_PATH"
    rm -rf "$UNPACK"; mkdir -p "$UNPACK"
    unzip -q -o "$ZIP_PATH" -d "$UNPACK"
fi
rm -f unzip.err

# The zip may contain a top-level D-NeRF/ folder OR the scenes at its root.
# Detect the directory that holds transforms_train.json scenes and move them
# into $DNERF_DIR.
ROOT_WITH_SCENES=""
for cand in "$UNPACK/D-NeRF" "$UNPACK" "$UNPACK"/*/; do
    [ -d "$cand" ] || continue
    if [ -n "$(find "$cand" -maxdepth 2 -name transforms_train.json -print -quit 2>/dev/null)" ]; then
        ROOT_WITH_SCENES="$cand"; break
    fi
done

if [ -z "$ROOT_WITH_SCENES" ]; then
    echo "ERROR: no scene with transforms_train.json found in the unzipped archive." >&2
    echo "       Inspect $UNPACK and move D-NeRF scenes into $DNERF_DIR manually." >&2
    exit 1
fi
echo "--- scenes found under: $ROOT_WITH_SCENES ---"

# Move each scene dir (the one directly containing transforms_train.json) into
# $DNERF_DIR, idempotently.
moved=0
while IFS= read -r scene_dir; do
    name="$(basename "$scene_dir")"
    if [ -d "$DNERF_DIR/$name" ] && [ -f "$DNERF_DIR/$name/transforms_train.json" ]; then
        echo "    $name: already in place, skip"
    else
        rm -rf "$DNERF_DIR/$name"
        mv "$scene_dir" "$DNERF_DIR/$name"
        echo "    $name: moved -> $DNERF_DIR/$name"
        moved=$((moved + 1))
    fi
done < <(find "$ROOT_WITH_SCENES" -maxdepth 1 -mindepth 1 -type d \
            -exec sh -c 'test -e "$1/transforms_train.json"' _ {} \; -print | sort)

# Clean up staging (keep the zip for re-runs unless PURGE=1).
if [ "${PURGE:-0}" = "1" ]; then rm -rf "$STAGE"; fi

echo "--- scenes in $DNERF_DIR ---"
find_scenes "$DNERF_DIR" | sed 's/^/    /'

scene_total=$(find "$DNERF_DIR" -maxdepth 2 -name transforms_train.json | wc -l)
echo "=== [01] Done. $scene_total scene(s) at: $DNERF_DIR ==="
echo "    train a scene:  GPU=0 SCENE=hook bash deformable_gaussians/run_all.sh"
echo "    (or: python train.py -s $DNERF_DIR/<scene> -m <out> --eval --is_blender)"
