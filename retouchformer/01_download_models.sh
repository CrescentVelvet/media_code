#!/usr/bin/env bash
# 01_download_models.sh — obtain the RetouchFormer checkpoint (gen_best.pth).
#
# ⚠️ The official release (per the repo README) is ONLY distributed via Baidu
# Netdisk (https://pan.baidu.com/s/1eVgPN12KJN8GSdOw544ZdQ, code: `reto`).
# There is NO public HTTP/HuggingFace mirror we can script against, so this
# script CANNOT auto-download it. Instead it:
#   1. Checks for an already-placed $WEIGHT_PATH; if present, done.
#   2. If WEIGHT_URL is set (you mirrored gen_best.pth somewhere), downloads it
#      via curl with proxy + CA-bundle + SSL-fallback support.
#   3. Otherwise prints clear Baidu instructions and exits non-zero.
#
# Expected layout after success (matches the official img_retouching.py
# `-c release_model` default — weights live UNDER release_model/):
#   $MODEL_DIR/release_model/gen_best.pth
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

RETOUCH_DIR="${RETOUCH_DIR:-$REPO_DIR/../RetouchFormer}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/RetouchFormer}"

# Default checkpoint name/epoch mirror the official img_retouching.py:
#   --ckpt release_model --epoch best  ->  {ckpt}/gen_{epoch}.pth
CKPT_DIR_NAME="${CKPT_DIR_NAME:-release_model}"
EPOCH="${EPOCH:-best}"
WEIGHT_FILE="${WEIGHT_FILE:-gen_${EPOCH}.pth}"        # gen_best.pth
CKPT_DIR="${CKPT_DIR:-$MODEL_DIR/$CKPT_DIR_NAME}"
WEIGHT_PATH="${WEIGHT_PATH:-$CKPT_DIR/$WEIGHT_FILE}"

echo "=== [01] Obtain RetouchFormer weights ==="
echo "  model dir:  $MODEL_DIR"
echo "  code dir:   $RETOUCH_DIR"
echo "  checkpoint: $WEIGHT_PATH"

if [ ! -d "$RETOUCH_DIR" ]; then
    echo "ERROR: RetouchFormer code dir not found at $RETOUCH_DIR." >&2
    echo "       Run run_all.sh first (it clones the official repo), or set RETOUCH_DIR." >&2
    exit 1
fi

mkdir -p "$CKPT_DIR"

# 1) Already present?
if [ -f "$WEIGHT_PATH" ]; then
    echo "--- checkpoint already present: $WEIGHT_PATH ($(du -h "$WEIGHT_PATH" | cut -f1)) ---"
    exit 0
fi

# 2) Optional direct download (user mirrored the file to an HTTP URL).
if [ -n "${WEIGHT_URL:-}" ]; then
    echo "--- WEIGHT_URL set -> downloading: $WEIGHT_URL ---"
    CURL_FLAGS=(--location --fail --show-error --retry 5 --retry-delay 5 \
        --connect-timeout 30 -o "$WEIGHT_PATH")
    [ -n "${http_proxy:-}" ]    && CURL_FLAGS+=(--proxy "$http_proxy")
    [ -n "${REQUESTS_CA_BUNDLE:-}" ] && CURL_FLAGS+=(--cacert "$REQUESTS_CA_BUNDLE")
    if ! curl "${CURL_FLAGS[@]}" "$WEIGHT_URL"; then
        echo "--- curl failed (likely SSL on CDN); retrying with SSL verification disabled ---"
        curl --location --fail --show-error --retry 5 -k -o "$WEIGHT_PATH" "$WEIGHT_URL"
    fi
    if [ -f "$WEIGHT_PATH" ]; then
        echo "--- downloaded OK: $WEIGHT_PATH ($(du -h "$WEIGHT_PATH" | cut -f1)) ---"
        exit 0
    else
        echo "ERROR: WEIGHT_URL download did not produce $WEIGHT_PATH." >&2
        exit 1
    fi
fi

# 3) No weight + no URL -> instruct manual Baidu download.
cat >&2 <<EOF

================================================================================
  RetouchFormer checkpoint NOT found: $WEIGHT_PATH

  The official release is distributed ONLY via Baidu Netdisk (no HTTP mirror):
    Link:          https://pan.baidu.com/s/1eVgPN12KJN8GSdOw544ZdQ
    提取码 (code): reto

  Manual steps (do once):
    1. Open the link above in a browser, enter code \`reto\`, download \`gen_best.pth\`.
    2. Copy it to the expected path:
         mkdir -p "$CKPT_DIR"
         cp /path/to/gen_best.pth "$WEIGHT_PATH"
    3. Re-run this script (it will detect the file and pass):
         bash "$SCRIPT_DIR/01_download_models.sh"

  Optional (if you host the file on an internal server):
    WEIGHT_URL=https://your.host/path/gen_best.pth bash "$SCRIPT_DIR/01_download_models.sh"
================================================================================
EOF
exit 1
