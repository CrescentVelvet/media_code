#!/usr/bin/env bash
# setup_ca_bundle.sh — build ~/.ca-bundle.crt from the corporate proxy's TLS
# cert chain, so pip/hf/git trust the proxy's MITM certs. Run once on the
# server when 01_download_models.sh fails with SSLCertVerificationError.
# After this, _env.sh auto-uses ~/.ca-bundle.crt — no proxy.env edit needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"   # conda env + proxy env vars

echo "=== Building CA bundle from proxy cert chain ==="
python "$SCRIPT_DIR/_extract_ca.py"
echo "=== Done. ==="
