#!/usr/bin/env bash
# 02_run_inference.sh — run the official run_example.py on a single sample image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_DIR="$SCRIPT_DIR"
REPO_DIR="$(dirname "$ALGO_DIR")"

# ---- Config ----------------------------------------------------------------
VENV_DIR="${VENV_DIR:-$ALGO_DIR/.venv}"
TRIPOSPLAT_DIR="${TRIPOSPLAT_DIR:-$REPO_DIR/../TripoSplat}"
# ----------------------------------------------------------------------------

echo "=== [02] Running TripoSplat inference ==="
echo "  code dir: $TRIPOSPLAT_DIR"

if [ ! -d "$TRIPOSPLAT_DIR" ]; then
    echo "ERROR: TripoSplat code dir not found at $TRIPOSPLAT_DIR." >&2
    exit 1
fi
if [ ! -e "$TRIPOSPLAT_DIR/ckpts" ]; then
    echo "ERROR: $TRIPOSPLAT_DIR/ckpts missing. Run 01_download_models.sh first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# run_example.py loads weights from ./ckpts/... (a symlink to the shared
# model store) and reads ./static/example_inputs/... — launch from repo root.
cd "$TRIPOSPLAT_DIR"
python run_example.py

echo "--- outputs ---"
ls -lh output.ply output.splat preprocessed_image.webp 2>/dev/null || true
for n in 32768 65536 131072 262144; do
    ls -lh "output_${n}.ply" 2>/dev/null || true
done

echo "=== [02] Done. Outputs written under: $TRIPOSPLAT_DIR ==="
