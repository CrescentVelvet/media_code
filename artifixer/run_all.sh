#!/usr/bin/env bash
# run_all.sh — one-click ArtiFixer pipeline:
#   00 (setup) → 01 (download) → 02 (prepare data) → 03 (ArtiFixer)
#   → 04 (ArtiFixer3D) → 05 (ArtiFixer3D+)
#
# Usage:
#   GPU=0 bash artifixer/run_all.sh
#   GPU=0 SCENE_NAME=my_scene COLMAP_DIR=/path/to/colmap bash artifixer/run_all.sh
#
# Skip steps with SKIP_SETUP=1, SKIP_DOWNLOAD=1, SKIP_PREP=1, SKIP_3D=1, SKIP_3D_PLUS=1.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional proxy (must be set before git clone).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

# --- Corporate proxy TLS interception workaround (git clone over HTTPS) ---
SYS_CA=/etc/ssl/certs/ca-certificates.crt
if [ -f "$SYS_CA" ]; then
    : "${GIT_SSL_CAINFO:=$SYS_CA}"
    export GIT_SSL_CAINFO
fi

echo "=== [run_all] ArtiFixer one-click pipeline (conda env: ${CONDA_ENV:-artifixer}) ==="
echo "    variant: ${MODEL_VARIANT:-1.3b}  (1.3b fits on single 80 GB A100)"

# 0. Clone + env setup.
if [ "${SKIP_SETUP:-0}" = "1" ]; then
    echo "⏭️  skipping setup (SKIP_SETUP=1)"
else
    if [ "${INSTALL_DEPS:-1}" = "1" ]; then
        INSTALL_DEPS=1 bash "$SCRIPT_DIR/00_setup_env.sh"
    else
        bash "$SCRIPT_DIR/00_setup_env.sh"
    fi
fi

# 1. Download weights.
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
    echo "⏭️  skipping download (SKIP_DOWNLOAD=1)"
else
    bash "$SCRIPT_DIR/01_download_models.sh"
fi

# 2. Prepare scene data.
if [ "${SKIP_PREP:-0}" = "1" ]; then
    echo "⏭️  skipping data prep (SKIP_PREP=1)"
else
    bash "$SCRIPT_DIR/02_prepare_data.sh"
fi

# 3. ArtiFixer inference.
bash "$SCRIPT_DIR/03_run_inference.sh"

# 4. ArtiFixer3D distillation.
if [ "${SKIP_3D:-0}" = "1" ]; then
    echo "⏭️  skipping ArtiFixer3D (SKIP_3D=1)"
else
    bash "$SCRIPT_DIR/04_run_artifixer3d.sh"
fi

# 5. ArtiFixer3D+ inference.
if [ "${SKIP_3D_PLUS:-0}" = "1" ]; then
    echo "⏭️  skipping ArtiFixer3D+ (SKIP_3D_PLUS=1)"
else
    bash "$SCRIPT_DIR/05_run_artifixer3d_plus.sh"
fi

echo "=== [run_all] All steps finished. ==="
echo "    Results: ${RESULTS_DIR:-../artifixer_results}/"
echo "    To run on your own COLMAP scene:"
echo "      GPU=0 COLMAP_DIR=/path/to/colmap SCENE_NAME=my_scene bash artifixer/run_all.sh"
