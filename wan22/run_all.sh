#!/usr/bin/env bash
# run_all.sh — one-click: clone DiffSynth-Studio -> verify env -> verify weights ->
# inference (on a default prompt). Dataset construction (03) and LoRA training (04)
# are separate because they need your own data.
# Uses the existing conda env (torch preinstalled); set INSTALL_DEPS=1 once to
# install the diffsynth package.
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

echo "=== [run_all] Wan2.2-TI2V-5B one-click pipeline (conda env: ${CONDA_ENV:-wan22}) ==="

# 0. Clone + env setup.
if [ "${INSTALL_DEPS:-1}" = "1" ]; then
    INSTALL_DEPS=1 bash "$SCRIPT_DIR/00_setup_env.sh"
else
    bash "$SCRIPT_DIR/00_setup_env.sh"
fi

# 1. Verify weights (user pre-downloaded to $MODEL_DIR).
bash "$SCRIPT_DIR/01_verify_models.sh"

# 2. Inference on the default prompt (T2V).
bash "$SCRIPT_DIR/02_run_inference.sh"

echo "=== [run_all] All steps finished. ==="
echo "    To train on your own videos:"
echo "      DATA_DIR=/path/to/videos bash $SCRIPT_DIR/03_build_dataset.sh"
echo "      DATASET_BASE_PATH=/path/to/dataset bash $SCRIPT_DIR/04_train_lora.sh"
