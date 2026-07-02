#!/usr/bin/env bash
# run_all.sh — one-click: clone official repo -> verify env -> weights ->
# inference (on the bundled examples). Dataset construction (03) and LoRA
# training (04) are separate because they need your own data.
# Uses the existing conda env (torch preinstalled); set INSTALL_DEPS=1 once to
# install the official requirements.txt.
set -euo pipefail

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

HYPIR_DIR="${HYPIR_DIR:-$REPO_DIR/../HYPIR}"
HYPIR_REPO="${HYPIR_REPO:-https://github.com/XPixelGroup/HYPIR.git}"

echo "=== [run_all] HYPIR one-click pipeline (conda env: ${CONDA_ENV:-doll}) ==="

# 0. Clone the official repo if it is not present yet.
if [ ! -d "$HYPIR_DIR" ]; then
    mkdir -p "$(dirname "$HYPIR_DIR")"
    echo "--- cloning official repo -> $HYPIR_DIR ---"
    git clone "$HYPIR_REPO" "$HYPIR_DIR" || \
        git -c http.sslVerify=false clone "$HYPIR_REPO" "$HYPIR_DIR"
else
    echo "--- official repo already present: $HYPIR_DIR ---"
fi

export HYPIR_DIR

# First-time setup: install requirements into the env. HYPIR pins
# diffusers/transformers/peft versions that conflict with other algos — use a
# dedicated env (CONDA_ENV=hypir). On later runs omit INSTALL_DEPS to skip pip.
if [ "${INSTALL_DEPS:-1}" = "1" ]; then
    INSTALL_DEPS=1 bash "$SCRIPT_DIR/00_setup_env.sh"
else
    bash "$SCRIPT_DIR/00_setup_env.sh"
fi
bash "$SCRIPT_DIR/01_download_models.sh"
bash "$SCRIPT_DIR/02_run_inference.sh"

echo "=== [run_all] All steps finished. ==="
echo "    To train on your own data:"
echo "      DATA_DIR=/path/to/images bash $SCRIPT_DIR/03_build_dataset.sh"
echo "      PARQUET_PATH=/path/to/hypir_train.parquet bash $SCRIPT_DIR/04_train.sh"
