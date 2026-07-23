#!/usr/bin/env bash
# run_all.sh — one-click: clone official repo -> verify env -> weights ->
# inference (on the bundled notebook/images example). Dataset download
# (data/README.md) and SAM3/SAM2 setup are separate because they need your
# own access / extra repos.
# Uses the existing conda env (torch preinstalled); set INSTALL_DEPS=1 once to
# install the official dependencies from INSTALL.md.
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

SAM3D_DIR="${SAM3D_DIR:-$REPO_DIR/../sam-3d-body}"
SAM3D_REPO="${SAM3D_REPO:-https://github.com/facebookresearch/sam-3d-body.git}"

echo "=== [run_all] SAM 3D Body one-click pipeline (conda env: ${CONDA_ENV:-sam_3d_body}) ==="

# 0. Clone the official repo if it is not present yet.
if [ ! -d "$SAM3D_DIR" ]; then
    mkdir -p "$(dirname "$SAM3D_DIR")"
    echo "--- cloning official repo -> $SAM3D_DIR ---"
    git clone "$SAM3D_REPO" "$SAM3D_DIR" || \
        git -c http.sslVerify=false clone "$SAM3D_REPO" "$SAM3D_DIR"
else
    echo "--- official repo already present: $SAM3D_DIR ---"
fi

export SAM3D_DIR

# First-time setup: install dependencies into the env. Use a DEDICATED env
# (CONDA_ENV=sam_3d_body) — the detectron2 / networkx pins clash with other
# algos. On later runs omit INSTALL_DEPS to skip pip.
if [ "${INSTALL_DEPS:-1}" = "1" ]; then
    INSTALL_DEPS=1 bash "$SCRIPT_DIR/00_setup_env.sh"
else
    bash "$SCRIPT_DIR/00_setup_env.sh"
fi
bash "$SCRIPT_DIR/01_download_models.sh"

# Default to the bundled example image (notebook/images/dancing.jpg) so the
# pipeline works out of the box once weights are present.
INPUT_DIR="${INPUT_DIR:-$SAM3D_DIR/notebook/images}" bash "$SCRIPT_DIR/02_run_inference.sh"

echo "=== [run_all] All steps finished. ==="
echo "    To run on your own images:"
echo "      GPU=0 INPUT_DIR=/path/to/images bash $SCRIPT_DIR/02_run_inference.sh"
echo "    To use the ViT-H backbone instead of DINOv3-H+:"
echo "      HF_REPO_ID=facebook/sam-3d-body-vith bash $SCRIPT_DIR/01_download_models.sh"
echo "      HF_REPO_ID=facebook/sam-3d-body-vith GPU=0 INPUT_DIR=/path/to/images bash $SCRIPT_DIR/02_run_inference.sh"
echo "    ⚠️ The SAM 3D Body checkpoint is GATED — you must request access on"
echo "       HuggingFace and pass HF_TOKEN to 01. See sam_3d_body/README.md."
