#!/usr/bin/env bash
# 00_setup_env.sh — activate conda env (torch preinstalled) & verify torch+CUDA,
# then install TRELLIS.2's extension deps via the official setup.sh.
#
# Reuses an existing conda env (default name `trellis2`) that already has a
# CUDA-enabled torch — no torch download. Set CREATE_ENV=1 to let the official
# setup.sh create the `trellis2` env fresh (with torch 2.6.0 + cu124) and install
# everything in one shot. Set INSTALL_DEPS=0 to skip the dep install.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

CREATE_ENV="${CREATE_ENV:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
TRELLIS_DIR="${TRELLIS_DIR:-$REPO_DIR/../TRELLIS.2}"
# Default component set (matches the README's recommended install). Override
# with SETUP_FLAGS="--basic --nvdiffrast ..." etc.
SETUP_FLAGS="${SETUP_FLAGS:---basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm}"

if [ "$CREATE_ENV" = "1" ]; then
    # setup.sh --new-env creates the env + installs torch + all flagged deps.
    # Source proxy/CA/CUDA_HOME but skip conda activate (env doesn't exist yet).
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/_env.sh" no-activate
else
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/_env.sh"
    echo "=== [00] Verify torch in conda env '$CONDA_ENV' ==="
    python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda not available in this env — install a CUDA-enabled torch or check GPU visibility.")
PY
fi

if [ "$INSTALL_DEPS" = "1" ]; then
    if [ ! -f "$TRELLIS_DIR/setup.sh" ]; then
        echo "ERROR: $TRELLIS_DIR/setup.sh not found." >&2
        echo "       Run run_all.sh first (it clones the official repo), or set TRELLIS_DIR." >&2
        exit 1
    fi
    FLAGS="$SETUP_FLAGS"
    [ "$CREATE_ENV" = "1" ] && FLAGS="--new-env $FLAGS"
    echo "=== [00] Installing TRELLIS.2 deps via official setup.sh ==="
    echo "  flags: $FLAGS"
    echo "  (CUDA_HOME=$CUDA_HOME — should point at CUDA Toolkit 12.4)"
    # setup.sh must be SOURCED (it uses `conda activate` + `return`), and run
    # from the repo root so its relative `cp -r o-voxel ...` resolves. It is
    # designed to be sourced plainly (no `set -e`): a failing component should
    # not abort the others, so relax strict mode around the sourcing.
    # shellcheck disable=SC2164
    cd "$TRELLIS_DIR"
    set +e +u +o pipefail
    # shellcheck disable=SC1091
    . ./setup.sh $FLAGS
    SETUP_RC=$?
    set -e -u -o pipefail
    cd "$REPO_DIR"
    if [ "$SETUP_RC" -ne 0 ]; then
        echo "WARNING: setup.sh exited with code $SETUP_RC — some components may have failed." >&2
        echo "         Re-run with a reduced SETUP_FLAGS (e.g. drop --flash-attn), or pip install the missing piece." >&2
    fi
    echo "=== [00] Deps install attempted. (Missing a package? pip install it, or adjust SETUP_FLAGS) ==="
else
    echo "=== [00] INSTALL_DEPS=0 — skipping dep install. Env '$CONDA_ENV' ready. ==="
fi
