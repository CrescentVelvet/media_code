#!/usr/bin/env bash
# 00_setup_env.sh — server variant (conda env already exists; just verify + deps).
#
# On a server the osediff env is expected to already exist (a CUDA-enabled torch
# already installed). This script verifies it and optionally installs the rest
# of OSEDiff's requirements.txt. For WSL/local dev use 00a_setup_env.sh instead.
#
# Usage:
#   bash osediff/00_setup_env.sh                # verify only
#   INSTALL_DEPS=1 bash osediff/00_setup_env.sh # verify + pip install -r requirements.txt
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [00] Verify osediff env (server)"
echo "  🐍 conda env: $CONDA_ENV  (at $CONDA_PREFIX)"

# ── 1. Verify torch (CUDA-enabled) ────────────────────────────────────────
echo ""
echo "=== verify torch ==="
python - <<'PY' || exit 1
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("❌ ERROR: torch.cuda not available — install a CUDA build of torch")
PY

# ── 2. Clone official repo if absent ──────────────────────────────────────
mkdir -p "$(dirname "$OSEDIFF_DIR")"
if [ ! -d "$OSEDIFF_DIR/.git" ]; then
    echo ""
    echo "📦 cloning OSEDiff -> $OSEDIFF_DIR"
    # LD_LIBRARY_PATH= avoids conda libffi clashing with system libp11-kit (git crash).
    LD_LIBRARY_PATH= git clone "$OSEDIFF_REPO" "$OSEDIFF_DIR" || \
        LD_LIBRARY_PATH= git -c http.sslVerify=false clone "$OSEDIFF_REPO" "$OSEDIFF_DIR"
else
    echo "⏭️  OSEDiff already present: $OSEDIFF_DIR"
fi

# ── 3. Install requirements (optional) ───────────────────────────────────
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org \
        --timeout 600 --retries 10)
    REQ="$OSEDIFF_DIR/requirements.txt"
    if [ ! -f "$REQ" ]; then
        echo "❌ ERROR: $REQ not found (clone failed?)" >&2
        exit 1
    fi
    echo ""
    echo "📦 installing requirements (skipping torch/torchvision/xformers — keep server's)"
    TMP_REQ="$(mktemp).txt"
    grep -v -iE '^[[:space:]]*(torch|torchvision|xformers)([=<>!~]|$|[[:space:]])' "$REQ" > "$TMP_REQ"
    pip install "${PIP_FLAGS[@]}" -r "$TMP_REQ"
    rm -f "$TMP_REQ"
else
    echo "⏭️  INSTALL_DEPS!=1 — skipping requirements install"
fi

# ── 4. Verify imports ─────────────────────────────────────────────────────
echo ""
echo "=== verify imports ==="
python -c "import diffusers, transformers, peft; print('  ✅ diffusers/transformers/peft')" 2>/dev/null || \
    echo "  ⚠️ diffusers/transformers/peft missing — run INSTALL_DEPS=1 bash $0"
python -c "import xformers.ops; print('  ✅ xformers')" 2>/dev/null || \
    echo "  [---] xformers (optional; training -O2 needs it)"
[ -f "$OSEDIFF_PKL" ] && echo "  ✅ osediff.pkl: $(du -h "$OSEDIFF_PKL" | cut -f1)" || \
    echo "  [MISS] osediff.pkl (should ship in preset/models/ after clone)"

echo ""
echo "🎉 [00] Done. Next:"
echo "  GPU=0 bash osediff/01_download_models.sh"
echo "  GPU=0 bash osediff/02_run_inference.sh"
