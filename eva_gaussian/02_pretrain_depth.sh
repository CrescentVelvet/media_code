#!/usr/bin/env bash
# 02_pretrain_depth.sh — Stage 1: pretrain the Gaussian position estimation
# network (depthnet) with GT depth. Runs depthnet_pretrain.py from the
# EVA-Gaussian repo.
#
# depthnet_pretrain.py hardcodes cfg.load("config/pretrain.yaml") and output
# paths "experiments/...", so we:
#   1. Run make_config.py to write pretrain.yaml at $EVA_DIR/config/
#   2. Symlink $EVA_DIR/experiments -> $RESULTS_DIR/experiments (output redirect)
#   3. Run depthnet_pretrain.py from $EVA_DIR (cd $EVA_DIR)
#
# Output: $RESULTS_DIR/experiments/pretrain_MMDD/{ckpt,logs,file}/
#
# Env:
#   DATA_ROOT=/path/to/dataset   (required; passed to config)
#   GPU=0                        (physical GPU id)
#   ANCHOR=1                     (enable anchor loss in pretrain)
#   LR=0.0002  BATCH_SIZE=6  NUM_STEPS=100000   (hyperparameter overrides)
#   RESUME_CKPT=/path/to/pretrain_latest.pth  (resume from checkpoint)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

DATA_ROOT="${DATA_ROOT:?❌ DATA_ROOT env var required}"

# Ensure EVA-Gaussian repo is cloned.
if [ ! -f "$EVA_DIR/depthnet_pretrain.py" ]; then
    echo "❌ ERROR: $EVA_DIR/depthnet_pretrain.py not found." >&2
    echo "       Run INSTALL_DEPS=1 BUILD_CUDA=1 bash $SCRIPT_DIR/00_setup_env.sh first" >&2
    exit 1
fi

# Generate pretrain.yaml (in case DATA_ROOT / params changed).
echo "📐 [02] generating pretrain.yaml"
export DATA_ROOT EVA_DIR
python "$SCRIPT_DIR/make_config.py" pretrain

# If resuming, patch restore_ckpt into the config.
if [ -n "${RESUME_CKPT:-}" ]; then
    echo "🏋️ [02] resume from checkpoint: $RESUME_CKPT"
    # yacs merge_from_file doesn't have restore_ckpt in the default config node,
    # so we add it via a python one-liner that appends to the yaml.
    python - "$EVA_DIR/config/pretrain.yaml" "$RESUME_CKPT" <<'PY'
import sys
path, ckpt = sys.argv[1], sys.argv[2]
with open(path) as f:
    txt = f.read()
# remove any existing restore_ckpt line, then add it.
lines = [l for l in txt.splitlines() if not l.startswith("restore_ckpt:")]
lines.insert(1, f"restore_ckpt: '{ckpt}'")
with open(path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"  ✅ patched restore_ckpt -> {ckpt}")
PY
fi

# Redirect experiments/ output to $RESULTS_DIR via symlink.
EXP_DIR="$RESULTS_DIR/experiments"
mkdir -p "$EXP_DIR"
if [ -L "$EVA_DIR/experiments" ]; then
    rm -f "$EVA_DIR/experiments"
fi
if [ ! -e "$EVA_DIR/experiments" ]; then
    ln -sfn "$EXP_DIR" "$EVA_DIR/experiments"
else
    echo "⚠️ $EVA_DIR/experiments already exists (not a symlink) — output stays in repo" >&2
fi

echo "🚀 [02] pretrain depthnet (stage 1)"
echo "  🤖 model: EVANet (with_gs_render=False)"
echo "  📁 DATA_ROOT:  $DATA_ROOT"
echo "  💾 output:     $EXP_DIR/pretrain_*/"
echo "  🎮 GPU:         ${GPU:-unset}"
echo ""

# Run from $EVA_DIR so "config/pretrain.yaml" and "experiments/" resolve correctly.
cd "$EVA_DIR"
python depthnet_pretrain.py
if [ $? -ne 0 ]; then
    echo "❌ FAILED: depthnet_pretrain.py" >&2
    exit 1
fi

# Find the output checkpoint (prefer _final.pth, fall back to _latest.pth).
LATEST_CKPT="$(find "$EXP_DIR" -name 'pretrain_*_final.pth' 2>/dev/null | sort | tail -1)"
if [ -z "$LATEST_CKPT" ]; then
    LATEST_CKPT="$(find "$EXP_DIR" -name 'pretrain_*_latest.pth' 2>/dev/null | sort | tail -1)"
fi
echo ""
echo "🎉 [02] Done. Stage-1 pretrain complete."
if [ -n "$LATEST_CKPT" ]; then
    echo "    🏋️ checkpoint: $LATEST_CKPT"
    echo "    Next: GPU=0 DATA_ROOT=$DATA_ROOT STAGE1_CKPT=$LATEST_CKPT bash $SCRIPT_DIR/03_train.sh"
else
    echo "    ⚠️ checkpoint not found — check $EXP_DIR/pretrain_*/ckpt/"
    echo "    Next: GPU=0 DATA_ROOT=$DATA_ROOT STAGE1_CKPT=<path> bash $SCRIPT_DIR/03_train.sh"
fi
