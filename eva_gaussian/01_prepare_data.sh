#!/usr/bin/env bash
# 01_prepare_data.sh — verify dataset structure + generate config YAMLs.
#
# The dataset must be in GPS-Gaussian format (rendered multi-view from THuman2.0
# or THumansit scans). See:
#   https://github.com/aipixel/GPS-Gaussian/blob/main/prepare_data/MAKE_DATA.md
#
# Expected layout:
#   $DATA_ROOT/
#   ├── train/
#   │   ├── img/<sample>/<0,1,2,3,4>.jpg
#   │   ├── mask/<sample>/<0,1>.png
#   │   ├── depth/<sample>/<0,1>.png        (uint16, /2^15 = meters)
#   │   └── parm/<sample>/<0,1>_intrinsic.npy, <0,1>_extrinsic.npy
#   ├── val/   (same structure)
#   └── landmark.json   (only if ANCHOR=1; run 04_gen_landmarks.sh to generate)
#
# Env:
#   DATA_ROOT=/path/to/dataset   (required)
#   ANCHOR=1                     (enable anchor loss; needs landmark.json)
#   STAGE1_CKPT=/path/to/stage1.pth  (for train config; auto-set by 03_train.sh)
#   SOURCE_ID=0,1                (source view IDs)
#   TRAIN_NOVEL_ID=2,3,4         (novel view IDs for training)
#   VAL_NOVEL_ID=3               (novel view IDs for validation)
#   USE_HR_IMG=1                 (use _hr.jpg high-res images)
#   LR / WDECAY / BATCH_SIZE / NUM_STEPS  (hyperparameter overrides)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

DATA_ROOT="${DATA_ROOT:?❌ DATA_ROOT env var required (dataset root with train/ and val/)}"

echo "🚀 [01] prepare data + generate configs"
echo "  📁 DATA_ROOT: $DATA_ROOT"
echo "  🔍 ANCHOR:    ${ANCHOR:-0}"

# ---------------------------------------------------------------------------
# 1. Check dataset structure.
# ---------------------------------------------------------------------------
ERR=0
for phase in train val; do
    for sub in img mask depth parm; do
        d="$DATA_ROOT/$phase/$sub"
        if [ ! -d "$d" ]; then
            echo "  ❌ missing: $d" >&2
            ERR=1
        else
            n=$(find "$d" -mindepth 1 -maxdepth 1 -type d | wc -l)
            echo "  ✅ $phase/$sub  ($n samples)"
        fi
    done
done

if [ "$ERR" -ne 0 ]; then
    echo "" >&2
    echo "❌ Dataset structure incomplete. See:" >&2
    echo "   https://github.com/aipixel/GPS-Gaussian/blob/main/prepare_data/MAKE_DATA.md" >&2
    echo "   Render THuman2.0 scans with prepare_data/render_data.py (taichi_three)." >&2
    exit 1
fi

# Check landmark.json if anchor loss enabled.
if [ "${ANCHOR:-0}" = "1" ]; then
    if [ ! -f "$DATA_ROOT/landmark.json" ]; then
        echo "⚠️ ANCHOR=1 but landmark.json not found at $DATA_ROOT/landmark.json" >&2
        echo "   Run: GPU=0 DATA_ROOT=$DATA_ROOT bash $SCRIPT_DIR/04_gen_landmarks.sh" >&2
        exit 1
    else
        echo "  ✅ landmark.json present (anchor loss enabled)"
    fi
fi

# ---------------------------------------------------------------------------
# 2. Generate config YAMLs via make_config.py.
# ---------------------------------------------------------------------------
echo "📐 [01] generating config YAMLs"
export DATA_ROOT EVA_DIR
python "$SCRIPT_DIR/make_config.py" all
if [ $? -ne 0 ]; then
    echo "❌ FAILED: config generation" >&2
    exit 1
fi

echo ""
echo "🎉 [01] Done. Configs generated at $EVA_DIR/config/"
echo "    pretrain.yaml — for stage 1 (depthnet pretrain)"
echo "    train.yaml    — for stage 2 (full EVA-Gaussian)"
echo "    Next: GPU=0 DATA_ROOT=$DATA_ROOT bash $SCRIPT_DIR/02_pretrain_depth.sh"
