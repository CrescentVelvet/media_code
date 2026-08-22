#!/usr/bin/env bash
# 02_prepare_data.sh — download a DL3DV demo scene and prepare it for ArtiFixer inference.
#
# Runs data_processing.prepare_colmap_artifixer_inputs which:
#   1. prepare  — symlink images, write transforms.json, selected_indices
#   2. reconstruct — 3DGRUT MCMC training (10k iters, GPU-intensive)
#   3. render   — 3DGRUT rendering along source cameras
#   4. scale    — MoGe metric scale alignment
#   5. caption  — Qwen3-VL captioning → Wan text encoder → caption.h5
#
# For your own COLMAP scene, set COLMAP_DIR=/path/to/scene (with images/ + sparse/0/*.bin).
# For a novel camera trajectory, set TRAJECTORY_PATH=/path/to/orbit_360.json.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# Data prep needs network access for Qwen3-VL (if not cached) and MoGe (if not downloaded).
export HF_HUB_OFFLINE=0

# --- params (all overridable via env) ---
# DL3DV demo scene (from the ArtiFixer README)
DL3DV_ROOT="${DL3DV_ROOT:-$REPO_DIR/../DL3DV-ALL-960P}"
DL3DV_SCENE_ID="${DL3DV_SCENE_ID:-15ff83e2531668d27c92091c97d31401ce323e24ee7c844cb32d5109ab9335f7}"
DL3DV_SUBDIR="${DL3DV_SUBDIR:-8K}"

# Output root for prepared scene
PREP_ROOT="${PREP_ROOT:-$RESULTS_DIR/prep}"
SCENE_NAME="${SCENE_NAME:-my_scene}"

# User's own COLMAP scene (if set, skip DL3DV download)
COLMAP_DIR="${COLMAP_DIR:-}"

# Selected training images (optional: newline-delimited file of image names)
SELECTED_IMAGE_NAMES_FILE="${SELECTED_IMAGE_NAMES_FILE:-}"

# Novel camera trajectory (optional: transforms-style JSON)
TRAJECTORY_PATH="${TRAJECTORY_PATH:-}"

# 3DGRUT reconstruction iterations
RECON_STEPS="${RECON_STEPS:-10000}"

# Phases to run (override to skip e.g. caption)
PHASES="${PHASES:-prepare,reconstruct,render,scale,caption}"

# Metric scale (if known, skip MoGe alignment)
METRIC_SCALE="${METRIC_SCALE:-}"

echo "🚀 [02] Prepare scene data"
echo "  📁 output root: $PREP_ROOT"
echo "  📁 scene name:  $SCENE_NAME"
if [ -n "$COLMAP_DIR" ]; then
    echo "  🖼️  colmap dir:   $COLMAP_DIR  (user-provided)"
else
    echo "  🖼️  dl3dv scene:  $DL3DV_SCENE_ID  ($DL3DV_SUBDIR)"
fi
echo "  ⚙️  recon steps:  $RECON_STEPS"
echo "  📝 phases:        $PHASES"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU:          physical $CUDA_VISIBLE_DEVICES"
fi

# --- checks ---
if [ ! -d "$ARTIFIXER_DIR" ]; then
    echo "❌ ERROR: ArtiFixer code not found at $ARTIFIXER_DIR. Run 00_setup_env.sh first." >&2
    exit 1
fi
if ! python -c "import threedgrut" 2>/dev/null; then
    echo "❌ ERROR: threedgrut not importable. Run INSTALL_DEPS=1 bash 00_setup_env.sh" >&2
    exit 1
fi

OUTPUT_ROOT="$PREP_ROOT/$SCENE_NAME"

# --- step 1: obtain COLMAP scene ---
if [ -n "$COLMAP_DIR" ]; then
    # User-provided COLMAP scene
    if [ ! -d "$COLMAP_DIR/images" ]; then
        echo "❌ ERROR: $COLMAP_DIR/images/ not found" >&2
        exit 1
    fi
    if [ ! -f "$COLMAP_DIR/sparse/0/cameras.bin" ]; then
        echo "❌ ERROR: $COLMAP_DIR/sparse/0/cameras.bin not found" >&2
        exit 1
    fi
    echo "✅ using user-provided COLMAP scene: $COLMAP_DIR"
else
    # Download DL3DV demo scene
    DL3DV_ZIP="$DL3DV_ROOT/$DL3DV_SUBDIR/$DL3DV_SCENE_ID.zip"
    if [ ! -f "$DL3DV_ZIP" ]; then
        echo "📦 downloading DL3DV scene..."
        cd "$ARTIFIXER_DIR"
        python scripts/download_dl3dv_scene.py \
            --local-dir "$DL3DV_ROOT" \
            --scene-id "$DL3DV_SCENE_ID" \
            --subdir "$DL3DV_SUBDIR"
        if [ $? -ne 0 ]; then
            echo "❌ FAILED: DL3DV scene download" >&2
            exit 1
        fi
        cd "$SCRIPT_DIR"
    else
        echo "⏭️  DL3DV scene already downloaded: $DL3DV_ZIP"
    fi

    # Extract and find COLMAP structure
    EXTRACT_DIR="$DL3DV_ROOT/$DL3DV_SUBDIR/${DL3DV_SCENE_ID}_extracted"
    if [ ! -d "$EXTRACT_DIR" ]; then
        echo "📦 extracting $DL3DV_ZIP..."
        mkdir -p "$EXTRACT_DIR"
        unzip -q -o "$DL3DV_ZIP" -d "$EXTRACT_DIR"
    fi

    # Find the COLMAP directory (images/ + sparse/0/*.bin)
    COLMAP_DIR=""
    for candidate in "$EXTRACT_DIR" "$EXTRACT_DIR"/*; do
        if [ -d "$candidate/images" ] && [ -f "$candidate/sparse/0/cameras.bin" ]; then
            COLMAP_DIR="$candidate"
            break
        fi
        # DL3DV might have images at a deeper level
        if [ -d "$candidate/$DL3DV_SCENE_ID/images" ] && [ -f "$candidate/$DL3DV_SCENE_ID/sparse/0/cameras.bin" ]; then
            COLMAP_DIR="$candidate/$DL3DV_SCENE_ID"
            break
        fi
    done

    if [ -z "$COLMAP_DIR" ]; then
        echo "❌ ERROR: could not find COLMAP structure (images/ + sparse/0/*.bin) in $EXTRACT_DIR" >&2
        echo "   DL3DV scene may not include COLMAP sparse recon. Run COLMAP on the images first." >&2
        exit 1
    fi
    echo "✅ found COLMAP scene: $COLMAP_DIR"
fi

# --- step 2: run prepare_colmap_artifixer_inputs ---
mkdir -p "$OUTPUT_ROOT"

CMD=(python -m data_processing.prepare_colmap_artifixer_inputs
    --colmap_dir "$COLMAP_DIR"
    --output_root "$OUTPUT_ROOT"
    --text_encoder_model_id "$WAN_MODEL_ID"
    --reconstruction_steps "$RECON_STEPS"
    --phases "$PHASES"
)

if [ -n "$SELECTED_IMAGE_NAMES_FILE" ]; then
    CMD+=(--selected_image_names_file "$SELECTED_IMAGE_NAMES_FILE")
fi
if [ -n "$TRAJECTORY_PATH" ]; then
    CMD+=(--trajectory_path "$TRAJECTORY_PATH")
fi
if [ -n "$METRIC_SCALE" ]; then
    CMD+=(--metric_scale "$METRIC_SCALE")
fi
if [ "${REPLACE_PREP:-0}" = "1" ]; then
    CMD+=(--replace)
fi

echo "🔍 running prepare_colmap_artifixer_inputs..."
echo "  cmd: ${CMD[*]}"

cd "$ARTIFIXER_DIR"
export PYTHONPATH="$ARTIFIXER_DIR:${PYTHONPATH:-}"
"${CMD[@]}"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: prepare_colmap_artifixer_inputs" >&2
    exit 1
fi
cd "$SCRIPT_DIR"

# --- verify output ---
SPLIT_PATH="$OUTPUT_ROOT/split.json"
if [ -f "$SPLIT_PATH" ]; then
    echo "✅ prepared scene: $OUTPUT_ROOT"
    echo "  📁 split:        $SPLIT_PATH"
    echo "  📁 3dgrut input: $OUTPUT_ROOT/3dgrut_input/"
    echo "  📁 recon:        $OUTPUT_ROOT/recon_results/"
    echo "  📁 captions:    $OUTPUT_ROOT/captions/"
    echo "  📁 scale:        $OUTPUT_ROOT/metric_alignment/"
else
    echo "⚠️  split.json not found at $SPLIT_PATH — some phases may have been skipped." >&2
    echo "   Re-run with PHASES=prepare,reconstruct,render,scale,caption for full prep." >&2
fi

echo "🎉 [02] Done."
echo "    Next: bash $SCRIPT_DIR/03_run_inference.sh  (ArtiFixer inference)"
