#!/usr/bin/env bash
# 02_run_inference.sh — render an F3G-Avatar from a trained checkpoint.
#
# Wraps the official `python main_avatar.py -c <yaml> -m test`: fills a DERIVED
# test YAML (no official file is modified) from env vars via gen_test_config.py,
# then launches. Outputs go to test.output_dir (default
# ./test_results/<subject>/<exp>/.../batch_<iter>/). Per the official code:
#   - rgb_map/%08d.jpg  rendered avatar frames
#   - mask_map/%08d.png (alpha) + live_skeleton/ (if render_skeleton) + ply/tex
#     (if save_ply / save_tex_map).
#
# main_avatar.py::test() ALWAYS builds a training dataset from train.data (to
# read SMPL betas + PCA), so DATA_DIR must be the prepared multiview capture
# (has smpl_params.npz + smpl_pos_map/ from `gen_data.gen_pos_maps`). To animate
# an EXTERNAL pose sequence instead, set POSE_DATA (an AMASS/.npz); view_setting
# free then renders a turntable.
#
# Inference also needs (see 01_download_models.sh):
#   - a trained checkpoint dir containing net.pt  (PREV_CKPT)
#   - SMPL-X models in $F3G_DIR/smpl_files/smplx/
#   - the two CUDA extensions built (BUILD_CUDA=1 in 00)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

F3G_DIR="${F3G_DIR:-$REPO_DIR/../F3G-avatar}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/F3G-avatar}"
CKPT_DIR="${CKPT_DIR:-$MODEL_DIR/avatarrex_zzr}"
PREV_CKPT="${PREV_CKPT:-$CKPT_DIR}"

DATA_DIR="${DATA_DIR:-}"
SUBJECT_NAME="${SUBJECT_NAME:-}"
DATA_FRAME_RANGE="${DATA_FRAME_RANGE:-}"   # e.g. 0,2001,1  (train.data + test.data)
POSE_DATA="${POSE_DATA:-}"                  # .npz pose sequence -> free-view animation
POSE_FRAME_RANGE="${POSE_FRAME_RANGE:-}"    # e.g. 2000,2500

VIEW_SETTING="${VIEW_SETTING:-free}"        # free | camera | front | back | moving | cano
RENDER_VIEW_IDX="${RENDER_VIEW_IDX:-13}"
IMG_SCALE="${IMG_SCALE:-1.0}"
SAVE_PLY="${SAVE_PLY:-0}"
SAVE_TEX_MAP="${SAVE_TEX_MAP:-0}"
N_PCA="${N_PCA:-20}"          # <1 disables PCA (vanilla); >=1 enables pose variation
SIGMA_PCA="${SIGMA_PCA:-2.0}"
GLOBAL_ORIENT="${GLOBAL_ORIENT:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-}"   # default: ./test_results/<subject>/<exp>/... per code

echo "=== [02] F3G-Avatar inference (render) ==="
echo "  代码路径:      $F3G_DIR"
echo "  checkpoint:   $PREV_CKPT/net.pt"
echo "  数据路径:      ${DATA_DIR:-<none>}"
echo "  pose_data:    ${POSE_DATA:-<none -> render training frames>}"
echo "  view_setting: $VIEW_SETTING  render_view_idx=$RENDER_VIEW_IDX  img_scale=$IMG_SCALE"
echo "  n_pca=$N_PCA  sigma_pca=$SIGMA_PCA  save_ply=$SAVE_PLY  save_tex_map=$SAVE_TEX_MAP"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:          physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change]"
else
    echo "  GPU:          default cuda:0 (= first visible)  [set GPU=N to pin a card]"
fi

# --- checks ---
if [ ! -d "$F3G_DIR" ]; then
    echo "ERROR: F3G code dir not found at $F3G_DIR. Run run_all.sh first." >&2; exit 1
fi
if [ ! -f "$PREV_CKPT/net.pt" ]; then
    cat >&2 <<EOF
ERROR: checkpoint not found: $PREV_CKPT/net.pt
       Run 01_download_models.sh (for the released checkpoint), or point
       PREV_CKPT at your trained folder, e.g.:
         PREV_CKPT=$F3G_DIR/results/avatarrex_zzr/avatar/batch_700000 bash $0
EOF
    exit 1
fi
if [ -z "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR not set. It must be the prepared multiview capture" >&2
    echo "       (with smpl_params.npz + smpl_pos_map/). See README \"Data Preparation\"." >&2
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR not found: $DATA_DIR" >&2; exit 1
fi
if [ -n "$POSE_DATA" ] && [ ! -f "$POSE_DATA" ]; then
    echo "ERROR: POSE_DATA not found: $POSE_DATA" >&2; exit 1
fi

# SMPL-X presence (the smplx package loads it lazily; warn early).
SMPLX_DIR="$F3G_DIR/smpl_files/smplx"
if [ ! -f "$SMPLX_DIR/SMPLX_NEUTRAL.npz" ] && [ ! -f "$SMPLX_DIR/SMPLX_NEUTRAL.pkl" ]; then
    echo "WARNING: SMPLX_NEUTRAL model not found in $SMPLX_DIR —" >&2
    echo "         run 01_download_models.sh and place it (see README). Inference will fail at load." >&2
fi

# Verify the CUDA extensions are importable (built once with BUILD_CUDA=1).
python - <<'PY' || { echo "ERROR: CUDA extensions not built. Run: BUILD_CUDA=1 bash f3g_avatar/00_setup_env.sh" >&2; exit 1; }
try:
    import diff_gaussian_rasterization  # noqa
    import styleunet  # noqa
except Exception as e:
    raise SystemExit(f"import failed: {e}")
PY

BASE_CONFIG="${BASE_CONFIG:-$F3G_DIR/configs/avatarrex_zzr/avatar.yaml}"
if [ ! -f "$BASE_CONFIG" ]; then
    echo "ERROR: base config not found: $BASE_CONFIG" >&2; exit 1
fi

# --- derive the test YAML (env -> gen_test_config.py -> _test_derived.yaml) ---
OUT_CONFIG="$F3G_DIR/configs/avatarrex_zzr/_test_derived.yaml"
export BASE_CONFIG F3G_DIR DATA_DIR SUBJECT_NAME DATA_FRAME_RANGE PREV_CKPT
export POSE_DATA POSE_FRAME_RANGE VIEW_SETTING RENDER_VIEW_IDX IMG_SCALE
export SAVE_PLY SAVE_TEX_MAP N_PCA SIGMA_PCA GLOBAL_ORIENT OUTPUT_DIR OUT_CONFIG
python "$SCRIPT_DIR/gen_test_config.py"

# --- launch from the repo root (config.PROJ_DIR resolves to cwd) ---
export PYTHONPATH="$F3G_DIR:${PYTHONPATH:-}"
echo "--- launching main_avatar.py -m test ---"
( cd "$F3G_DIR" && python main_avatar.py -c "$OUT_CONFIG" -m test )

echo "=== [02] Done. Renders under: ${OUTPUT_DIR:-./test_results/<subject>/<exp>/...} ==="
