#!/usr/bin/env bash
# 01_download_models.sh — obtain F3G-Avatar weights & assets:
#   1. Pretrained avatar checkpoint  -> $MODEL_DIR/avatarrex_zzr/net.pt
#      (from HuggingFace wjmenu/F3G-avatar, file checkpoints/avatarrex_zzr/epoch_latest.pt)
#   2. SMPL-X body models             -> $F3G_DIR/smpl_files/smplx/   (gated, manual)
#   3. (optional) external repos for the MHR template pipeline:
#        othercode/{NeuS2,4d-dress,PhysAvatar,StyleAvatar}   (set CLONE_EXTRA=1)
#
# The avatar checkpoint is the saved net.pt dict ({avatar_net, epoch_idx,
# iter_idx}); main_avatar.py::load_ckpt reads <prev_ckpt>/net.pt, so the file
# is placed as $MODEL_DIR/avatarrex_zzr/net.pt and inference points
# test.prev_ckpt at that directory.
#
# ⚠️ As of writing the HF model card lists the checkpoint as "Coming soon" — the
# download will 404 until it is published. The script then prints instructions
# to train from scratch (see README) and continues (exit 0) so the SMPL-X check
# and CLONE_EXTRA steps still run. Inference (02) will error clearly if no
# checkpoint is present.
#
# SMPL-X is GATED (https://smpl-x.is.tue.mpg.de/download.php, license login) —
# there is NO public scriptable mirror, so this script CANNOT auto-download it.
# It checks for the expected file and prints manual instructions if missing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

F3G_DIR="${F3G_DIR:-$REPO_DIR/../F3G-avatar}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/F3G-avatar}"

HF_REPO="${HF_REPO:-wjmenu/F3G-avatar}"
CKPT_FILE="${CKPT_FILE:-checkpoints/avatarrex_zzr/epoch_latest.pt}"
CKPT_DIR="${CKPT_DIR:-$MODEL_DIR/avatarrex_zzr}"
NET_PT="${NET_PT:-$CKPT_DIR/net.pt}"

# SMPL-X: code loads smplx.SMPLX(model_path = config.PROJ_DIR + '/smpl_files/smplx',
# gender='neutral', use_pca=False) -> expects SMPLX_NEUTRAL.npz (or .pkl) here.
SMPLX_DIR="${SMPLX_DIR:-$F3G_DIR/smpl_files/smplx}"

echo "=== [01] Downloading F3G-Avatar weights & assets ==="
echo "  代码路径:        $F3G_DIR"
echo "  模型路径:        $MODEL_DIR"
echo "  权重仓库:        $HF_REPO -> $CKPT_FILE"
echo "  checkpoint:     $NET_PT  (prev_ckpt dir: $CKPT_DIR)"
echo "  SMPL-X 目录:    $SMPLX_DIR"

if [ ! -d "$F3G_DIR" ]; then
    echo "ERROR: F3G code dir not found at $F3G_DIR." >&2
    echo "       Run run_all.sh first (it clones the official repo), or set F3G_DIR." >&2
    exit 1
fi

mkdir -p "$MODEL_DIR" "$CKPT_DIR"

# ---------------------------------------------------------------------------
# 1) Pretrained avatar checkpoint (HuggingFace).
# ---------------------------------------------------------------------------
CKPT_FOUND=0
if [ -f "$NET_PT" ]; then
    echo "--- checkpoint already present: $NET_PT ($(du -h "$NET_PT" | cut -f1)) ---"
    CKPT_FOUND=1
fi

if [ "$CKPT_FOUND" = "0" ]; then
    if ! python -c "import huggingface_hub" 2>/dev/null; then
        echo "ERROR: huggingface_hub not installed in env '$CONDA_ENV'. Install with: pip install huggingface_hub" >&2
        exit 1
    fi

    echo "--- [1/3] downloading checkpoint: $HF_REPO / $CKPT_FILE ---"
    TMP_DIR="$(mktemp -d)"
    TOKEN_ARGS=(); [ -n "${HF_TOKEN:-}" ] && TOKEN_ARGS=(--token "$HF_TOKEN")
    if [ "${HF_DISABLE_SSL:-0}" = "1" ]; then
        echo "    (SSL verification DISABLED via HF_DISABLE_SSL=1)"
        python "$SCRIPT_DIR/_hf_download.py" "$HF_REPO" "$CKPT_FILE" "$TMP_DIR" "${HF_TOKEN:-}" || true
    else
        if ! hf download "$HF_REPO" "$CKPT_FILE" "${TOKEN_ARGS[@]}" --local-dir "$TMP_DIR"; then
            echo "    --- hf download failed (likely SSL on CDN, or checkpoint not yet released); retrying with SSL verification disabled ---"
            python "$SCRIPT_DIR/_hf_download.py" "$HF_REPO" "$CKPT_FILE" "$TMP_DIR" "${HF_TOKEN:-}" || true
        fi
    fi

    # snapshot places the file at $TMP_DIR/$CKPT_FILE (relative path preserved).
    SRC="$TMP_DIR/$CKPT_FILE"
    if [ -f "$SRC" ]; then
        cp "$SRC" "$NET_PT"
        echo "--- checkpoint OK: $NET_PT ($(du -h "$NET_PT" | cut -f1)) ---"
        CKPT_FOUND=1
    else
        echo "WARNING: checkpoint not downloaded (likely not yet released on HF)." >&2
    fi
    rm -rf "$TMP_DIR"
fi

if [ "$CKPT_FOUND" = "0" ]; then
    cat >&2 <<EOF

================================================================================
  Pretrained avatar checkpoint NOT found: $NET_PT

  The HF model card (https://huggingface.co/wjmenu/F3G-avatar) lists the
  released checkpoint as "Coming soon" — it may not be published yet.

  Until it is available, TRAIN FROM SCRATCH on your own multiview capture:
    1. Prepare data (face crops + MHR template + pose maps) — see README
       "Data Preparation".
    2. Train:
         python main_avatar.py -c configs/avatarrex_zzr/avatar.yaml -m train
       (edit the YAML: train.data.data_dir -> your dataset)
    3. Your checkpoint lands at results/avatarrex_zzr/avatar/batch_<N>/net.pt
       -> run inference with PREV_CKPT pointing at that folder:
         PREV_CKPT=results/avatarrex_zzr/avatar/batch_<N> bash f3g_avatar/02_run_inference.sh

  If you mirrored the file elsewhere, point CKPT_FILE / HF_REPO at it and rerun.
================================================================================
EOF
fi

# ---------------------------------------------------------------------------
# 2) SMPL-X body models (gated, manual).
# ---------------------------------------------------------------------------
echo "--- [2/3] checking SMPL-X models in $SMPLX_DIR ---"
SMPLX_OK=0
for f in SMPLX_NEUTRAL.npz SMPLX_NEUTRAL.pkl; do
    if [ -f "$SMPLX_DIR/$f" ]; then SMPLX_OK=1; break; fi
done
if [ "$SMPLX_OK" = "1" ]; then
    echo "    SMPL-X neutral model found: OK"
else
    cat >&2 <<EOF

================================================================================
  SMPL-X model NOT found in: $SMPLX_DIR

  SMPL-X is GATED (license) — no public scriptable mirror. Download manually:
    1. Register / log in at https://smpl-x.is.tue.mpg.de/download.php
    2. Download the "SMPL-X" package (with SMPL+H + DMG, or the plain SMPL-X).
       You need at least the neutral model. Place these under:
         $SMPLX_DIR/
       (expected file: SMPLX_NEUTRAL.npz — also SMPLX_MALE/FEMALE if you use
        other genders; the code uses gender='neutral'.)
    3. Re-run this script (it will detect the file and pass).

  The mano/ text files in smpl_files/ are already in the repo (small aux assets).
================================================================================
EOF
fi

# ---------------------------------------------------------------------------
# 3) (optional) external repos for the MHR template pipeline.
# ---------------------------------------------------------------------------
if [ "${CLONE_EXTRA:-0}" = "1" ]; then
    echo "--- [3/3] cloning external repos for the MHR template pipeline ---"
    mkdir -p "$F3G_DIR/othercode"
    clone_if_absent() {
        local url="$1" dst="$2"
        if [ -d "$dst" ]; then
            echo "    $dst already present"
        else
            git clone --recursive "$url" "$dst" || \
                git -c http.sslVerify=false clone --recursive "$url" "$dst"
        fi
    }
    clone_if_absent https://github.com/19reborn/NeuS2.git        "$F3G_DIR/othercode/NeuS2"
    clone_if_absent https://github.com/eth-ait/4d-dress.git     "$F3G_DIR/othercode/4d-dress"
    clone_if_absent https://github.com/y-zheng18/PhysAvatar.git "$F3G_DIR/othercode/PhysAvatar"
    clone_if_absent https://github.com/LizhenWangT/StyleAvatar.git "$F3G_DIR/othercode/StyleAvatar"
    echo "    (NeuS2 still needs a cmake build; 4D-Dress needs Graphonomy/SAM"
    echo "     checkpoints — see README \"Data Preparation\".)"
else
    echo "--- [3/3] SKIPPED external repos (set CLONE_EXTRA=1 to clone NeuS2/4D-Dress/PhysAvatar/StyleAvatar) ---"
fi

echo "=== [01] Done. Weights at: $MODEL_DIR ==="
echo "    checkpoint: $NET_PT"
echo "    SMPL-X:     $SMPLX_DIR"
