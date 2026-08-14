#!/usr/bin/env bash
# 03_train_pdfgs.sh — PDF-GS (CVPR 2026 Findings) progressive distractor filtering
# for robust 3D Gaussian Splatting on human micro-motion.
#
# Consumes the COLMAP scene from step 02 (Pi3 poses + dense cloud). Trains a 3D
# Gaussian model in N phases; each phase computes DINOv3-feature similarity between
# the previous phase's RENDER and the GT image — pixels below --sim_thr (the
# micro-motion: breathing / hair / clothing that can never be reconstructed to
# match all views) are masked out of the L1+SSIM loss. --sim_thr rises each phase
# (0.6 → 0.7 → 0.8), so distractor filtering gets progressively stricter. The
# result is a clean static-body reconstruction with micro-motion regions excluded.
#
# No mesh in v1: PDF-GS ships only train.py / render.py / metrics.py (no
# extract_mesh). Output is the gaussian point cloud + rendered novel views + (opt)
# PSNR/SSIM/LPIPS. For a mesh, run wan22_rotate step 05/05a/05b instead (or on
# this gaussians — future work).
#
# Why real photos (not Wan2.2 rotate video): the distractor model assumes SPARSE
# outliers among mostly-static pixels — real body micro-motion fits this. A
# synthetic rotate video's drift is PERVASIVE temporal inconsistency (every frame
# is the model's guess), which violates the sparsity assumption and may
# over-filter legitimate regions. See README.md for the full rationale.
#
# Prerequisites:
#   - INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh (first time)
#   - step 01 (segmented_frames/) + step 02 (source/) already run
#
# Env (all optional, defaults shown):
#   OUTPUT_NAME=orbit          # base name (must match step 02)
#   RESULTS_DIR=               # output root
#   SOURCE_DIR=                # COLMAP scene (default: $RESULTS_DIR/<name>/pi3/source)
#   GAUSSIAN_DIR=              # gaussians output (default: $RESULTS_DIR/<name>/model_pdfgs)
#   NUM_PHASES=4               # progressive filtering phases
#   ITER_PER_PHASE=10000      # iters per phase (total = NUM_PHASES × this)
#   SIM_THR=0.6 0.7 0.8        # per-transition thresholds (len = NUM_PHASES-1, OR single value)
#   COLOR_UPDATE_INTERVAL=30  # how often to update SH color (phase != last)
#   WHITE_BG=1                 # 1=white rasterizer bg (matches segmented input)
#   RES=                       # --resolution factor (UNSET = full-res for human photos;
#                              #   README's -r 8 is for RobustSplat benchmark downsampling)
#   SKIP_TRAIN=0              # 1 = reuse existing model_pdfgs/
#   SKIP_RENDER=0             # 1 = skip novel-view rendering
#   SKIP_METRICS=1            # 1 = skip PSNR/SSIM/LPIPS (no held-out test set makes
#                              #   PSNR just train-fit; set 0 to run anyway)
#   TRAIN_EXTRA_ARGS=         # extra args passed verbatim to train.py
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

OUTPUT_NAME="${OUTPUT_NAME:-orbit}"
SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/$OUTPUT_NAME/pi3/source}"
# ⚠️ _env.sh already exports MODEL_DIR as the WEIGHT root ($REPO_DIR/../../model).
#    Using ${MODEL_DIR:-default} would keep the weight root (bug: gaussians written
#    to weight dir). Use a separate GAUSSIAN_DIR to avoid the name collision.
GAUSSIAN_DIR="${GAUSSIAN_DIR:-$RESULTS_DIR/$OUTPUT_NAME/model_pdfgs}"
NUM_PHASES="${NUM_PHASES:-4}"
ITER_PER_PHASE="${ITER_PER_PHASE:-10000}"
SIM_THR="${SIM_THR:-0.6 0.7 0.8}"
COLOR_UPDATE_INTERVAL="${COLOR_UPDATE_INTERVAL:-30}"
WHITE_BG="${WHITE_BG:-1}"
RES="${RES:-}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_RENDER="${SKIP_RENDER:-0}"
SKIP_METRICS="${SKIP_METRICS:-1}"

# Final-phase output (train.py saves each phase under $GAUSSIAN_DIR/phase_<N>/).
PHASE_MODEL="$GAUSSIAN_DIR/phase_$NUM_PHASES"

echo "🚀 [03] PDF-GS progressive distractor filtering"
echo "  🤖 PDF-GS:        $PDFGS_DIR"
echo "  📂 source:        $SOURCE_DIR"
echo "  💾 model:         $GAUSSIAN_DIR"
echo "  📐 phases:        $NUM_PHASES × $ITER_PER_PHASE iters  (sim_thr: $SIM_THR)"
echo "  🎨 white_bg:      $WHITE_BG  color_update_interval: $COLOR_UPDATE_INTERVAL"
[ -n "$RES" ] && echo "  📐 resolution:    $RES"
echo "  🏋️ final phase:   $PHASE_MODEL"
echo ""

# ── 0. Sanity checks ──────────────────────────────────────────────────────
if [ ! -f "$PDFGS_DIR/train.py" ]; then
    echo "❌ ERROR: PDF-GS repo not found at $PDFGS_DIR (no train.py)." >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi
if [ ! -d "$SOURCE_DIR/images" ] || [ ! -d "$SOURCE_DIR/sparse/0" ]; then
    echo "❌ ERROR: COLMAP scene not ready: $SOURCE_DIR" >&2
    echo "       Expected: $SOURCE_DIR/images/ + $SOURCE_DIR/sparse/0/*.txt" >&2
    echo "       Run step 02 first: INPUT=... bash $SCRIPT_DIR/02_pi3_colmap.sh" >&2
    exit 1
fi
if ! python -c "import diff_gaussian_rasterization, simple_knn" 2>/dev/null; then
    echo "❌ ERROR: PDF-GS CUDA extensions not importable." >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh (needs nvcc)" >&2
    exit 1
fi
# DINOv3: DINOV3_REPO (from _env.sh) is either a local dir (e.g. vitl16, loads
# offline without token) OR the gated HF repo id (loaded from $HF_HOME/hub cache).
# Also verify train.py was patched to honor DINOV3_REPO (00 does this idempotently).
if [ -d "$DINOV3_REPO" ]; then
    echo "  🏋️ DINOv3 local: $DINOV3_REPO"
elif [ -n "$(ls -A "$HF_HOME/hub" 2>/dev/null)" ]; then
    echo "  🏋️ DINOv3 HF cache populated ($DINOV3_REPO)"
else
    echo "⚠️ WARNING: DINOv3 not found (DINOV3_REPO=$DINOV3_REPO, HF cache empty)." >&2
    echo "       Put a local dir at $MODEL_DIR/dinov3-vitl16-pretrain-lvd1689m, or" >&2
    echo "       HF_TOKEN=hf_xxx INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
fi
if ! grep -qF "DINOV3_REPO" "$PDFGS_DIR/train.py" 2>/dev/null; then
    echo "⚠️ WARNING: train.py not patched to honor DINOV3_REPO — re-run INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
fi

# ── 1. PDF-GS training (progressive distractor filtering) ─────────────────
if [ "$SKIP_TRAIN" = "1" ]; then
    echo "⏭️ skip training (SKIP_TRAIN=1, reusing existing $GAUSSIAN_DIR)"
    if [ ! -f "$PHASE_MODEL/point_cloud/iteration_$ITER_PER_PHASE/point_cloud.ply" ]; then
        echo "❌ ERROR: final-phase gaussians not found at $PHASE_MODEL" >&2
        echo "       Run step 03 without SKIP_TRAIN first." >&2
        exit 1
    fi
else
    mkdir -p "$GAUSSIAN_DIR"
    echo "🏋️ PDF-GS training ($NUM_PHASES phases × $ITER_PER_PHASE iters)"
    echo "  📂 source:    $SOURCE_DIR"
    echo "  💾 model:     $GAUSSIAN_DIR"
    echo "  ✂️ sim_thr:   $SIM_THR  (distractor filter thresholds, rising per phase)"
    echo ""
    TRAIN_FLAGS=(
        -s "$SOURCE_DIR"
        -m "$GAUSSIAN_DIR"
        --num_phases "$NUM_PHASES"
        --iter_per_phase "$ITER_PER_PHASE"
        # shellcheck disable=SC2206
        --sim_thr $SIM_THR
        --color_update_interval "$COLOR_UPDATE_INTERVAL"
        --port 0
        --disable_viewer
    )
    [ "$WHITE_BG" = "1" ] && TRAIN_FLAGS+=(--white_background)
    [ -n "$RES" ] && TRAIN_FLAGS+=(--resolution "$RES")
    if [ -n "${TRAIN_EXTRA_ARGS:-}" ]; then
        # shellcheck disable=SC2206
        TRAIN_FLAGS+=($TRAIN_EXTRA_ARGS)
    fi
    # Run inside PDF-GS repo for relative imports (scene/, gaussian_renderer/, arguments/)
    ( cd "$PDFGS_DIR" && python train.py "${TRAIN_FLAGS[@]}" )
    if [ $? -ne 0 ]; then
        echo "❌ FAILED. PDF-GS training did not complete." >&2
        exit 1
    fi
    echo "✅ PDF-GS training done"
    echo "  🏋️ final gaussians: $PHASE_MODEL/point_cloud/iteration_$ITER_PER_PHASE/point_cloud.ply"
    echo ""
fi

# ── 2. Rendering (novel views / reconstruction-vs-GT) ────────────────────
# render.py loads gaussians from $PHASE_MODEL (final phase) + cameras from
# $SOURCE_DIR. Renders all training views (no held-out test split without --eval,
# so scene.getTestCameras() is empty and the "test" set is skipped automatically).
if [ "$SKIP_RENDER" = "1" ]; then
    echo "⏭️ skip rendering (SKIP_RENDER=1)"
else
    if [ ! -f "$PHASE_MODEL/point_cloud/iteration_$ITER_PER_PHASE/point_cloud.ply" ]; then
        echo "❌ ERROR: final-phase gaussians not found — cannot render." >&2
        echo "       Run training first or: SKIP_TRAIN=0 bash $0" >&2
        exit 1
    fi
    echo "🖼️ PDF-GS rendering (train views)"
    echo "  💾 model:     $PHASE_MODEL"
    echo "  📐 iteration: $ITER_PER_PHASE"
    echo ""
    RENDER_FLAGS=(
        -s "$SOURCE_DIR"
        -m "$PHASE_MODEL"
        --iteration "$ITER_PER_PHASE"
    )
    [ "$WHITE_BG" = "1" ] && RENDER_FLAGS+=(--white_background)
    if [ -n "${RENDER_EXTRA_ARGS:-}" ]; then
        # shellcheck disable=SC2206
        RENDER_FLAGS+=($RENDER_EXTRA_ARGS)
    fi
    ( cd "$PDFGS_DIR" && python render.py "${RENDER_FLAGS[@]}" )
    if [ $? -ne 0 ]; then
        echo "⚠️ render.py failed (continuing — renders are non-fatal)." >&2
    else
        echo "✅ rendering done"
        echo "  🖼️ renders: $PHASE_MODEL/train/ours_$ITER_PER_PHASE/renders/*.png"
        echo "  🖼️ gt:      $PHASE_MODEL/train/ours_$ITER_PER_PHASE/gt/*.png"
        echo ""
    fi
fi

# ── 3. Metrics (optional; PSNR/SSIM/LPIPS) ───────────────────────────────
# Skipped by default: without a held-out test split, PSNR measures only train-fit
# (less informative). Set SKIP_METRICS=0 to run.
if [ "$SKIP_METRICS" = "1" ]; then
    echo "⏭️ skip metrics (SKIP_METRICS=1; no held-out test set → PSNR is just train fit)"
else
    echo "📊 PDF-GS metrics (PSNR / SSIM / LPIPS)"
    echo "  💾 model: $PHASE_MODEL"
    echo ""
    METRIC_FLAGS=(-m "$PHASE_MODEL")
    if [ -n "${METRICS_EXTRA_ARGS:-}" ]; then
        # shellcheck disable=SC2206
        METRIC_FLAGS+=($METRICS_EXTRA_ARGS)
    fi
    ( cd "$PDFGS_DIR" && python metrics.py "${METRIC_FLAGS[@]}" )
    if [ $? -ne 0 ]; then
        echo "⚠️ metrics.py failed (non-fatal)." >&2
    else
        echo "✅ metrics done"
        echo "  📊 results: $PHASE_MODEL/results.json (or $PHASE_MODEL/*.json)"
        echo ""
    fi
fi

echo "🎉 [03] Done. PDF-GS reconstruction complete."
echo "  🏋️ Gaussians:  $PHASE_MODEL/point_cloud/iteration_$ITER_PER_PHASE/point_cloud.ply"
echo "  🖼️ Renders:    $PHASE_MODEL/train/ours_$ITER_PER_PHASE/renders/*.png"
echo ""
echo "  🎬 Showcase: render a turntable/orbit video of the person:"
echo "     GPU=0 RESULTS_DIR=$RESULTS_DIR bash $SCRIPT_DIR/04_render_orbit.sh"
echo ""
echo "  Inspect gaussians: https://playcanvas.com/supersplat/editor (drag .ply)"
echo "  Or MeshLab:        meshlab $PHASE_MODEL/point_cloud/iteration_$ITER_PER_PHASE/point_cloud.ply"
echo ""
echo "  💡 v1 has NO mesh (PDF-GS ships no extract_mesh). For a mesh, run"
echo "     wan22_rotate step 05/05a/05b, or add a TSDF-on-depth step (future)."
