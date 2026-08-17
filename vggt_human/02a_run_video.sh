#!/usr/bin/env bash
# 02a_run_video.sh — 单个视频 → VGGT-Omega 重建 + COLMAP (快速测视频用).
#
# 02 的单视频测试变体: INPUT_DIR 指向一个视频文件, 跑 VGGT-Omega 推理
# (predictions.npz + scene.ply), 再转 COLMAP (source/). 一条命令出重建结果,
# 便于检查 VGGT-Omega 在视频上的重建质量.
#
# 不影响原 01→02→03→04 流程. COLMAP 输出到 $VGGT_OUTPUT_DIR/<scene>/source/
# (与 predictions.npz 同级), 不与 03 的 $RESULTS_DIR/source/ 冲突.
#
# Prerequisites (same as 02):
#   - INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh (first time)
#   - VGGT-Omega weights downloaded (gated HF; run vggt-omega/01_download_models.sh)
#
# Env (all optional, defaults shown):
#   INPUT_DIR=             # 单个视频文件路径 (.mp4/.mov/.avi/.mkv)
#   MODEL_DIR=             # VGGT-Omega checkpoint (gated)
#   RESULTS_DIR=           # output root
#   VGGT_OUTPUT_DIR=       # inference output (default: $RESULTS_DIR/vggt)
#   VARIANT=1b_512         # checkpoint variant (must match downloaded weights)
#   RESOLUTION=512         # input resolution (1b_256_text -> 256)
#   MODE=balanced          # balanced | max_size
#   CONF_THRES=20          # depth-confidence percentile (0-100)
#   MAX_POINTS=2000000     # cap on scene.ply points (0=none)
#   VIDEO_FPS=2            # frame sampling fps (default 2, vs 02's 1, for more frames)
#   SKIP_COLMAP=0          # 1 = skip npz→COLMAP (inference only, inspect scene.ply)
#   TARGET_POINTS=200000   # COLMAP voxel downsample target
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

INPUT_DIR="${INPUT_DIR:-$VGGT_DIR/examples}"
VGGT_OUTPUT_DIR="${VGGT_OUTPUT_DIR:-$RESULTS_DIR/vggt}"
VARIANT="${VARIANT:-1b_512}"
RESOLUTION="${RESOLUTION:-512}"
MODE="${MODE:-balanced}"
CONF_THRES="${CONF_THRES:-20}"
MAX_POINTS="${MAX_POINTS:-2000000}"
VIDEO_FPS="${VIDEO_FPS:-2}"
SKIP_COLMAP="${SKIP_COLMAP:-0}"
TARGET_POINTS="${TARGET_POINTS:-200000}"

VID_EXTS_RE='\.(mp4|mov|avi|mkv)$'

echo "🚀 [02a] Video → VGGT-Omega reconstruction + COLMAP (single video test)"
echo "  🤖 model:       $MODEL_DIR  (variant=$VARIANT)"
echo "  🎬 input video: $INPUT_DIR"
echo "  💾 output:      $VGGT_OUTPUT_DIR"
echo "  📐 resolution:  $RESOLUTION ($MODE), conf_thres=$CONF_THRES, max_points=$MAX_POINTS"
echo "  🎬 video_fps:   $VIDEO_FPS  (skip_colmap=$SKIP_COLMAP)"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU:          physical $CUDA_VISIBLE_DEVICES"
else
    echo "  🎮 GPU:          default cuda:0"
fi
echo ""

# Sanity checks
if [ ! -d "$VGGT_DIR" ]; then
    echo "❌ ERROR: VGGT-Omega code not found at $VGGT_DIR" >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi
case "$VARIANT" in
    1b_512) CKPT_FILE="vggt_omega_1b_512.pt" ;;
    1b_256_text) CKPT_FILE="vggt_omega_1b_256_text.pt" ;;
    *) echo "❌ ERROR: VARIANT must be 1b_512 | 1b_256_text (got '$VARIANT')" >&2; exit 1 ;;
esac
if [ ! -f "$MODEL_DIR/$CKPT_FILE" ]; then
    echo "❌ ERROR: $MODEL_DIR/$CKPT_FILE missing." >&2
    echo "       Download: VARIANT=$VARIANT bash ../vggt-omega/01_download_models.sh" >&2
    exit 1
fi
if [ ! -f "$INPUT_DIR" ]; then
    echo "❌ ERROR: input video not found: $INPUT_DIR" >&2
    echo "       02a expects a single video file (.mp4/.mov/.avi/.mkv)" >&2
    exit 1
fi
if ! echo "$INPUT_DIR" | grep -qiE "$VID_EXTS_RE"; then
    echo "⚠️ WARNING: '$INPUT_DIR' doesn't look like a video (.mp4/.mov/.avi/.mkv)" >&2
    echo "       02a is for single video test. For image folders use 02_run_inference.sh." >&2
fi

# Scene name = video filename without extension (matches run_batch.py's naming)
SCENE_NAME="$(basename "$INPUT_DIR")"
SCENE_NAME="${SCENE_NAME%.*}"

# --- Stage 1: VGGT-Omega inference (same run_batch.py as 02) ---
export VGGT_DIR MODEL_DIR INPUT_DIR VARIANT RESOLUTION MODE CONF_THRES MAX_POINTS VIDEO_FPS
export OUTPUT_DIR="$VGGT_OUTPUT_DIR"

echo "📦 [02a stage 1] VGGT-Omega inference..."
python "$SCRIPT_DIR/run_batch.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED (inference)" >&2
    exit 1
fi
echo ""

# --- Stage 2: npz → COLMAP conversion (single scene) ---
SCENE_DIR="$VGGT_OUTPUT_DIR/$SCENE_NAME"
NPZ_PATH="$SCENE_DIR/predictions.npz"
FRAMES_DIR="$SCENE_DIR/frames"
SOURCE_DIR="$SCENE_DIR/source"

if [ "$SKIP_COLMAP" = "1" ]; then
    echo "⏭️ [02a stage 2] skipping COLMAP conversion (SKIP_COLMAP=1)"
    echo "🎉 Done. Reconstruction in: $SCENE_DIR"
    echo "  scene.ply — view in MeshLab/SuperSplat"
    exit 0
fi

if [ ! -f "$NPZ_PATH" ]; then
    echo "❌ ERROR: predictions.npz not found at $NPZ_PATH" >&2
    echo "       Inference may have failed; check logs above." >&2
    exit 1
fi

echo "✂️ [02a stage 2] npz → COLMAP"
echo "  📄 npz:    $NPZ_PATH"
echo "  🖼️ frames: $FRAMES_DIR"
echo "  💾 source: $SOURCE_DIR"
echo ""
# For standalone testing (no 3DGS training), disable PoseAdjuster so npz_to_colmap.py
# applies ALIGN (center scene at origin) — useful for inspection. Original 03 leaves
# POSE_ADJUST as-is (default 1) because step 04's PoseAdjuster handles the transform.
export POSE_ADJUST=0

NPZ_PATH="$NPZ_PATH" FRAMES_DIR="$FRAMES_DIR" SOURCE_DIR="$SOURCE_DIR" \
    TARGET_POINTS="$TARGET_POINTS" ALIGN=1 \
    python "$SCRIPT_DIR/npz_to_colmap.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED (COLMAP conversion)" >&2
    exit 1
fi

echo ""
echo "🎉 Done. Reconstruction in: $SCENE_DIR"
echo "    predictions.npz   # raw VGGT-Omega output"
echo "    scene.ply         # point cloud (view in MeshLab/SuperSplat)"
echo "    frames/           # extracted video frames"
echo "    source/           # COLMAP scene (images + sparse/0/)"
