#!/usr/bin/env bash
# 06c_closeup_finetune.sh — 方案二：外插近景视角 HYPIR 增强 + 人脸 finetune
#
# 流程（对应 grill-me 设计定稿）：
#   前提: 04 训完 30k（或 BA 精修后重训的 04b_model_3dgs_ba）
#         SAM2 人脸 mask 已生成（sam2_face_masks.py）
#         人脸 3D 中心已计算（face_center_3d.py → 06c_face_center.json）
#   1. render_closeup.py: 方位角分桶 10 帧 → 向人脸中心推进 → pitch ±10° → 20 视角
#   2. HYPIR 增强这 20 张渲染图（face_enhance.py，只增强人脸区域）
#   3. SAM2 对增强后的近景图做人脸 mask
#   4. train_face_finetune.py: 原训练视角 + 20 近景视角 共同 finetune
#
# Env (all optional, defaults shown):
#   RESULTS_DIR=                 # 输出根
#   GAUSSIAN_DIR=                # 基线模型 (默认 04_model_3dgs 或 04b_model_3dgs_ba)
#   SOURCE_DIR=                  # COLMAP 场景
#   MASKS_DIR=                   # SAM2 人脸 mask (默认 06c_sam2_face_masks)
#   FACE_CENTER=                # 06c_face_center.json (默认 $RESULTS_DIR/06c_face_center.json)
#   CLOSEUP_RENDERS=             # 近景渲染输出 (默认 06c_closeup_renders)
#   CLOSEUP_ENHANCED=            # HYPIR 增强后近景图 (默认 06c_closeup_enhanced/images)
#   CLOSEUP_MASKS=               # 近景图人脸 mask (默认 06c_closeup_masks)
#   GAUSSIAN_CLOSEUP_DIR=        # 输出模型 (默认 06c_model_3dgs_closeup)
#   N_BINS=10                    # 方位角分桶数
#   TARGET_COV=40                # 目标人脸覆盖率 %
#   MIN_DIST_RATIO=0.3           # 最小物距比例
#   PITCH_DEG=10                 # 俯仰角外插度数
#   ITERATION=30000              # 3DGS 迭代
#   FINETUNE_ITERATIONS=35000    # finetune 终点
#   FACE_WEIGHT=0.5              # 人脸监督权重
#   MASK_SOURCE=sam2|geo         # 近景 mask 来源 (geo=方案E 几何 mask)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

RESULTS_DIR="${RESULTS_DIR:-$RESULTS_ROOT}"
GAUSSIAN_DIR="${GAUSSIAN_DIR:-$RESULTS_DIR/04b_model_3dgs_ba}"
SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/03b_source_ba}"
# 近景 mask 来源: sam2 (默认, 旧路径) | geo (方案E SH hack 几何 mask, 模糊近景首选 ——
# 实测 SAM2 中心兜底在模糊近景上会乱检; geo mask 跟随 3D 人的真实位置)
MASK_SOURCE="${MASK_SOURCE:-sam2}"
if [ "$MASK_SOURCE" = "geo" ]; then
    MASKS_DIR="${MASKS_DIR:-$RESULTS_DIR/06c_sam2_face_masks}"          # 训练视角 mask 仍用 SAM2
    CLOSEUP_MASKS="${CLOSEUP_MASKS:-$RESULTS_DIR/06c_closeup_masks_geo}"  # 近景换几何 mask
else
    MASKS_DIR="${MASKS_DIR:-$RESULTS_DIR/06c_sam2_face_masks}"
    CLOSEUP_MASKS="${CLOSEUP_MASKS:-$RESULTS_DIR/06c_closeup_masks}"
fi
FACE_CENTER="${FACE_CENTER:-$RESULTS_DIR/06c_face_center.json}"
CLOSEUP_RENDERS="${CLOSEUP_RENDERS:-$RESULTS_DIR/06c_closeup_renders}"
CLOSEUP_ALPHA="${CLOSEUP_ALPHA:-$RESULTS_DIR/06c_closeup_alpha}"
CLOSEUP_ENHANCED="${CLOSEUP_ENHANCED:-$RESULTS_DIR/06c_closeup_enhanced/images}"
CLOSEUP_MASKS="${CLOSEUP_MASKS:-$RESULTS_DIR/06c_closeup_masks}"
GAUSSIAN_CLOSEUP_DIR="${GAUSSIAN_CLOSEUP_DIR:-$RESULTS_DIR/06c_model_3dgs_closeup}"
N_BINS="${N_BINS:-10}"
TARGET_COV="${TARGET_COV:-40}"
MIN_DIST_RATIO="${MIN_DIST_RATIO:-0.3}"
PITCH_DEG="${PITCH_DEG:-10}"
ITERATION="${ITERATION:-30000}"
FINETUNE_ITERATIONS="${FINETUNE_ITERATIONS:-35000}"
FACE_WEIGHT="${FACE_WEIGHT:-0.5}"

echo "🧑 [06c] 方案二：外插近景视角 + HYPIR 增强 + finetune"
echo "  📂 3DGS model:  $GAUSSIAN_DIR (iter=$ITERATION)"
echo "  📂 scene:       $SOURCE_DIR"
echo "  📂 SAM2 masks:  $MASKS_DIR"
echo "  📐 bins=$N_BINS, target_cov=${TARGET_COV}%, pitch=±${PITCH_DEG}°"
echo ""

# ── Step 1: Render closeup views ──────────────────────────────────────────
if [ ! -d "$CLOSEUP_RENDERS" ] || [ "$(ls "$CLOSEUP_RENDERS"/*.png 2>/dev/null | wc -l)" -lt 2 ]; then
    echo "🔭 Step 1: render closeup views"
    GS_DIR="$GS_DIR" \
    GAUSSIAN_DIR="$GAUSSIAN_DIR" \
    SOURCE_DIR="$SOURCE_DIR" \
    RESULTS_DIR="$RESULTS_DIR" \
    FACE_CENTER="$FACE_CENTER" \
    MASKS_DIR="$MASKS_DIR" \
    N_BINS="$N_BINS" \
    TARGET_COV="$TARGET_COV" \
    MIN_DIST_RATIO="$MIN_DIST_RATIO" \
    PITCH_DEG="$PITCH_DEG" \
    ITERATION="$ITERATION" \
    python "$SCRIPT_DIR/render_closeup.py"
    if [ $? -ne 0 ]; then
        echo "❌ render_closeup failed" >&2
        exit 1
    fi
else
    echo "🔭 Step 1: closeup renders exist, skipping ($CLOSEUP_RENDERS)"
fi
echo ""

# ── Step 2: HYPIR enhance closeup renders ─────────────────────────────────
# face_enhance.py expects INPUT_SOURCE_DIR (with images/ subdir or plain images)
# and SOURCE_FACE_DIR (output root; creates SOURCE_FACE_DIR/images/).
# Our 06c_closeup_renders/ has images directly, so we pass it as INPUT_SOURCE_DIR
# and face_enhance.py will detect it as a plain image folder.
if [ ! -d "$CLOSEUP_ENHANCED" ] || [ "$(ls "$CLOSEUP_ENHANCED"/*.png 2>/dev/null | wc -l)" -lt 2 ]; then
    echo "🎨 Step 2: HYPIR enhance closeup renders"
    mkdir -p "$(dirname "$CLOSEUP_ENHANCED")"
    INPUT_SOURCE_DIR="$CLOSEUP_RENDERS" \
    SOURCE_FACE_DIR="$(dirname "$CLOSEUP_ENHANCED")" \
    HYPIR_BASE_MODEL="$HYPIR_BASE_MODEL" \
    HYPIR_WEIGHT="$HYPIR_WEIGHT" \
    python "$SCRIPT_DIR/face_enhance.py"
    if [ $? -ne 0 ]; then
        echo "❌ HYPIR enhance failed" >&2
        exit 1
    fi
else
    echo "🎨 Step 2: enhanced images exist, skipping ($CLOSEUP_ENHANCED)"
fi
echo ""

# ── Step 3: SAM2 masks for closeup enhanced images ────────────────────────
if [ ! -d "$CLOSEUP_MASKS" ] || [ "$(ls "$CLOSEUP_MASKS"/*.mask.png 2>/dev/null | wc -l)" -lt 2 ]; then
    echo "🎭 Step 3: SAM2 masks for closeup images"
    SAM2_CKPT="$SAM2_CKPT" SAM2_CFG="$SAM2_CFG" \
    python "$SCRIPT_DIR/sam2_face_masks.py" \
        --images_dir "$CLOSEUP_ENHANCED" \
        --output_dir "$CLOSEUP_MASKS"
    if [ $? -ne 0 ]; then
        echo "❌ SAM2 mask generation failed" >&2
        exit 1
    fi
else
    echo "🎭 Step 3: closeup masks exist, skipping ($CLOSEUP_MASKS)"
fi
echo ""

# ── Step 4: Finetune with original + closeup views ───────────────────────
echo "🏋️ Step 4: finetune (original + closeup views)"

# Merge face images: original enhanced + closeup enhanced
MERGED_FACE_IMAGES="$RESULTS_DIR/06c_merged_face_images/images"
mkdir -p "$MERGED_FACE_IMAGES"
# Original enhanced images are in 05_source_aug/images/ (from 01 face_enhance step)
cp "$RESULTS_DIR/05_source_aug/images/"*.png "$MERGED_FACE_IMAGES/" 2>/dev/null || true
cp "$RESULTS_DIR/05_source_aug/images/"*.jpg "$MERGED_FACE_IMAGES/" 2>/dev/null || true
cp "$CLOSEUP_ENHANCED/"*.png "$MERGED_FACE_IMAGES/" 2>/dev/null || true

# Merge masks: original SAM2 + closeup SAM2
MERGED_MASKS="$RESULTS_DIR/06c_merged_face_masks"
mkdir -p "$MERGED_MASKS"
cp "$MASKS_DIR"/*.mask.png "$MERGED_MASKS/" 2>/dev/null || true
cp "$MASKS_DIR"/*.alpha.png "$MERGED_MASKS/" 2>/dev/null || true
cp "$CLOSEUP_MASKS"/*.mask.png "$MERGED_MASKS/" 2>/dev/null || true
cp "$CLOSEUP_MASKS"/*.alpha.png "$MERGED_MASKS/" 2>/dev/null || true

START_PLY="$GAUSSIAN_DIR/point_cloud/iteration_$ITERATION/point_cloud.ply"

TRAIN_FLAGS=(
    -s "$SOURCE_DIR"
    -m "$GAUSSIAN_CLOSEUP_DIR"
    --iterations "$FINETUNE_ITERATIONS"
    --start_ply "$START_PLY"
    --lr_scale 0.1
    --face_images_dir "$MERGED_FACE_IMAGES"
    --face_masks_dir "$MERGED_MASKS"
    --face_weight "$FACE_WEIGHT"
    --port 0
    --disable_viewer
    --test_iterations "$FINETUNE_ITERATIONS"
    --save_iterations "$FINETUNE_ITERATIONS"
)
[ "${WHITE_BG:-0}" = "1" ] && TRAIN_FLAGS+=(--white_background)
[ -n "${RES:-}" ] && TRAIN_FLAGS+=(--resolution "$RES")

( cd "$GS_DIR" && python "$SCRIPT_DIR/train_face_finetune.py" "${TRAIN_FLAGS[@]}" )
if [ $? -ne 0 ]; then
    echo "❌ finetune failed" >&2
    exit 1
fi

echo ""
echo "✅ [06c] Done. closeup-finetuned gaussians:"
echo "  $GAUSSIAN_CLOSEUP_DIR/point_cloud/iteration_$FINETUNE_ITERATIONS/point_cloud.ply"
