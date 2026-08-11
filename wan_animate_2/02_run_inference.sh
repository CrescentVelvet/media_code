#!/usr/bin/env bash
# 02_run_inference.sh — Wan-Animate-2 character animation inference.
# Takes a reference image + driving (template) video + prompt -> animated video.
#
# Two model variants:
#   base (default):        wan_animate_2.yaml,            40 steps, guide 3.0
#   distillation (fast):   wan_animate_2_distillation.yaml, 10 steps, guide 1.0
#
# Multi-GPU: leave GPU unset to use all visible cards (sp_size/sharding_size auto
# = visible GPU count). For single-card set GPU=0 (sp_size=1, may need lower res).
# Official: 8×A800 for 720P, 2×A800 for 480P.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# --- params (all overridable via env) ---
MODEL_VARIANT="${MODEL_VARIANT:-base}"     # base | distillation
PROMPT="${PROMPT:-人物外观描述：一只银灰色虎斑纹的小猫，拥有圆润的脸庞、竖立的耳朵和巨大的圆形眼睛。它身穿一套深蓝色的制服套装，包括一件带有金色纽扣的西装外套和一条百褶裙。外套里面搭配着白色衬衫，领口处系着一个红色的蝴蝶结，袖口露出白色的衬衫边缘。背景描述：背景为纯白色，光线均匀明亮，无其他杂物或装饰。}"
PROMPT_REF="${PROMPT_REF:-人物动作的参考视频}"

# Default reference image / driving video = official examples (cloned with the repo).
REFER_IMAGE="${REFER_IMAGE:-$OFFICIAL_DIR/examples/demo1/reference.png}"
REFER_VIDEO="${REFER_VIDEO:-$OFFICIAL_DIR/examples/demo1/template.mp4}"
# Fallback: if template.mp4 missing, pick any mov/mp4 in examples/demo1.
if [ ! -f "$REFER_VIDEO" ] && [ -d "$OFFICIAL_DIR/examples/demo1" ]; then
    _fv="$(ls "$OFFICIAL_DIR/examples/demo1/"*.mp4 "$OFFICIAL_DIR/examples/demo1/"*.mov 2>/dev/null | head -1)"
    [ -n "$_fv" ] && REFER_VIDEO="$_fv"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR}"
OUTPUT_NAME="${OUTPUT_NAME:-animate}"

# video shape / sampling (defaults differ by variant; overridden below for distillation)
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-800}"
FPS="${FPS:-24}"
CLIP_LEN="${CLIP_LEN:-81}"
SEED="${SEED:--1}"

if [ "$MODEL_VARIANT" = "distillation" ]; then
    STEP="${STEP:-10}"
    SAMPLE_GUIDE_SCALE="${SAMPLE_GUIDE_SCALE:-1.0}"
    YAML_NAME="wan_animate_2_distillation.yaml"
else
    STEP="${STEP:-40}"
    SAMPLE_GUIDE_SCALE="${SAMPLE_GUIDE_SCALE:-3.0}"
    YAML_NAME="wan_animate_2.yaml"
fi
YAML_PATH="$OFFICIAL_DIR/infer/$YAML_NAME"

echo "=== [02] Wan-Animate-2 inference ($MODEL_VARIANT) ==="
echo "  🤖 代码路径:   $OFFICIAL_DIR"
echo "  🏋️ 权重:       $CKPTS_DIR  (-> $OFFICIAL_DIR/ckpts)"
echo "  🎯 变体:       $MODEL_VARIANT  ($YAML_NAME)"
echo "  🖼️ 参考图:     $REFER_IMAGE"
echo "  🎬 驱动视频:   $REFER_VIDEO"
echo "  prompt:       $PROMPT"
echo "  📐 分辨率:     ${WIDTH}x${HEIGHT}  clip_len=$CLIP_LEN  fps=$FPS"
echo "  step=$STEP  guide_scale=$SAMPLE_GUIDE_SCALE  seed=$SEED"
echo "  💾 输出:       $OUTPUT_DIR/${OUTPUT_NAME}.mp4"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU:        physical $CUDA_VISIBLE_DEVICES"
else
    echo "  🎮 GPU:        all visible (multi-GPU sp/sharding)"
fi

# --- checks ---
if [ ! -d "$OFFICIAL_DIR" ]; then
    echo "❌ ERROR: official repo not found at $OFFICIAL_DIR. Run run_all.sh or 00_setup_env.sh first." >&2; exit 1
fi
if ! python -c "import core" 2>/dev/null; then
    echo "❌ ERROR: 'import core' failed in env '$CONDA_ENV'. Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2; exit 1
fi
if [ ! -f "$REFER_IMAGE" ]; then
    echo "❌ ERROR: reference image not found: $REFER_IMAGE" >&2; exit 1
fi
if [ ! -f "$REFER_VIDEO" ]; then
    echo "❌ ERROR: driving video not found: $REFER_VIDEO" >&2; exit 1
fi
if [ ! -f "$YAML_PATH" ]; then
    echo "❌ ERROR: config YAML not found: $YAML_PATH" >&2; exit 1
fi
if [ ! -e "$OFFICIAL_DIR/ckpts" ]; then
    echo "❌ ERROR: ckpts symlink missing at $OFFICIAL_DIR/ckpts. Run: bash $SCRIPT_DIR/01_download_models.sh" >&2; exit 1
fi

mkdir -p "$OUTPUT_DIR"

export MODEL_VARIANT PROMPT PROMPT_REF REFER_IMAGE REFER_VIDEO
export OUTPUT_DIR OUTPUT_NAME WIDTH HEIGHT FPS CLIP_LEN SEED
export STEP SAMPLE_GUIDE_SCALE YAML_PATH OFFICIAL_DIR

python "$SCRIPT_DIR/run_inference.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "=== [02] Done. Video at: $OUTPUT_DIR/${OUTPUT_NAME}.mp4 ==="
