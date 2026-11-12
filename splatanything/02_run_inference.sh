#!/usr/bin/env bash
# 02_run_inference.sh — 跑 SplatAnything 前馈推理，导出高斯 PLY + 相机 + 深度。
#
# 两种输入：
#   # a) 官方示例（需先检出 examples：git -C $SPLATANYTHING_DIR sparse-checkout add examples）
#   EXAMPLE=object bash 02_run_inference.sh
#   EXAMPLE=indoor MODE=pose RESOLUTION="504 504" bash 02_run_inference.sh
#
#   # b) 自有场景（目录下必须有 images/，pose 模式还需 prior_camera.json）
#   INPUT=/path/to/my_scene bash 02_run_inference.sh
#   INPUT=/path/to/my_scene MODE=pose bash 02_run_inference.sh
#
# 输出：$OUT_ROOT/<scene>/  {gaussians.ply, points.ply, cameras.json, depth/, rgb/, config.yaml, run.json}
#
# Env (all optional):
#   EXAMPLE=            # 官方示例名；与 INPUT 二选一
#   INPUT=              # 场景目录（含 images/）
#   MODE=rgb            # rgb（模型自估位姿）/ pose（读 prior_camera.json）
#   RESOLUTION="756 756"# 模型输入宽高，必须是正的 14 的倍数
#   OUT_ROOT=           # 输出根；默认 $SPLATANYTHING_RESULTS_DIR/<MODE>，已存在则加时间戳
#   SKY_SEG=0           # 1 = 启用天空分割（需 skyseg.onnx）
#   OPACITY_THRESH=     # 覆盖 YAML 的 0.05
#   NO_DEPTH_RGB=0      # 1 = 不导出逐视角 depth/rgb
#   EXTRA_ARGS=         # 透传给 inference.py（如 "--no_cam_align"）
#   PROFILE=1           # 1 = 采样 GPU 峰值显存
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${MODE:=rgb}"
: "${RESOLUTION:=756 756}"
: "${SKY_SEG:=0}"
: "${NO_DEPTH_RGB:=0}"
: "${PROFILE:=1}"

# ---- 前置检查 ----
if [ ! -f "$SPLATANYTHING_MODEL_DIR/SplatAnything.ckpt" ] || \
   [ ! -f "$SPLATANYTHING_MODEL_DIR/worldmirror2.safetensors" ]; then
    echo "❌ ERROR: 权重缺失（$SPLATANYTHING_MODEL_DIR）" >&2
    echo "      先跑: bash $SCRIPT_DIR/01_download_models.sh" >&2
    exit 1
fi
if ! python -c "import torch" >/dev/null 2>&1; then
    echo "❌ ERROR: conda env '$CONDA_ENV' 不可用（先跑 00_setup_env.sh）" >&2
    exit 1
fi

# ---- 解析输入 ----
if [ -n "${EXAMPLE:-}" ]; then
    INPUT="$SPLATANYTHING_DIR/examples/$EXAMPLE"
    if [ ! -d "$INPUT/images" ]; then
        echo "❌ ERROR: 示例 $EXAMPLE 未检出（稀疏克隆默认不含 examples/）" >&2
        echo "      执行: git -C \"$SPLATANYTHING_DIR\" sparse-checkout add examples" >&2
        exit 1
    fi
fi
if [ -z "${INPUT:-}" ] || [ ! -d "$INPUT/images" ]; then
    echo "❌ ERROR: 需要 EXAMPLE 或指向含 images/ 的 INPUT 目录（当前: '${INPUT:-}'）" >&2
    exit 1
fi
INPUT="$(cd "$INPUT" && pwd)"
SCENE="$(basename "$INPUT")"
if [ "$SCENE" = "images" ]; then SCENE="$(basename "$(dirname "$INPUT")")"; fi

if [ "$MODE" = "pose" ] && [ ! -f "$INPUT/prior_camera.json" ]; then
    echo "❌ ERROR: pose 模式要求 $INPUT/prior_camera.json" >&2
    exit 1
fi

: "${OUT_ROOT:=$SPLATANYTHING_RESULTS_DIR/$MODE}"
if [ -e "$OUT_ROOT/$SCENE" ]; then
    OUT_ROOT="${OUT_ROOT}_$(date +%m%d_%H%M%S)"
    echo "⚠️  同名场景输出已存在，改输出根为 $OUT_ROOT"
fi
mkdir -p "$OUT_ROOT"

N=$(find "$INPUT/images" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l)

echo "🚀 [02] SplatAnything 前馈推理"
echo "  🖼️  场景: $INPUT（$N 张图）"
echo "  🎛️  模式: $MODE  分辨率: $RESOLUTION px"
echo "  💾 输出: $OUT_ROOT/$SCENE"
echo "  🏋️  权重: $SPLATANYTHING_MODEL_DIR"

ARGS=(--input_path "$INPUT" --mode "$MODE" --resolution $RESOLUTION --output_path "$OUT_ROOT")
[ "$SKY_SEG" = "1" ] && ARGS+=(--sky_seg --sky_onnx_path "$SPLATANYTHING_MODEL_DIR/skyseg.onnx")
[ -n "${OPACITY_THRESH:-}" ] && ARGS+=(--opacity_thresh "$OPACITY_THRESH")
[ "$NO_DEPTH_RGB" = "1" ] && ARGS+=(--no_depth_rgb)
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)

# ---- 显存采样（推理期间每 0.5 s 记录一次 used）----
PEAK_LOG=""
if [ "$PROFILE" = "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
    PEAK_LOG="$OUT_ROOT/.gpu_peak_$$.log"
    ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) >"$PEAK_LOG" 2>/dev/null &
    SAMPLER_PID=$!
fi

START=$(date +%s)
cd "$SPLATANYTHING_DIR" || exit 1
python inference.py "${ARGS[@]}"
RC=$?
END=$(date +%s)

if [ -n "$PEAK_LOG" ]; then
    kill "$SAMPLER_PID" 2>/dev/null
    wait "$SAMPLER_PID" 2>/dev/null
fi
if [ $RC -ne 0 ]; then
    echo "❌ 推理失败 (exit $RC)" >&2
    exit 1
fi

OUT_DIR="$OUT_ROOT/$SCENE"
echo "⏱️  耗时 $((END - START)) s"
if [ -n "$PEAK_LOG" ] && [ -s "$PEAK_LOG" ]; then
    awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{printf "  🎮 显存峰值: %.2f GB\n", m/1024}' "$PEAK_LOG"
    rm -f "$PEAK_LOG"
fi

python - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
gs = d / 'gaussians.ply'
pts = d / 'points.ply'
cam = d / 'cameras.json'
print(f"  ✅ 高斯: {gs.stat().st_size/1e6:.1f} MB  → {gs}" if gs.exists() else "  ⚠️ 无 gaussians.ply")
if pts.exists():
    print(f"  📦 点云: {pts.stat().st_size/1e6:.1f} MB → {pts}")
if cam.exists():
    n = len(json.loads(cam.read_text()).get('extrinsics', []))
    print(f"  📷 相机: {n}")
run = d / 'run.json'
if run.exists():
    r = json.loads(run.read_text())
    print(f"  🎛️  resolution={r.get('resolution')} mode={'pose' if r.get('pose_condition') else 'rgb'}")
PY

echo "🎉 Done: $OUT_DIR"
