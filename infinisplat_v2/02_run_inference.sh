#!/usr/bin/env bash
# 02_run_inference.sh — 跑 InfiniSplat v2 前馈推理，导出 SH0 高斯 PLY。
#
# 三种用法：
#   # a) 官方示例（先跑 01_download_models.sh）
#   EXAMPLE=dtu_provided bash 02_run_inference.sh
#   EXAMPLE=hypersim_provided MAX_VIEWS=50 bash 02_run_inference.sh
#   EXAMPLE=hypersim_da3_pose bash 02_run_inference.sh
#
#   # b) 自有图像 + 相机
#   IMAGES_DIR=/path/to/images CAMERAS=/path/to/cameras.npz OUT_NAME=myscene bash 02_run_inference.sh
#
#   # c) 自有图像、无相机（DA3 先测位姿；POSE_MODE=pose-free 可跳过预检）
#   IMAGES_DIR=/path/to/images POSE_MODE=da3-pose OUT_NAME=myscene bash 02_run_inference.sh
#
# 输出：$RESULTS_DIR/<OUT_NAME>/{scene.ply, cameras.npz, result.json}
#
# Env (all optional):
#   EXAMPLE=            # 官方示例名；与 IMAGES_DIR 二选一
#   IMAGES_DIR=         # 自有图像目录（JPG/PNG/WebP）
#   CAMERAS=            # NPZ: c2w[N,4,4] + intrinsics[N,3,3]（OpenCV c2w，像素内参）
#   POSE_MODE=          # provided / da3-pose / pose-free（有 CAMERAS 时自动 provided）
#   OUT_NAME=           # 输出目录名，默认 = EXAMPLE 或 images 目录名
#   LONG_SIDE=518       # 长边像素，112–2044，对齐 14 的倍数
#   MAX_VIEWS=32        # 上限 2–50；50 视角示例需显式给 50
#   HTML=0              # 1 = 额外导出 scene.html（需 npm splat-transform）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${LONG_SIDE:=518}"
: "${MAX_VIEWS:=32}"
: "${HTML:=0}"

if [ ! -f "$WEIGHTS_DIR/model.safetensors" ] || [ ! -f "$WEIGHTS_DIR/config.json" ]; then
    echo "❌ ERROR: 权重缺失（$WEIGHTS_DIR）" >&2
    echo "      先跑: bash $SCRIPT_DIR/01_download_models.sh" >&2
    exit 1
fi
if ! command -v infinisplat >/dev/null 2>&1; then
    echo "❌ ERROR: infinisplat 命令不在 PATH（先跑 00_setup_env.sh）" >&2
    exit 1
fi

# ---- 解析输入 ----
if [ -n "${EXAMPLE:-}" ]; then
    IMAGES_DIR="$INFINISPLAT_V2/examples/data/$EXAMPLE/images"
    if [ -f "$INFINISPLAT_V2/examples/data/$EXAMPLE/cameras.npz" ]; then
        CAMERAS="$INFINISPLAT_V2/examples/data/$EXAMPLE/cameras.npz"
    fi
    : "${OUT_NAME:=$EXAMPLE}"
fi
if [ -z "${IMAGES_DIR:-}" ] || [ ! -d "$IMAGES_DIR" ]; then
    echo "❌ ERROR: 需要 EXAMPLE 或有效的 IMAGES_DIR（当前: '${IMAGES_DIR:-}'）" >&2
    exit 1
fi
: "${OUT_NAME:=$(basename "$IMAGES_DIR")}"

OUT_DIR="$RESULTS_DIR/$OUT_NAME"
if [ -d "$OUT_DIR" ]; then
    echo "⚠️  输出目录已存在，改名为 ${OUT_NAME}_$(date +%H%M%S)"
    OUT_NAME="${OUT_NAME}_$(date +%H%M%S)"
    OUT_DIR="$RESULTS_DIR/$OUT_NAME"
fi
mkdir -p "$RESULTS_DIR"

N=$(find "$IMAGES_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l)

echo "🚀 [02] InfiniSplat v2 前馈推理"
echo "  🖼️  输入: $IMAGES_DIR ($N 张)"
echo "  📐 长边: $LONG_SIDE  上限视角: $MAX_VIEWS"
echo "  💾 输出: $OUT_DIR"
echo "  🏋️  权重: $WEIGHTS_DIR"

ARGS=(--model "$WEIGHTS_DIR" --images-dir "$IMAGES_DIR" --output "$OUT_DIR"
      --long-side "$LONG_SIDE" --max-views "$MAX_VIEWS")
if [ -n "${CAMERAS:-}" ]; then
    echo "  📷 相机: $CAMERAS (provided)"
    ARGS+=(--cameras "$CAMERAS")
elif [ -n "${POSE_MODE:-}" ]; then
    echo "  📷 相机: $POSE_MODE"
    ARGS+=(--pose-mode "$POSE_MODE")
else
    echo "  📷 相机: da3-pose（默认，DA3 先测位姿）"
fi
[ "$HTML" = "1" ] || ARGS+=(--no-html)

START=$(date +%s.%N)
infinisplat "${ARGS[@]}"
RC=$?
END=$(date +%s.%N)
if [ $RC -ne 0 ]; then
    echo "❌ 推理失败 (exit $RC)" >&2
    exit 1
fi

echo "⏱️  耗时 $(echo "$END - $START" | bc)s"
python - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
meta = json.loads((d / 'result.json').read_text())
st = meta.get('stats', {})
print(f"  ✅ 高斯数: {st.get('final_count'):,}  (raw {st.get('raw_count'):,} → merge {st.get('merged_count'):,})")
print(f"  📦 voxel: {st.get('voxel_size'):.6f} (= median_scale {st.get('median_scale'):.6f} × 1.4)")
ply = d / 'scene.ply'
print(f"  📄 scene.ply {ply.stat().st_size/1e6:.1f} MB")
print(f"  📷 相机数: {len(meta.get('inputs', []))}  pose_mode={meta.get('pose_mode')}")
PY
echo "🎉 Done: $OUT_DIR"
