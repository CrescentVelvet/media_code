#!/usr/bin/env bash
# 07_run_capture.sh — 华为 Remy 采集包 → SplatAnything 场景目录（images/ + prior_camera.json）。
#
# 输出目录可直接作为 02_run_inference.sh 的 INPUT（MODE=pose）。
#
# 用法：
#   CAPTURE="/mnt/d/dataset/手部微动样例/01e954de...-佳佳抱枕" N_VIEWS=24 bash 07_run_capture.sh
#   CAPTURE=/path/to/capture SCENE=mytag N_VIEWS=32 bash 07_run_capture.sh
#
# Env:
#   CAPTURE=      采集包根目录（含 transforms.json + image/）——必填
#   SCENE=        输出子目录名，默认取采集包目录名
#   N_VIEWS=24    等间隔抽帧数（2..50）
#   FRAMES=       显式帧号（逗号分隔），优先级高于 N_VIEWS
#   OUT_DIR=      输出场景目录，默认 $SPLATANYTHING_RESULTS_DIR/_scenes/$SCENE
#   MAX_SIDE=0    长边上限，0 = 保持原分辨率
#   NO_UNDISTORT=0
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${N_VIEWS:=24}"
: "${NO_UNDISTORT:=0}"
: "${MAX_SIDE:=0}"

if [ -z "${CAPTURE:-}" ] || [ ! -f "$CAPTURE/transforms.json" ]; then
    echo "❌ ERROR: 需要 CAPTURE 指向含 transforms.json 的采集包（当前: '${CAPTURE:-}'）" >&2
    exit 1
fi
: "${SCENE:=$(basename "$CAPTURE")}"
: "${OUT_DIR:=$SPLATANYTHING_RESULTS_DIR/_scenes/$SCENE}"

echo "🚀 [07] 采集包 → 场景目录"
echo "  📦 采集包: $CAPTURE"
echo "  🖼️  抽帧: ${FRAMES:+帧号 $FRAMES}${FRAMES:-$N_VIEWS 等间隔}"
echo "  💾 输出: $OUT_DIR"

ARGS=(--raw-dir "$CAPTURE" --out-dir "$OUT_DIR" --max-side "$MAX_SIDE")
if [ -n "${FRAMES:-}" ]; then
    ARGS+=(--frames "$FRAMES")
else
    ARGS+=(--n-views "$N_VIEWS")
fi
[ "$NO_UNDISTORT" = "1" ] && ARGS+=(--no-undistort)

python "$SCRIPT_DIR/07_capture_to_scene.py" "${ARGS[@]}" || exit 1

# 用官方校验器验一遍（和推理时同一套相机校验代码）
if [ -f "$SPLATANYTHING_DIR/scripts/convert_cameras.py" ]; then
    python "$SPLATANYTHING_DIR/scripts/convert_cameras.py" check \
        --cameras "$OUT_DIR/prior_camera.json" --images "$OUT_DIR/images" || exit 1
fi

echo "🎉 Done: $OUT_DIR"
echo "   下一步: INPUT=\"$OUT_DIR\" MODE=pose bash $SCRIPT_DIR/02_run_inference.sh"
