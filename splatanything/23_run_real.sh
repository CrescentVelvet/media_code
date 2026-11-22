#!/usr/bin/env bash
# 23_run_real.sh -- 两真实近静态包：预注册、SA RGB / Zip RGB 同图前馈、SA pose 辅助轨。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/d/output/results/splatanything_results/n1_real_20261122/protocol_v1}"
PREPARE="${PREPARE:-0}"
RUN_RGB="${RUN_RGB:-1}"
RUN_POSE="${RUN_POSE:-1}"
RUN_ZIP="${RUN_ZIP:-1}"
ONLY_SCENE="${ONLY_SCENE:-}"
GPU="${GPU:-0}"
export GPU ROOT

if [ "$PREPARE" = "1" ]; then
  if [ -e "$ROOT" ]; then echo "❌ 预注册目录已存在，不覆盖: $ROOT" >&2; exit 1; fi
  CONDA_ENV=splatanything source "$SCRIPT_DIR/_env.sh"
  "$CONDA_PREFIX/bin/python" "$SCRIPT_DIR/23_n1_prepare_real.py" || exit 1
  CONDA_ENV=sam3 source "$SCRIPT_DIR/_env.sh"
  "$CONDA_PREFIX/bin/python" "$SCRIPT_DIR/23_n1_prepare_real.py" --landmarks || exit 1
  echo "🎉 预注册与自动 ROI 完成；目检通过后才执行模型。"
  exit 0
fi

if [ ! -f "$ROOT/protocol.json" ] || [ ! -f "$ROOT/roi_review.json" ]; then
  echo "❌ 缺少 protocol.json / roi_review.json；先完成预注册与逐视角 ROI 目检。" >&2
  exit 1
fi
mkdir -p "$ROOT/models"
PYTHON="$HOME/miniconda3/envs/splatanything/bin/python"
"$PYTHON" - "$ROOT/protocol.json" > "$ROOT/model_inputs.txt" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
for pack in p["packages"]:
    print(pack["scene"] + "|" + pack["source_scene"])
PY
[ -s "$ROOT/model_inputs.txt" ] || { echo "❌ 没有可运行的模型输入" >&2; exit 1; }
while IFS='|' read -r SCENE SOURCE_SCENE; do
  [ -n "$SCENE" ] || continue
  # 断点续跑必须显式指定身份；已有输出仍绝不覆盖。
  if [ -n "$ONLY_SCENE" ] && [ "$(basename "$SOURCE_SCENE")" != "$ONLY_SCENE" ] && [ "$SCENE" != "$ONLY_SCENE" ]; then
    continue
  fi
  # 官方 02 以输入目录原名建输出目录；source 子目录的 _source 后缀必须保留。
  SCENE="$(basename "$SOURCE_SCENE")"
  if [ "$RUN_RGB" = "1" ]; then
    export CONDA_ENV=splatanything
    source "$SCRIPT_DIR/_env.sh"
    OUT="$ROOT/models/sa_rgb/$SCENE"
    [ ! -e "$OUT" ] || { echo "❌ 输出已存在: $OUT" >&2; exit 1; }
    echo "🚀 SA RGB-only: $SCENE"
    INPUT="$SOURCE_SCENE" MODE=rgb RESOLUTION="756 756" \
      SPLATANYTHING_MODEL_DIR="${SPLATANYTHING_MODEL_DIR:-/mnt/d/wheel/splatanything_ms}" \
      OUT_ROOT="$ROOT/models/sa_rgb" PROFILE=1 \
      bash "$SCRIPT_DIR/02_run_inference.sh" || exit 1
    [ -f "$OUT/gaussians.ply" ] || { echo "❌ SA 原生 PLY 未生成: $OUT" >&2; exit 1; }
  fi
  if [ "$RUN_POSE" = "1" ]; then
    export CONDA_ENV=splatanything
    source "$SCRIPT_DIR/_env.sh"
    OUT="$ROOT/models/sa_pose/$SCENE"
    [ ! -e "$OUT" ] || { echo "❌ 输出已存在: $OUT" >&2; exit 1; }
    echo "🚀 SA known-pose auxiliary: $SCENE"
    INPUT="$SOURCE_SCENE" MODE=pose RESOLUTION="756 756" \
      SPLATANYTHING_MODEL_DIR="${SPLATANYTHING_MODEL_DIR:-/mnt/d/wheel/splatanything_ms}" \
      OUT_ROOT="$ROOT/models/sa_pose" PROFILE=1 \
      bash "$SCRIPT_DIR/02_run_inference.sh" || exit 1
    [ -f "$OUT/gaussians.ply" ] || { echo "❌ SA pose PLY 未生成: $OUT" >&2; exit 1; }
  fi
  if [ "$RUN_ZIP" = "1" ]; then
    export CONDA_ENV=zipsplat
    source "$SCRIPT_DIR/../zipsplat/_env.sh"
    OUT="$ROOT/models/zip_rgb/$SCENE"
    [ ! -e "$OUT" ] || { echo "❌ 输出已存在: $OUT" >&2; exit 1; }
    echo "🚀 Zip RGB-only / BA=0: $SCENE"
    IMAGES_DIR="$SOURCE_SCENE/images" SCENE_NAME="$SCENE" \
      WEIGHT_PATH="${WEIGHT_PATH:-$HOME/model/zipsplat/zipsplat-da3g-504.tar}" \
      OUT_DIR="$OUT" MAX_VIEWS=3 BA=0 SAVE_FEEDFORWARD=1 RENDER_VIDEO=0 \
      bash "$SCRIPT_DIR/../zipsplat/02_run_inference.sh" || exit 1
    [ -f "$OUT/gaussians_feedforward.ply" ] || { echo "❌ Zip 原生 PLY 未生成: $OUT" >&2; exit 1; }
  fi
done < "$ROOT/model_inputs.txt"
echo "🎉 双身份前馈完成；下一步验证原生 PLY/相机并运行真实外观验收。"
