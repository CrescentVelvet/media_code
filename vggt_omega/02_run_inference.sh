#!/usr/bin/env bash
# 02_run_inference.sh — 华为 Remy 采集包 → VGGT-Ω（位姿+深度）→ COLMAP 场景
# 链路整体转发 vggt_human 的 02（VGGT-Ω 推理，run_batch.py）+ 03（npz→COLMAP），
# 这两个脚本长期在 vggt_human env 下验证过；本脚本只负责采集包转换与参数转发。
# 输出: $RESULTS_DIR/<场景名>/{02_vggt/predictions.npz, 03_source/(images+sparse/0)}
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

export RAW_DIR="${RAW_DIR:?❌ 请指定采集包路径 RAW_DIR=/mnt/d/dataset/手部微动样例/<包名>}"
export SCENE_NAME="${SCENE_NAME:-$(basename "$RAW_DIR" | sed 's/^[0-9a-f]\{32\}-//' | tr -d '[:space:]')}"
export N_VIEWS="${N_VIEWS:-32}"
export RESOLUTION="${RESOLUTION:-512}"
export CONF_THRES="${CONF_THRES:-20}"

CKPT="$MODEL_DIR/vggt_omega_1b_512.pt"
if [ ! -f "$CKPT" ]; then
    echo "❌ ERROR: 缺 VGGT-Ω 权重 $CKPT（gated，先跑 bash vggt_omega/01_download_models.sh）" >&2
    exit 1
fi

CAP_DIR="$RESULTS_DIR/_captures/$SCENE_NAME"
SCENE_RESULTS="$RESULTS_DIR/$SCENE_NAME"

echo "🚀 [02] VGGT-Ω 重建: $SCENE_NAME"
echo "  📁 采集包: $RAW_DIR（$N_VIEWS 视角 @${RESOLUTION}px）"
echo "  💾 输出:   $SCENE_RESULTS"

# 1) 采集包 → 去畸变 png（infinisplat_v2 转换器，HEIC 解码在 infinisplat_v2 env）
CONVERT_PYTHON="${CONVERT_PYTHON:-$HOME/miniconda3/envs/infinisplat_v2/bin/python}"
[ -x "$CONVERT_PYTHON" ] || CONVERT_PYTHON=python
if [ -d "$CAP_DIR/images" ] && [ "$(ls "$CAP_DIR/images" | wc -l)" -eq "$N_VIEWS" ] && [ "${FORCE_CONVERT:-0}" != "1" ]; then
    echo "⏭️ 转换产物已存在: $CAP_DIR/images"
else
    echo "🔍 转换采集包 → $CAP_DIR"
    "$CONVERT_PYTHON" "$MEDIA_CODE_ROOT/infinisplat_v2/07_capture_to_npz.py" \
        --raw-dir "$RAW_DIR" --out-dir "$CAP_DIR" --n-views "$N_VIEWS"
    if [ $? -ne 0 ]; then echo "❌ 采集包转换失败" >&2; exit 1; fi
fi

# 2) VGGT-Ω 推理（转发 vggt_human/02；显式给 VGGT_DIR/MODEL_DIR，避免依赖 proxy.env 覆盖）
# run_batch 以 INPUT_DIR 的 basename 作为场景名：直接喂 .../images/ 会得到场景名 "images"，
# predictions.npz 落到 02_vggt/images/ 与下游 03/04 的 SCENE_NAME 约定错位。
# 造一层 <SCENE_NAME> 符号链接目录再喂给 run_batch，产物落在 02_vggt/<SCENE_NAME>/ 下。
echo "🤖 VGGT-Ω 前馈推理（vggt_human env）"
VGGT_INPUT_DIR="$SCENE_RESULTS/_vggt_input"
mkdir -p "$VGGT_INPUT_DIR"
ln -sfn "$CAP_DIR/images" "$VGGT_INPUT_DIR/$SCENE_NAME"
CONDA_ENV=vggt_human \
VGGT_DIR="$VGGT_OMEGA_REPO" \
MODEL_DIR="$MODEL_DIR" \
RESULTS_DIR="$SCENE_RESULTS" \
INPUT_DIR="$VGGT_INPUT_DIR" \
VGGT_OUTPUT_DIR="$SCENE_RESULTS/02_vggt" \
RESOLUTION="$RESOLUTION" CONF_THRES="$CONF_THRES" \
bash "$MEDIA_CODE_ROOT/vggt_human/02_run_inference.sh"
if [ $? -ne 0 ]; then echo "❌ FAILED" >&2; exit 1; fi

# 3) npz → COLMAP 场景
echo "✂️ npz → COLMAP"
CONDA_ENV=vggt_human \
VGGT_DIR="$VGGT_OMEGA_REPO" \
MODEL_DIR="$MODEL_DIR" \
RESULTS_DIR="$SCENE_RESULTS" \
VGGT_OUTPUT_DIR="$SCENE_RESULTS/02_vggt" \
SOURCE_DIR="$SCENE_RESULTS/03_source" \
bash "$MEDIA_CODE_ROOT/vggt_human/03_npz_to_colmap.sh"
if [ $? -ne 0 ]; then echo "❌ FAILED" >&2; exit 1; fi

echo "🎉 Done. COLMAP 场景: $SCENE_RESULTS/03_source"
echo "  下一步: RAW_DIR=... bash vggt_omega/03_train_3dgs.sh"
