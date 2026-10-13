#!/usr/bin/env bash
# 01_convert_capture.sh — 手机采集格式（transforms.json + image/*.heic + pcd.ply）
# 转 D-3DGS COLMAP 场景。与 deformable_human/01_prepare_data.sh 平级：
# 那个面向「视频→COLMAP 重建」，这个面向「手机 App 已带位姿/点云」的采集包。
#
# 输入：RAW_DIR/{transforms.json, image/*.heic, pcd.ply}（/mnt/d 只读访问即可）
# 输出：$RESULTS_DIR/$SCENE_NAME/colmap_scene/{images, sparse/0/*.txt}
#
# Env (all optional except RAW_DIR):
#   SCENE_NAME=double_wave   # 纯 ASCII，做目录名/fid 对齐
#   RAW_DIR=                # 必填，采集包路径（WSL 路径，如 /mnt/d/dataset/...）
#   KEEP_FRAC=0.90          # 点云离群点过滤比例（按到相机中心距离）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE_NAME="${SCENE_NAME:-double_wave}"
RAW_DIR="${RAW_DIR:-}"
KEEP_FRAC="${KEEP_FRAC:-0.90}"
OUT_DIR="$RESULTS_DIR/$SCENE_NAME/colmap_scene"

echo "🔄 [01] 采集包 → COLMAP 场景"
echo "  📂 输入:  $RAW_DIR"
echo "  💾 输出:  $OUT_DIR"
echo "  🧹 点云保留比例: $KEEP_FRAC"
echo ""

if [ -z "$RAW_DIR" ]; then
    echo "❌ ERROR: 必须指定 RAW_DIR（采集包路径）" >&2
    exit 1
fi
if [ ! -f "$RAW_DIR/transforms.json" ] || [ ! -d "$RAW_DIR/image" ] || [ ! -f "$RAW_DIR/pcd.ply" ]; then
    echo "❌ ERROR: 采集包不完整（需要 transforms.json + image/ + pcd.ply）: $RAW_DIR" >&2
    exit 1
fi

python "$SCRIPT_DIR/01_convert_capture.py" \
    --raw_dir "$RAW_DIR" \
    --out_dir "$OUT_DIR" \
    --keep_frac "$KEEP_FRAC"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: 转换失败（投影自检没过就看脚本末尾的 ERROR 行）" >&2
    exit 1
fi

N_IMG="$(ls "$OUT_DIR/images" | wc -l)"
echo ""
echo "🎉 [01] Done. $N_IMG 张去畸变 PNG + sparse/0 三件套"
echo "  → 训练: GPU=0 SCENE_NAME=$SCENE_NAME SOURCE_PATH=$OUT_DIR bash som_human/02_train.sh"
