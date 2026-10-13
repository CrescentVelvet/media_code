#!/usr/bin/env bash
# 01b_capture_to_frames.sh — 手机采集包 → 预置帧（喂 01 的预置帧模式）。
#
# 采集包 = 华为/iPhone 导出的目录，约定布局（hand_motion、wave2p 均为此格式）：
#   <CAPTURE_DIR>/
#     image/*.heic        # 帧序列（文件名任意，字典序=时序）
#     transforms.json     # （可选）NeRF 格式位姿+内参，01e 降级路线的输入
#     pcd.ply             # （可选）稠密点云，01e 的初始化来源
#
# 本脚本只做一件事：调 vggt_human/01b_heic_to_jpg.sh 把 image/*.heic 转成 JPG，
# 落到 $RESULTS_DIR/$SCENE_NAME/frames/image/——之后 01_prepare_data.sh 检测到
# 预置帧会自动跳过抽帧，直接进 COLMAP。
#
# 用法：
#   CAPTURE_DIR="/mnt/d/dataset/手部微动样例/xxx-双人招手" SCENE_NAME=wave2p \
#     RESULTS_DIR=~/output/deformable_human_results \
#     bash deformable_human/01b_capture_to_frames.sh
#
# Env (all optional, defaults shown):
#   CAPTURE_DIR=         # 采集包目录（必传）
#   SCENE_NAME=human_seq
#   RESULTS_DIR=         # 默认走 _env.sh（WSL 经 proxy.env 覆盖为 ~/output/...）
#   RUN_PIPELINE=0       # 1=转换完自动接 01（COLMAP + POSE_ADJUST）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

CAPTURE_DIR="${CAPTURE_DIR:-}"
SCENE_NAME="${SCENE_NAME:-human_seq}"
SCENE_ROOT="$RESULTS_DIR/$SCENE_NAME"
RUN_PIPELINE="${RUN_PIPELINE:-0}"

echo "🚀 [01b] 采集包 → 预置帧"
echo "  📁 采集包: $CAPTURE_DIR"
echo "  💾 输出:   $SCENE_ROOT/frames/image/"
echo ""

if [ -z "$CAPTURE_DIR" ] || [ ! -d "$CAPTURE_DIR/image" ]; then
    echo "❌ ERROR: CAPTURE_DIR 未设置或缺少 image/ 子目录: ${CAPTURE_DIR:-（空）}" >&2
    exit 1
fi
N_HEIC="$(find "$CAPTURE_DIR/image" -maxdepth 1 -iname '*.heic' | wc -l)"
if [ "$N_HEIC" -eq 0 ]; then
    echo "❌ ERROR: $CAPTURE_DIR/image/ 下没有 .heic 文件" >&2
    exit 1
fi
echo "  🖼️  发现 $N_HEIC 张 HEIC"
[ -f "$CAPTURE_DIR/transforms.json" ] && \
    echo "  💡 附带 transforms.json + pcd.ply → COLMAP 失败时可走 01e 降级位姿路线"

# ── HEIC → JPG（委托 vggt_human/01b，文件做接口）──────────────────────────
echo "🎬 转换: 调 vggt_human/01b_heic_to_jpg.sh"
INPUT_DIR="$CAPTURE_DIR/image" \
OUTPUT_DIR="$SCENE_ROOT/frames/image" \
    bash "$VGGT_HUMAN_DIR/01b_heic_to_jpg.sh"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: HEIC 转换失败" >&2
    exit 1
fi

N_JPG="$(find "$SCENE_ROOT/frames/image" -maxdepth 1 -iname '*.jpg' | wc -l)"
echo "✅ 预置帧就绪: $N_JPG/$N_HEIC 张 → $SCENE_ROOT/frames/image/"
if [ "$N_JPG" -ne "$N_HEIC" ]; then
    echo "⚠️  WARNING: 转换数量不一致（可能有失败帧），建议检查后再进 01" >&2
fi

# ── 可选：直接接 01（预置帧模式自动生效）───────────────────────────────────
if [ "$RUN_PIPELINE" = "1" ]; then
    echo ""
    echo "🗺️  RUN_PIPELINE=1，接 01_prepare_data.sh（预置帧模式）"
    SCENE_NAME="$SCENE_NAME" RESULTS_DIR="$RESULTS_DIR" \
        bash "$SCRIPT_DIR/01_prepare_data.sh" || exit 1
fi

echo ""
echo "🎉 [01b] Done."
echo "  → 下一步: SCENE_NAME=$SCENE_NAME RESULTS_DIR=$RESULTS_DIR \\"
echo "    bash deformable_human/01_prepare_data.sh   # 检测到预置帧，直接 COLMAP"
