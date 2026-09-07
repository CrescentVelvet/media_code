#!/usr/bin/env bash
# 03f_fit_head_3dmm.sh — 多视角 3DMM 头拟合（每人一个世界坐标人头网格）
#
# 头/身/景拆分 step 3：把 3DMM 模板拟合到多视角 landmark 上。
# 拟合变量 = 全局尺度 s + 身份系数 β + 每帧刚体 (R,t)；
# 观测 = 03e 检出的 468 landmark 2D 重投影。
#
# 输入: $MODEL_3DMM_DIR/3dmm_template.npz        (03d 产出)
#       $RESULTS_DIR/03e_head_3dmm/face_landmarks.json  (03e 产出)
#       $RESULTS_DIR/03_source/                  (COLMAP 相机)
# 输出: $RESULTS_DIR/03e_head_3dmm/head_fit.json
#       $RESULTS_DIR/03e_head_3dmm/head_mesh_p<id>.npz   (含后脑的完整头)
#
# Env (all optional, defaults shown):
#   MODEL_3DMM_DIR=   # 3DMM 模板根（默认 $HOME/model/ICT-FaceKit）
#   RESULTS_DIR=      # 输出根（默认 $HOME/output/vggt_human_results）
#   SOURCE_DIR=       # COLMAP 场景（默认 $RESULTS_DIR/03_source）
#   MIN_FRAMES=8      # 少于该观测帧数的人跳过
#   FIT_BETA=1        # 1 = 拟合身份系数（需要 03d 下过 identity 网格）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MODEL_3DMM_DIR="${MODEL_3DMM_DIR:-$HOME/model/ICT-FaceKit}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/output/vggt_human_results}"
SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/03_source}"
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/03e_head_3dmm}"
MIN_FRAMES="${MIN_FRAMES:-8}"
FIT_BETA="${FIT_BETA:-1}"

echo "🗿 [03f] 多视角 3DMM 头拟合"
echo "  📦 模板:   $MODEL_3DMM_DIR/3dmm_template.npz"
echo "  📂 场景:   $SOURCE_DIR"
echo "  💾 输出:   $OUT_DIR"
echo "  ⚙️ min_frames=$MIN_FRAMES fit_beta=$FIT_BETA"
echo ""

if [ ! -f "$MODEL_3DMM_DIR/3dmm_template.npz" ]; then
    echo "❌ 缺少 3DMM 模板，先跑: bash 03d_prepare_3dmm.sh" >&2
    exit 1
fi
if [ ! -f "$OUT_DIR/face_landmarks.json" ]; then
    echo "❌ 缺少 landmark，先跑: bash 03e_detect_landmarks.sh" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

MODEL_3DMM_DIR="$MODEL_3DMM_DIR" \
RESULTS_DIR="$RESULTS_DIR" \
SOURCE_DIR="$SOURCE_DIR" \
OUT_DIR="$OUT_DIR" \
MIN_FRAMES="$MIN_FRAMES" \
FIT_BETA="$FIT_BETA" \
python "$SCRIPT_DIR/fit_head_3dmm.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi

echo ""
echo "✅ [03f] 头拟合完成"
echo "  📄 $OUT_DIR/head_fit.json"
echo "  🧱 $OUT_DIR/head_mesh_p*.npz"
