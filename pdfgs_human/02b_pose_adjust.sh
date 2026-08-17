#!/usr/bin/env bash
# 02b_pose_adjust.sh — 对 COLMAP 场景做 PoseAdjuster 一次性位姿调整.
#
# 在 step 02 (Pi3 → COLMAP) 之后、step 03 (PDF-GS 训练) 之前运行。
# 读取 source/sparse/0/{cameras,images,points3D}.txt → 视线交点居中 +
# SVD 重力对齐 + 尺度归一化 → 写 source_adjusted/（相机外参+点云变换，内参不变）。
#
# 无需改 PDF-GS train.py — 03 直接用 source_adjusted/ 训练。
#
# A/B 对比用法:
#   # A: 无位姿优化（标准流程）
#   GPU=0 OUTPUT_NAME=orbit bash pdfgs_human/02_pi3_colmap.sh
#   GPU=0 OUTPUT_NAME=orbit bash pdfgs_human/03_train_pdfgs.sh
#   # → model at $RESULTS_DIR/orbit/model_pdfgs/
#
#   # B: 有位姿优化
#   GPU=0 OUTPUT_NAME=orbit_pose bash pdfgs_human/02_pi3_colmap.sh
#   GPU=0 OUTPUT_NAME=orbit_pose bash pdfgs_human/02b_pose_adjust.sh
#   GPU=0 SOURCE_DIR=$RESULTS_DIR/orbit_pose/pi3/source_adjusted OUTPUT_NAME=orbit_pose bash pdfgs_human/03_train_pdfgs.sh
#   # → model at $RESULTS_DIR/orbit_pose/model_pdfgs/
#
# Prerequisites: step 02 (COLMAP 场景已生成).
#
# Env (all optional, defaults shown):
#   OUTPUT_NAME=orbit          # 基名 (须与 step 02 一致)
#   RESULTS_DIR=                # 输出根
#   SOURCE_DIR=                 # 输入 COLMAP (默认: $RESULTS_DIR/$OUTPUT_NAME/pi3/source)
#   SOURCE_ADJUSTED_DIR=        # 输出 (默认: $SOURCE_DIR/../source_adjusted)
#   GRAVITY_PRIOR=0             # 0=SVD 估计重力, 1=用 [0,-1,0]
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

OUTPUT_NAME="${OUTPUT_NAME:-orbit}"
SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/$OUTPUT_NAME/pi3/source}"
SOURCE_ADJUSTED_DIR="${SOURCE_ADJUSTED_DIR:-$RESULTS_DIR/$OUTPUT_NAME/pi3/source_adjusted}"
GRAVITY_PRIOR="${GRAVITY_PRIOR:-0}"

echo "🏋️ [02b] PoseAdjuster: COLMAP 场景位姿调整"
echo "  📂 input:         $SOURCE_DIR"
echo "  💾 output:        $SOURCE_ADJUSTED_DIR"
echo "  📐 gravity_prior: $GRAVITY_PRIOR"
echo ""

# Sanity checks
if [ ! -d "$SOURCE_DIR/sparse/0" ]; then
    echo "❌ ERROR: COLMAP scene not found: $SOURCE_DIR/sparse/0" >&2
    echo "       Run step 02 first: OUTPUT_NAME=$OUTPUT_NAME bash $SCRIPT_DIR/02_pi3_colmap.sh" >&2
    exit 1
fi
if [ ! -f "$SOURCE_DIR/sparse/0/cameras.txt" ] || [ ! -f "$SOURCE_DIR/sparse/0/images.txt" ]; then
    echo "❌ ERROR: cameras.txt or images.txt not found in $SOURCE_DIR/sparse/0" >&2
    exit 1
fi

# Check that vggt_human/pose_adjuster.py exists (we import from there)
VGGT_HUMAN_DIR="$REPO_DIR/vggt_human"
if [ ! -f "$VGGT_HUMAN_DIR/pose_adjuster.py" ]; then
    echo "❌ ERROR: pose_adjuster.py not found at $VGGT_HUMAN_DIR/pose_adjuster.py" >&2
    echo "       vggt_human/ must exist alongside pdfgs_human/ in media_code/" >&2
    exit 1
fi

export SOURCE_DIR SOURCE_ADJUSTED_DIR GRAVITY_PRIOR
python "$SCRIPT_DIR/pose_adjust_colmap.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi

echo ""
echo "✅ [02b] Done. Adjusted COLMAP: $SOURCE_ADJUSTED_DIR"
echo "  Next: SOURCE_DIR=$SOURCE_ADJUSTED_DIR OUTPUT_NAME=$OUTPUT_NAME bash $SCRIPT_DIR/03_train_pdfgs.sh"
