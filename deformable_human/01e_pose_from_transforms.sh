#!/usr/bin/env bash
# 01e_pose_from_transforms.sh — 用手机自带位姿（transforms.json + pcd.ply）
# 构建 COLMAP 文本场景，跳过 SfM（01 的替代路径）。
#
# 用法：
#   GPU=0 SCENE_NAME=hand_motion \
#     TRANSFORMS_JSON=/mnt/d/dataset/.../transforms.json \
#     bash deformable_human/01e_pose_from_transforms.sh
#
# 输出：$SCENE_ROOT/colmap_phone/{images/, sparse/0/*.txt, qc/}
# 后续：01d（SOURCE_PATH 指到 colmap_phone 重新生成 masks）→ 02（SOURCE_PATH 同上）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE_NAME="${SCENE_NAME:-hand_motion}"
SCENE_ROOT="${SCENE_ROOT:-$RESULTS_DIR/$SCENE_NAME}"
TRANSFORMS_JSON="${TRANSFORMS_JSON:-}"
FRAMES_DIR="${FRAMES_DIR:-$SCENE_ROOT/frames/image}"
OUT_SCENE="${OUT_SCENE:-$SCENE_ROOT/colmap_phone}"

echo "🚀 [01e] 手机位姿 → COLMAP 文本场景（跳过 SfM）"
echo "  📄 transforms: $TRANSFORMS_JSON"
echo "  🖼️  frames:     $FRAMES_DIR"
echo "  💾 输出:       $OUT_SCENE"

[ -f "$TRANSFORMS_JSON" ] || { echo "❌ ERROR: TRANSFORMS_JSON 未设置或不存在" >&2; exit 1; }
[ -d "$FRAMES_DIR" ] || { echo "❌ ERROR: frames 目录不存在: $FRAMES_DIR" >&2; exit 1; }

python "$SCRIPT_DIR/nerf_to_colmap.py" \
    --transforms "$TRANSFORMS_JSON" \
    --frames_dir "$FRAMES_DIR" \
    --out_scene "$OUT_SCENE"
if [ $? -ne 0 ]; then
    echo "❌ [01e] FAILED" >&2
    exit 1
fi

echo ""
echo "🔍 验证"
for f in "$OUT_SCENE/images" "$OUT_SCENE/sparse/0/cameras.txt" \
         "$OUT_SCENE/sparse/0/images.txt" "$OUT_SCENE/sparse/0/points3D.txt"; do
    [ -e "$f" ] && echo "  ✅ $f" || echo "  ❌ $f MISSING" >&2
done
echo "  👀 目检投影对齐: $OUT_SCENE/qc/proj_phone_*.jpg"
echo ""
echo "🎉 [01e] Done. 下一步："
echo "  ① SOURCE_PATH=$OUT_SCENE bash deformable_human/01d_fg_masks.sh   # 重新生成 masks"
echo "  ② GPU=0 SCENE_NAME=$SCENE_NAME SOURCE_PATH=$OUT_SCENE \\"
echo "       MODEL_PATH=$SCENE_ROOT/model_phone USE_MASK_LOSS=1 bash deformable_human/02_train.sh"
