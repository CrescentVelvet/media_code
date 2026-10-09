#!/usr/bin/env bash
# 01d_fg_masks.sh — 生成前景（人+持有物）软 mask，供 02 mask 加权 loss 使用。
#
# 做法：复用 vggt_human/sam2_face_masks.py 的 SAM3 后端（文件做接口，不复制代码），
#   pass 1: prompt=PERSON_PROMPT（默认 person）
#   pass 2: prompt=EXTRA_PROMPTS（默认 dinosaur plush toy；| 分隔多个，应对持有物）
#   merge : merge_fg_masks.py 合并——extra 实例必须与（膨胀后的）person mask
#           重叠 ≥OVERLAP_MIN 才收，滤掉误检的静态背景（实测雕塑被当玩偶）。
# 输出：$SOURCE_PATH/masks/{stem}.png（0-255 软 mask，羽毛边，直接当 loss 权重）
#       $SOURCE_PATH/masks/_qc_overlay.jpg（抽检叠加图，务必目视检查）
#
# Env (all optional, defaults shown):
#   SCENE_NAME=human_seq
#   SOURCE_PATH=        # 默认 $RESULTS_DIR/$SCENE_NAME/colmap_scene
#   PERSON_PROMPT=person
#   EXTRA_PROMPTS="dinosaur plush toy"   # | 分隔；空串=只要 person
#   MASK_MODE=video     # video=跨帧 obj_id 一致（推荐）；image=逐帧独立
#   OVERLAP_MIN=0.3     # extra 实例与 person（膨胀）的最低重叠比
#   DILATE_FRAC=0.03    # person 膨胀核 = min(H,W)*比例
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE_NAME="${SCENE_NAME:-human_seq}"
SOURCE_PATH="${SOURCE_PATH:-$RESULTS_DIR/$SCENE_NAME/colmap_scene}"
PERSON_PROMPT="${PERSON_PROMPT:-person}"
EXTRA_PROMPTS="${EXTRA_PROMPTS:-dinosaur plush toy}"
MASK_MODE="${MASK_MODE:-video}"
OVERLAP_MIN="${OVERLAP_MIN:-0.3}"
DILATE_FRAC="${DILATE_FRAC:-0.03}"

IMAGES_DIR="$SOURCE_PATH/images"
MASKS_PERSON="$SOURCE_PATH/masks_person"
OUT_DIR="$SOURCE_PATH/masks"

echo "🎭 [01d] 前景 mask 生成（SAM3 后端，复用 vggt_human）"
echo "  📂 images:   $IMAGES_DIR"
echo "  💾 masks:    $OUT_DIR"
echo "  🔧 person:   '$PERSON_PROMPT'  extra: '$EXTRA_PROMPTS'  mode: $MASK_MODE"
echo ""

[ -d "$IMAGES_DIR" ] || { echo "❌ ERROR: $IMAGES_DIR 不存在（先跑 01）" >&2; exit 1; }

run_sam3 () {  # $1=prompt  $2=out_dir
    MASK_BACKEND=sam3 MASK_MODE="$MASK_MODE" SAM3_PROMPT="$1" \
        python "$VGGT_HUMAN_DIR/sam2_face_masks.py" \
        --images_dir "$IMAGES_DIR" --output_dir "$2"
}

echo "=== pass 1: $PERSON_PROMPT ==="
run_sam3 "$PERSON_PROMPT" "$MASKS_PERSON" || { echo "❌ FAILED: person pass" >&2; exit 1; }

EXTRA_DIRS=()
if [ -n "$EXTRA_PROMPTS" ]; then
    IFS='|' read -ra _prompts <<< "$EXTRA_PROMPTS"
    i=0
    for p in "${_prompts[@]}"; do
        d="$SOURCE_PATH/masks_extra_$i"
        echo "=== pass 2.$i: $p ==="
        run_sam3 "$p" "$d" || { echo "❌ FAILED: extra pass '$p'" >&2; exit 1; }
        EXTRA_DIRS+=("$d")
        i=$((i+1))
    done
fi

echo "=== merge（person ∪ 邻接 extra）==="
MERGE_ARGS=(--person_dir "$MASKS_PERSON" --images_dir "$IMAGES_DIR" --out_dir "$OUT_DIR"
            --overlap_min "$OVERLAP_MIN" --dilate_frac "$DILATE_FRAC")
if [ ${#EXTRA_DIRS[@]} -gt 0 ]; then
    MERGE_ARGS+=(--extra_dirs "${EXTRA_DIRS[@]}")
fi
python "$SCRIPT_DIR/merge_fg_masks.py" "${MERGE_ARGS[@]}" || { echo "❌ FAILED: merge" >&2; exit 1; }

echo ""
echo "🎉 [01d] Done."
echo "  🎭 masks:    $OUT_DIR/<stem>.png"
echo "  🖼️ QC 叠加图: $OUT_DIR/_qc_overlay.jpg  ← 目视确认人+持有物都在、背景没误收"
echo "  → 下一步: USE_MASK_LOSS=1 bash deformable_human/02_train.sh"
