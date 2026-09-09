#!/usr/bin/env bash
# 01c_gen_person_masks.sh — 用 SAM3 生成 person mask（阶段二的输入）。
#
# 复用 vggt_human 的 sam2_face_masks.py 派发链（MASK_BACKEND=sam3 时切到
# sam3 conda env 跑 sam3_face_masks_worker.py），prompt 换成 "person"。
# 输出命名 {stem}.p{pid:02d}.mask.png / .alpha.png，与 flame_human 阶段二
# （match_faces.py 的 load_person_masks）约定一致。
#
# MASK_MODE:
#   video — SAM3 video predictor，obj_id 跨帧一致（多人场景推荐，慢）
#   image — 逐帧 image 模型（快 ~0.3-0.7s/img，但 pid 跟检测分数排序，
#           帧间可能跳号；flame_human 阶段二逐帧做匈牙利+时序检查，
#           pid 不跨帧一致也能用，但统计会碎）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

# 输入图像（默认上游 COLMAP source 的 images）
IMAGES_DIR="${IMAGES_DIR:-$SOURCE_DIR/images}"
# 输出 mask 目录（默认本链路自己的，不污染上游）
OUT_DIR="${PERSON_MASKS_DIR:-$RESULTS_DIR/01c_sam3_person_masks}"
MASK_MODE="${MASK_MODE:-video}"
SAM3_PROMPT="${SAM3_PROMPT:-person}"
MIN_SCORE="${MIN_SCORE:-0.5}"
# sam3 worker 的解释器与权重（vggt_human 已配好的一套）
SAM3_PYTHON="${SAM3_PYTHON:-$HOME/miniconda3/envs/sam3/bin/python}"
SAM3_CKPT="${SAM3_CKPT:-/mnt/d/wheel/vggt_human_ms/sam3/sam3.pt}"
SAM3_BPE="${SAM3_BPE:-$HOME/repos/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"
# worker 脚本在 vggt_human 目录（跨算法复用，不复制）
VGGT_HUMAN_DIR="${VGGT_HUMAN_DIR:-$SCRIPT_DIR/../vggt_human}"

echo "🚀 [flame_human 01c] SAM3 person mask 生成"
echo "  🖼️ images : $IMAGES_DIR"
echo "  💾 output : $OUT_DIR"
echo "  🎭 mode   : $MASK_MODE  prompt='$SAM3_PROMPT'"

if [ ! -d "$IMAGES_DIR" ] || [ -z "$(ls "$IMAGES_DIR" 2>/dev/null)" ]; then
    echo "❌ IMAGES_DIR 不存在或为空: $IMAGES_DIR" >&2
    echo "   → 显式传 IMAGES_DIR=... 指向上游图像目录" >&2
    exit 1
fi
if [ ! -f "$SAM3_CKPT" ]; then
    echo "❌ SAM3 权重缺失: $SAM3_CKPT" >&2
    echo "   → 先跑 vggt_human/00a_setup_env.sh 下载 sam3" >&2
    exit 1
fi
if [ ! -x "$SAM3_PYTHON" ]; then
    echo "❌ sam3 env python 不存在: $SAM3_PYTHON" >&2
    exit 1
fi
if [ ! -f "$VGGT_HUMAN_DIR/sam2_face_masks.py" ]; then
    echo "❌ 找不到 vggt_human 的 mask worker: $VGGT_HUMAN_DIR/sam2_face_masks.py" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

# 已有足量 mask 就跳过（重跑不重复花钱）
n_masks=$(ls "$OUT_DIR"/*.mask.png 2>/dev/null | wc -l)
n_imgs=$(ls "$IMAGES_DIR" | grep -c -i -E "\.(jpg|jpeg|png)$")
if [ "$n_masks" -ge "$n_imgs" ] && [ "$n_masks" -gt 0 ]; then
    echo "⏭️ 已有 $n_masks 张 mask ≥ $n_imgs 图像，跳过"
    exit 0
fi

# 注意：worker 由 vggt_human 的派发脚本启动，但实际跑在 sam3 env。
# 派发脚本本身只做 subprocess 转发，对当前 env 无依赖，用当前 python 即可。
MASK_BACKEND=sam3 MASK_MODE="$MASK_MODE" \
SAM3_PYTHON="$SAM3_PYTHON" SAM3_CKPT="$SAM3_CKPT" SAM3_BPE="$SAM3_BPE" \
SAM3_PROMPT="$SAM3_PROMPT" MIN_SCORE="$MIN_SCORE" \
python "$VGGT_HUMAN_DIR/sam2_face_masks.py" \
    --images_dir "$IMAGES_DIR" \
    --output_dir "$OUT_DIR"
if [ $? -ne 0 ]; then
    echo "❌ SAM3 person mask 生成失败" >&2
    exit 1
fi

n_masks=$(ls "$OUT_DIR"/*.mask.png 2>/dev/null | wc -l)
echo "✅ 共 $n_masks 张 mask → $OUT_DIR"
echo "🎉 Done."
