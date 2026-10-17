#!/usr/bin/env bash
# 01g_person_masks.sh — SAM3 多人前景 mask（video 模式，obj_id 跨帧一致）
#
# 用途：为「按人独立重建」准备逐帧逐人 mask。SAM3 video predictor 在首帧用
# 文本 prompt（默认 person）一次性检测所有实例，之后 propagate 全序列 —— 同一个
# obj_id 全程指向同一个人，这是 per-person 拆分的前提（image 模式每帧 ID 会乱序）。
#
# 复用 vggt_human/sam3_face_masks_worker.py（同一套 SAM3 链路，只是把
# SAM3_PROMPT 从 face 换成 person），不重复实现模型加载与保存逻辑。
#
# 输入：<SCENE_DIR>/images/
# 输出：<OUT_DIR>/{stem}.p{pid:02d}.mask.png   每人二值 mask（已腐蚀）
#       <OUT_DIR>/{stem}.p{pid:02d}.alpha.png  每人羽化软 mask
#       <OUT_DIR>/masks_manifest.json          逐帧实例清单（pid/score/box/coverage）
#
# Env:
#   SCENE_NAME=human_seq
#   SCENE_DIR=          # 默认 $RESULTS_DIR/$SCENE_NAME/colmap_refined
#   IMAGES_DIR=         # 默认 $SCENE_DIR/images
#   OUT_DIR=            # 默认 $SCENE_DIR/persons/masks
#   SAM3_PROMPT=person  # 文本 prompt（person / face / hand ...）
#   MASK_MODE=video     # video=跨帧同 ID（per-person 拆分必须）；image=逐帧独立
#   VGGT_HUMAN_DIR=     # worker 所在仓（默认仓库内 ../vggt_human）
#   SAM3_PYTHON=        # 默认 ~/miniconda3/envs/sam3/bin/python
#   SAM3_CKPT=          # 默认 /mnt/d/wheel/vggt_human_ms/sam3/sam3.pt
#   FORCE=0             # 1=已有 manifest 也重跑
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE_NAME="${SCENE_NAME:-human_seq}"
SCENE_DIR="${SCENE_DIR:-$RESULTS_DIR/$SCENE_NAME/colmap_refined}"
IMAGES_DIR="${IMAGES_DIR:-$SCENE_DIR/images}"
OUT_DIR="${OUT_DIR:-$SCENE_DIR/persons/masks}"
SAM3_PROMPT="${SAM3_PROMPT:-person}"
MASK_MODE="${MASK_MODE:-video}"
VGGT_HUMAN_DIR="${VGGT_HUMAN_DIR:-$SCRIPT_DIR/../vggt_human}"
SAM3_PYTHON="${SAM3_PYTHON:-$HOME/miniconda3/envs/sam3/bin/python}"
SAM3_CKPT="${SAM3_CKPT:-/mnt/d/wheel/vggt_human_ms/sam3/sam3.pt}"
SAM3_BPE="${SAM3_BPE:-$HOME/repos/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"
FORCE="${FORCE:-0}"

WORKER="$VGGT_HUMAN_DIR/sam3_face_masks_worker.py"

echo "🚀 [01g] SAM3 多人 mask（prompt='$SAM3_PROMPT', mode=$MASK_MODE）"
echo "  🖼️ 图像:   $IMAGES_DIR"
echo "  💾 输出:   $OUT_DIR"
echo "  🔧 worker: $WORKER"
echo "  🏋️ ckpt:   $SAM3_CKPT"
echo ""

# ── 前置检查 ───────────────────────────────────────────────────────────────
[ -d "$IMAGES_DIR" ] || { echo "❌ ERROR: 图像目录不存在: $IMAGES_DIR" >&2; exit 1; }
[ -f "$WORKER" ]     || { echo "❌ ERROR: 找不到 worker: $WORKER（用 VGGT_HUMAN_DIR 指定）" >&2; exit 1; }
[ -x "$SAM3_PYTHON" ] || { echo "❌ ERROR: sam3 env python 不存在: $SAM3_PYTHON" >&2; exit 1; }
[ -f "$SAM3_CKPT" ]  || { echo "❌ ERROR: SAM3 权重不存在: $SAM3_CKPT" >&2; exit 1; }
# ⚠️ -L 必须加：01c 产出的 images 是指向 colmap_scene/images 的**目录软链接**，
# 不加 -L 时 find 不进入链接目录 → 计数恒 0（2026-10-17 实踩）。
N_IMG=$(find -L "$IMAGES_DIR" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)
echo "  📐 待处理帧数: $N_IMG"
[ "$N_IMG" -gt 0 ] || { echo "❌ ERROR: 目录里没有图片" >&2; exit 1; }

if [ -f "$OUT_DIR/masks_manifest.json" ] && [ "$FORCE" != "1" ]; then
    echo "⏭️ 已存在 manifest（FORCE=1 可重跑）: $OUT_DIR/masks_manifest.json"
    exit 0
fi
mkdir -p "$OUT_DIR"

# ── 调用 SAM3 worker（它自己处理视频 session / 逐帧保存）─────────────────────
SAM3_PROMPT="$SAM3_PROMPT" MASK_MODE="$MASK_MODE" \
SAM3_CKPT="$SAM3_CKPT" SAM3_BPE="$SAM3_BPE" \
"$SAM3_PYTHON" "$WORKER" --images_dir "$IMAGES_DIR" --output_dir "$OUT_DIR"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: SAM3 worker 退出非 0（看上面的报错）" >&2
    exit 1
fi

# ── 汇总：每个 pid 的帧数与覆盖率 ────────────────────────────────────────────
"$SAM3_PYTHON" - "$OUT_DIR" <<'PY'
import json, os, sys
from collections import defaultdict

out_dir = sys.argv[1]
mf_path = os.path.join(out_dir, "masks_manifest.json")
with open(mf_path) as f:
    mf = json.load(f)
frames = mf.get("frames", {})
per_pid = defaultdict(lambda: {"frames": 0, "px": 0.0, "boxes": []})
for stem, insts in frames.items():
    for it in insts:
        d = per_pid[it["pid"]]
        d["frames"] += 1
        d["px"] += it.get("coverage", 0.0)
        d["boxes"].append(it.get("box_xywh"))
print("\n📊 每人统计（pid | 出现帧数 | 平均占画面%）")
for pid in sorted(per_pid):
    d = per_pid[pid]
    n = d["frames"]
    print("  p{:02d} | {:4d} 帧 | {:5.2f}%".format(
        pid, n, d["px"] / max(n, 1)))
print("\n  总帧数 {}，至少检测到 1 人的帧数 {}".format(
    len(frames), sum(1 for s in frames.values() if s)))
PY

echo ""
echo "🎉 [01g] Done."
echo "  📄 manifest: $OUT_DIR/masks_manifest.json"
echo "  → 下一步: python som_human/01h_build_person_scenes.py --scene $SCENE_DIR"
