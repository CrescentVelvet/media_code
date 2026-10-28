#!/usr/bin/env bash
# 05_preprocess_sam3.sh — SAM3 文本提示抠背景 + 主体居中，为 TaoFlowForge 批量造输入
#
# 链路（复用既有资产，零新模型）：
#   ① vggt_human/sam3_face_masks_worker.py（sam3 conda env，image 模式）
#      文本 prompt 逐图分割 → <OUT_ROOT>/masks/<物体>/{stem}.p00.{alpha,mask}.png
#   ② 05_compose_preprocessed.py（同 env，只需 PIL/numpy）
#      羽化合成白底 + bbox 正方形裁剪 + 缩 1024² → <OUT_ROOT>/<物体>/<帧号>.png
#
# SAM3 找不到实例的图 → fallback 原图直出（清单里标 fallback_original）。
#
# 用法：
#   bash taoflowforge/05_preprocess_sam3.sh                 # 全部 8 物体
#   ONLY=剃须刀,耳机 bash taoflowforge/05_preprocess_sam3.sh
#   MARGIN=0.2 SIZE=1024 bash taoflowforge/05_preprocess_sam3.sh
#
# Env:
#   DATASET_DIR  默认 /mnt/d/dataset/物体三图数据
#   OUT_ROOT     默认 <DATASET_DIR>_sam3（与原始数据并列，不污染原图）
#   SAM3_PYTHON  默认 ~/miniconda3/envs/sam3/bin/python
#   SAM3_CKPT    默认 /mnt/d/wheel/vggt_human_ms/sam3/sam3.pt
#   SAM3_BPE     默认 ~/repos/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz
#   MIN_SCORE    默认 0.5
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

DATASET_DIR="${DATASET_DIR:-/mnt/d/dataset/物体三图数据}"
OUT_ROOT="${OUT_ROOT:-${DATASET_DIR}_sam3}"
ONLY="${ONLY:-}"
MARGIN="${MARGIN:-0.15}"
SIZE="${SIZE:-1024}"
MIN_SCORE="${MIN_SCORE:-0.5}"

SAM3_PYTHON="${SAM3_PYTHON:-$HOME/miniconda3/envs/sam3/bin/python}"
SAM3_CKPT="${SAM3_CKPT:-/mnt/d/wheel/vggt_human_ms/sam3/sam3.pt}"
SAM3_BPE="${SAM3_BPE:-$HOME/repos/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"
WORKER="${WORKER:-$REPO_DIR/vggt_human/sam3_face_masks_worker.py}"

# ── SAM3 文本 prompt（物体名 → 英文名词短语；SAM3 文本编码器对英文更稳）──
declare -A PROMPTS=(
    [剃须刀]="electric shaver"
    [手表]="wrist watch"
    [榨汁机]="blender"
    [电动牙刷]="electric toothbrush"
    [电饭煲]="rice cooker"
    [耳机]="earbuds"
    [路由器]="router"
    [鼠标]="computer mouse"
)

# ── 前置检查 ───────────────────────────────────────────────────────────────
[ -d "$DATASET_DIR" ] || { echo "❌ ERROR: 数据集不存在: $DATASET_DIR" >&2; exit 1; }
[ -x "$SAM3_PYTHON" ] || { echo "❌ ERROR: sam3 python 不存在: $SAM3_PYTHON" >&2; exit 1; }
[ -f "$SAM3_CKPT" ]   || { echo "❌ ERROR: SAM3 权重不存在: $SAM3_CKPT" >&2; exit 1; }
[ -f "$SAM3_BPE" ]    || { echo "❌ ERROR: SAM3 BPE 不存在: $SAM3_BPE" >&2; exit 1; }
[ -f "$WORKER" ]      || { echo "❌ ERROR: worker 不存在: $WORKER" >&2; exit 1; }

echo "🚀 [05] SAM3 预处理: $DATASET_DIR"
echo "  💾 输出根: $OUT_ROOT"
echo "  📐 MARGIN=$MARGIN SIZE=$SIZE MIN_SCORE=$MIN_SCORE"
echo ""

_preprocess_one() {
    local obj="$1" obj_dir="$2"
    local prompt="${PROMPTS[$obj]:-$obj}"
    local masks_dir="$OUT_ROOT/masks/$obj"
    local out_dir="$OUT_ROOT/$obj"
    mkdir -p "$masks_dir" "$out_dir"

    echo "──── [$obj] prompt='$prompt' ────"
    # ① SAM3 分割（image 模式；断点友好：已有 mask 的帧 worker 会重算，量小无妨）
    MASK_MODE=image SAM3_PROMPT="$prompt" SAM3_CKPT="$SAM3_CKPT" \
        SAM3_BPE="$SAM3_BPE" MIN_SCORE="$MIN_SCORE" \
        "$SAM3_PYTHON" "$WORKER" \
        --images_dir "$obj_dir" --output_dir "$masks_dir" || {
        echo "  ❌ $obj SAM3 分割失败" >&2
        return 1
    }
    # ② 合成白底 + 居中裁剪
    MARGIN="$MARGIN" SIZE="$SIZE" "$SAM3_PYTHON" \
        "$SCRIPT_DIR/05_compose_preprocessed.py" "$masks_dir" "$obj_dir" "$out_dir" || {
        echo "  ❌ $obj 合成失败" >&2
        return 1
    }
}

for obj_dir in "$DATASET_DIR"/*/; do
    obj="$(basename "$obj_dir")"
    if [ -n "$ONLY" ]; then
        case ",$ONLY," in
            *",$obj,"*) ;;
            *) echo "⏭️  $obj（不在 ONLY 白名单）"; continue ;;
        esac
    fi
    _preprocess_one "$obj" "$obj_dir" || echo "  ⚠️ $obj 预处理失败，继续下一个"
done

echo ""
echo "════════ 预处理汇总 ════════"
echo "  产物: $OUT_ROOT/<物体>/<帧号>.png（+ masks/ + _compose_report.json）"
echo "  下一步: DATASET_DIR=$OUT_ROOT VIEW_SEL=all bash taoflowforge/03_run_batch.sh"
echo "🎉 [05] Done."
