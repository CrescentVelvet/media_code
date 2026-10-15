#!/usr/bin/env bash
# 09_hand_micro_batch.sh — 批量跑手部微动样例（som_human 采集包格式：image/*.heic + transforms.json + pcd.ply）。
#
# 每个样例的流程：
#   01b heic→jpg → 均匀抽帧(≤MAX_FRAMES，防 VGGT OOM) → 02 VGGT-Omega 推理
#   → 03 npz→COLMAP → 04 3DGS 训练 → render_novel 渲染 3 张新视角 → 汇总预览图到 Windows 盘
#
# 注意：transforms.json 自带位姿本步骤不使用（vggt_human 走 VGGT-Omega 前馈位姿）。
#
# 用法（WSL 内）：
#   bash vggt_human/09_hand_micro_batch.sh
# 可选环境变量：
#   DATASET_DIR=/mnt/d/dataset/手部微动样例   # 样例根目录
#   BATCH_ROOT=~/output/hand_micro            # 每样例工作目录（Linux fs）
#   PREVIEW_DIR=/mnt/d/output/hand_micro_preview  # 预览图输出（Windows 侧看）
#   MAX_FRAMES=150        # 喂给 VGGT 的最大帧数（均匀抽帧）
#   ITERATIONS=30000      # 3DGS 训练轮数
#   ONLY=男生室内          # 只跑名字包含该子串的样例（调试用）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

DATASET_DIR="${DATASET_DIR:-/mnt/d/dataset/手部微动样例}"
BATCH_ROOT="${BATCH_ROOT:-$HOME/output/hand_micro}"
PREVIEW_DIR="${PREVIEW_DIR:-/mnt/d/output/hand_micro_preview}"
MAX_FRAMES="${MAX_FRAMES:-150}"
ITERATIONS="${ITERATIONS:-30000}"
ONLY="${ONLY:-}"

mkdir -p "$BATCH_ROOT" "$PREVIEW_DIR"

echo "🚀 [09] 手部微动样例批量重建"
echo "  📂 数据集:   $DATASET_DIR"
echo "  💾 工作目录: $BATCH_ROOT"
echo "  🖼️ 预览输出: $PREVIEW_DIR"
echo "  📐 max_frames=$MAX_FRAMES iterations=$ITERATIONS"
echo ""

SUMMARY="$BATCH_ROOT/summary.tsv"
echo -e "sample\tstatus\tstages" > "$SUMMARY"

shopt -s nullglob
for SRC in "$DATASET_DIR"/*/; do
    NAME="$(basename "$SRC")"
    if [ -n "$ONLY" ] && [[ "$NAME" != *"$ONLY"* ]]; then
        echo "⏭️ 跳过 $NAME（不含 '$ONLY'）"
        continue
    fi
    WORK="$BATCH_ROOT/$NAME"
    mkdir -p "$WORK"
    echo ""
    echo "══════════════════════════════════════════════"
    echo "🎯 样例: $NAME"
    echo "══════════════════════════════════════════════"

    STAGES=""
    FAIL=0

    # ── 1) heic → jpg ──
    echo "🖼️ [1/6] heic→jpg"
    if GPU=0 INPUT_DIR="$SRC/image" OUTPUT_DIR="$WORK/image_jpg" \
        bash "$SCRIPT_DIR/01b_heic_to_jpg.sh" > "$WORK/01b.log" 2>&1; then
        STAGES="${STAGES}01b✅"
    else
        echo "❌ 01b 失败（见 $WORK/01b.log）"; STAGES="${STAGES}01b❌"; FAIL=1
    fi

    # ── 2) 均匀抽帧到 ≤MAX_FRAMES ──
    if [ "$FAIL" -eq 0 ]; then
        echo "✂️ [2/6] 均匀抽帧 → image_sub (≤$MAX_FRAMES)"
        if python3 - "$WORK/image_jpg" "$WORK/image_sub" "$MAX_FRAMES" <<'EOF'
import shutil, sys
from pathlib import Path
src, dst, max_frames = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
if not imgs:
    sys.exit(f"❌ no images in {src}")
k = max(1, -(-len(imgs) // max_frames))   # ceil(len/max)
picked = imgs[::k]
for p in picked:
    shutil.copy2(p, dst / p.name)
print(f"✅ {len(imgs)} -> {len(picked)} frames (stride {k})")
EOF
        then STAGES="${STAGES}sub✅"; else STAGES="${STAGES}sub❌"; FAIL=1; fi
    fi

    # ── 3) VGGT-Omega 推理 ──
    if [ "$FAIL" -eq 0 ]; then
        echo "🔍 [3/6] VGGT-Omega 推理"
        if GPU=0 INPUT_DIR="$WORK/image_sub" RESULTS_DIR="$WORK" \
            bash "$SCRIPT_DIR/02_run_inference.sh" > "$WORK/02.log" 2>&1; then
            STAGES="${STAGES}02✅"
        else
            echo "❌ 02 失败（见 $WORK/02.log）"; STAGES="${STAGES}02❌"; FAIL=1
        fi
    fi

    # ── 4) npz → COLMAP ──
    if [ "$FAIL" -eq 0 ]; then
        echo "✂️ [4/6] npz→COLMAP"
        if GPU=0 RESULTS_DIR="$WORK" \
            bash "$SCRIPT_DIR/03_npz_to_colmap.sh" > "$WORK/03.log" 2>&1; then
            STAGES="${STAGES}03✅"
        else
            echo "❌ 03 失败（见 $WORK/03.log）"; STAGES="${STAGES}03❌"; FAIL=1
        fi
    fi

    # ── 5) 3DGS 训练（跳过训练视角渲染，后面单独渲新视角）──
    if [ "$FAIL" -eq 0 ]; then
        echo "🏋️ [5/6] 3DGS 训练 ($ITERATIONS iters)"
        if GPU=0 RESULTS_DIR="$WORK" ITERATIONS="$ITERATIONS" SKIP_RENDER=1 \
            bash "$SCRIPT_DIR/04_train_3dgs.sh" > "$WORK/04.log" 2>&1; then
            STAGES="${STAGES}04✅"
        else
            echo "❌ 04 失败（见 $WORK/04.log）"; STAGES="${STAGES}04❌"; FAIL=1
        fi
    fi

    # ── 6) 渲染 3 张新视角 + 汇总预览 ──
    if [ "$FAIL" -eq 0 ]; then
        echo "🎬 [6/6] 渲染 3 张新视角"
        if GPU=0 RESULTS_DIR="$WORK" GAUSSIAN_DIR="$WORK/04_model_3dgs" \
            SOURCE_DIR="$WORK/03_source" ITERATION="$ITERATIONS" \
            NUM_NOVEL_VIEWS=3 \
            python "$SCRIPT_DIR/render_novel.py" > "$WORK/render.log" 2>&1; then
            STAGES="${STAGES}render✅"
            mkdir -p "$PREVIEW_DIR/$NAME"
            cp "$WORK"/05_novel_renders/*.png "$PREVIEW_DIR/$NAME/" 2>/dev/null
            for f in "$WORK"/05_novel_alpha/*.png; do
                [ -e "$f" ] || continue
                cp "$f" "$PREVIEW_DIR/$NAME/alpha_$(basename "$f")" 2>/dev/null
            done
            echo "  ✅ 预览: $PREVIEW_DIR/$NAME/"
            ls "$PREVIEW_DIR/$NAME/" | wc -l | xargs -I{} echo "  📦 拷贝 {} 个文件"
        else
            echo "❌ render 失败（见 $WORK/render.log）"; STAGES="${STAGES}render❌"; FAIL=1
        fi
    fi

    if [ "$FAIL" -eq 0 ]; then STATUS="OK"; else STATUS="FAIL"; fi
    echo -e "${NAME}\t${STATUS}\t${STAGES}" >> "$SUMMARY"
    echo "🏁 $NAME: $STATUS  ($STAGES)"
done

echo ""
echo "══════════════════════════════════════════════"
echo "🎉 批量完成。汇总: $SUMMARY"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo "🖼️ 预览图: $PREVIEW_DIR"
