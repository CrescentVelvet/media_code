#!/usr/bin/env bash
# run_batch.sh — 批量跑采集包目录：01b(+01) → 01d → 02 → 03 → eval_compare。
#
# 输入目录约定：$BATCH_ROOT/<id>-<场景名>/，每个子目录是一个手机采集包
# （image/*.heic + 可选 transforms.json/pcd.ply，见 01b 头注释）。
# 场景名取「第一个 - 之后的部分」作 SCENE_NAME（中文名在 WSL ext4 下没问题，
# 但输出路径也会带中文；介意的调用方先改名）。
#
# 幂等：每个步骤检测产物，已存在即跳过——中断后重跑同一命令即可续跑。
# 单场景失败不阻塞后续场景，结尾打印汇总表。
#
# 用法：
#   bash deformable_human/run_batch.sh                 # 跑全部剩余场景
#   SKIP_SCENES="双人招手" bash deformable_human/run_batch.sh
#
# Env (all optional, defaults shown):
#   BATCH_ROOT=/mnt/d/dataset/手部微动样例
#   RESULTS_DIR=              # 默认走 _env.sh
#   SKIP_SCENES=""            # 空格分隔的场景名（比对的是 - 之后部分）
#   ONLY_SCENES=""            # 非空则只跑这些（调试用）
#   ITERATIONS=10000  WARM_UP=1000
#   EXTRA_TRAIN_ARGS="-r 4 --densify_grad_threshold 0.0008 --densify_until_iter 2000"
#                         # 室外复杂背景教训（wave2p 首训 24GB 钉死）：
#                         # 默认按最保守档，室内场景亏了分辨率但保证全跑完
#   GPU=0
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

BATCH_ROOT="${BATCH_ROOT:-/mnt/d/dataset/手部微动样例}"
BATCH_ROOT="${BATCH_ROOT%/}"
if [ ! -d "$BATCH_ROOT" ]; then
    echo "❌ ERROR: BATCH_ROOT 不存在: $BATCH_ROOT" >&2
    exit 1
fi
SKIP_SCENES="${SKIP_SCENES:-}"
ONLY_SCENES="${ONLY_SCENES:-}"
ITERATIONS="${ITERATIONS:-10000}"
WARM_UP="${WARM_UP:-1000}"
GPU="${GPU:-0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:--r 4 --densify_grad_threshold 0.0008 --densify_until_iter 2000}"

echo "🚀 [batch] 采集包批量处理"
echo "  📂 BATCH_ROOT: $BATCH_ROOT"
echo "  💾 RESULTS_DIR: $RESULTS_DIR"
echo "  🏋️ 配方: iter=$ITERATIONS warm=$WARM_UP mask=on | $EXTRA_TRAIN_ARGS"
echo "  ⏭️ 跳过: ${SKIP_SCENES:-（无）}  只跑: ${ONLY_SCENES:-（全部）}"
echo ""

declare -a REPORT=()
for dir in "$BATCH_ROOT"/*/; do
    base="$(basename "$dir")"
    scene="${base#*-}"
    [[ " $SKIP_SCENES " == *" $scene "* ]] && { echo "⏭️  [$scene] 在 SKIP 名单"; continue; }
    if [ -n "$ONLY_SCENES" ] && [[ " $ONLY_SCENES " != *" $scene "* ]]; then
        continue
    fi
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "▶️  [$scene]  $dir"
    echo "════════════════════════════════════════════════════════════"
    SCENE_ROOT="$RESULTS_DIR/$scene"
    LOG="$SCENE_ROOT/batch.log"
    mkdir -p "$SCENE_ROOT"
    status="✅"

    # 持有物场景补充 prompt（merge 要求与 person 重叠，安全）
    extra=""
    case "$scene" in
        *抱枕*) extra="plush toy" ;;
    esac

    {
        # ── step 1: 采集包 → 预置帧 + COLMAP ──
        if [ -d "$SCENE_ROOT/colmap_scene/sparse/0" ]; then
            echo "⏭️  [$scene] COLMAP 已存在，跳过 01b+01"
        else
            echo "▶️  [$scene] 01b 转码 + 01 COLMAP"
            CAPTURE_DIR="${dir%/}" SCENE_NAME="$scene" RESULTS_DIR="$RESULTS_DIR" \
                GPU="$GPU" RUN_PIPELINE=1 \
                bash "$SCRIPT_DIR/01b_capture_to_frames.sh" || exit 1
        fi

        # ── step 2: SAM3 前景 mask ──
        n_masks=$(find "$SCENE_ROOT/colmap_scene/masks" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
        n_imgs=$(find "$SCENE_ROOT/colmap_scene/images" -maxdepth 1 -type f 2>/dev/null | wc -l)
        if [ "$n_masks" -gt 0 ] && [ "$n_masks" -ge "$n_imgs" ]; then
            echo "⏭️  [$scene] masks 已存在（$n_masks），跳过 01d"
        else
            echo "▶️  [$scene] 01d SAM3 mask（extra='$extra'）"
            SCENE_NAME="$scene" RESULTS_DIR="$RESULTS_DIR" EXTRA_PROMPTS="$extra" \
                bash "$SCRIPT_DIR/01d_fg_masks.sh" || exit 1
        fi

        # ── step 3: 训练 ──
        if ls "$SCENE_ROOT"/model/point_cloud/iteration_*/point_cloud.ply >/dev/null 2>&1; then
            echo "⏭️  [$scene] 模型已存在，跳过 02"
        else
            echo "▶️  [$scene] 02 训练"
            GPU="$GPU" SCENE_NAME="$scene" RESULTS_DIR="$RESULTS_DIR" \
                ITERATIONS="$ITERATIONS" WARM_UP="$WARM_UP" \
                USE_MASK_LOSS=1 FG_WEIGHT=1.0 BG_WEIGHT=0.2 \
                EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS" \
                bash "$SCRIPT_DIR/02_train.sh" || exit 1
        fi

        # ── step 4: 渲染 + 指标 ──
        if [ -f "$SCENE_ROOT/model/test/results.json" ]; then
            echo "⏭️  [$scene] 渲染已存在，跳过 03"
        else
            echo "▶️  [$scene] 03 渲染+指标"
            GPU="$GPU" SCENE_NAME="$scene" RESULTS_DIR="$RESULTS_DIR" \
                MODE=render RUN_METRICS=1 \
                bash "$SCRIPT_DIR/03_render.sh" || exit 1
        fi

        # ── step 5: 对比图 ──
        if [ -d "/mnt/d/output/${scene}_compare" ] && [ "$(ls "/mnt/d/output/${scene}_compare" 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "⏭️  [$scene] 对比图已存在，跳过"
        else
            echo "▶️  [$scene] 对比图"
            SCENE_NAME="$scene" RESULTS_DIR="$RESULTS_DIR" \
                python "$SCRIPT_DIR/eval_compare.py" || exit 1
        fi
    } 2>&1 | tee "$LOG" || status="❌"

    # 从日志里抓汇总指标（失败场景为空）
    full="$(grep -oP '全图 PSNR : \K[0-9.]+' "$LOG" | tail -1)"
    fg="$(grep -oP '前景 PSNR: \K[0-9.]+' "$LOG" | tail -1)"
    REPORT+=("$(printf '%-12s %s  full=%s fg=%s' "$scene" "$status" "${full:--}" "${fg:--}")")
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📊 批量汇总"
echo "════════════════════════════════════════════════════════════"
for line in "${REPORT[@]}"; do echo "  $line"; done
echo ""
echo "🎉 BATCH ALL DONE（❌ 场景看各自 \$RESULTS_DIR/<scene>/batch.log）"
