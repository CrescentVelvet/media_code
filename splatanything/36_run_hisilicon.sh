#!/usr/bin/env bash
# 36_run_hisilicon.sh — 本机 WSL 的海思静止视角批量推理与同视角渲染。
# 用法：bash 36_run_hisilicon.sh [extract|infer|compare|all]
# 只写每个 CASE/splatanything/ 与 CASE/splatanything.ply，原始采集文件不改动。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"
STAGE="${1:-all}"
: "${ROOT:=/mnt/d/dataset/测试数据/海思端侧重建拍摄数据}"
: "${CASE_FILTER:=}"
: "${RESOLUTION:=756 756}"
: "${FRAMES:=100}"
: "${FPS:=15}"
: "${RENDER_SIZE:=704}"
# 可显式指定已有编码器；缺省复用本机 imageio-ffmpeg，不在批处理中临时装包。
: "${FFMPEG:=$HOME/miniconda3/envs/da3_human/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v9.0.2}"
: "${FORCE:=0}"
: "${SA_ONLY:=0}"
: "${SKIP_EXISTING_VIDEO:=1}"
export FFMPEG

if [ "$STAGE" != "extract" ] && [ "$STAGE" != "infer" ] && \
   [ "$STAGE" != "compare" ] && [ "$STAGE" != "all" ]; then
    echo "Usage: bash 36_run_hisilicon.sh [extract|infer|compare|all]" >&2
    exit 2
fi
if [ ! -d "$ROOT" ]; then
    echo "❌ ROOT not found: $ROOT" >&2
    exit 1
fi
mapfile -t CASE_DIRS < <(python - "$ROOT" "$CASE_FILTER" <<'PY'
import sys
from pathlib import Path
root, selected = Path(sys.argv[1]), sys.argv[2]
paths = sorted((p for p in root.iterdir() if p.is_dir() and list(p.glob("*cameras.json"))),
               key=lambda p: int(p.name.split("_")[0][4:]))
for p in paths:
    if not selected or p.name == selected:
        print(p)
PY
)
if [ "${#CASE_DIRS[@]}" -eq 0 ]; then
    echo "❌ No cases selected" >&2
    exit 1
fi
FAILED=0
for CASE in "${CASE_DIRS[@]}"; do
    echo "======== $(basename "$CASE") ========"
    OUT="$CASE/splatanything"
    mkdir -p "$OUT/logs"
    if [ "$STAGE" = "extract" ] || [ "$STAGE" = "all" ]; then
        python "$SCRIPT_DIR/36_extract_hisilicon.py" --case "$CASE" \
            > "$OUT/logs/extract.log" 2>&1 || FAILED=1
        tail -n 3 "$OUT/logs/extract.log"
        if [ ! -f "$OUT/extraction.json" ] || [ ! -f "$OUT/prior_camera.json" ]; then
            echo "❌ 抽帧未完成，跳过本场景推理" >&2
            continue
        fi
    fi
    if [ "$STAGE" = "infer" ] || [ "$STAGE" = "all" ]; then
        if [ -s "$CASE/splatanything.ply" ] && [ "$FORCE" != "1" ]; then
            echo "⏭️  PLY exists; inference skipped (FORCE=1 to rebuild)"
        elif [ -e "$OUT/inference/splatanything" ]; then
            echo "⚠️  检测到未完成输出，拒绝静默覆盖；先备份该目录再重跑" >&2
            FAILED=1
        else
            # 空目录或仅含显存采样残留时，固定输出根，避免中断重跑改变交付路径。
            mkdir -p "$OUT/inference"
            INPUT="$OUT" MODE=pose RESOLUTION="$RESOLUTION" OUT_ROOT="$OUT/inference" \
                bash "$SCRIPT_DIR/02_run_inference.sh" > "$OUT/logs/inference.log" 2>&1
            RC=$?
            tail -n 12 "$OUT/logs/inference.log"
            if [ $RC -eq 0 ] && [ -s "$OUT/inference/splatanything/gaussians.ply" ]; then
                cp "$OUT/inference/splatanything/gaussians.ply" "$CASE/splatanything.ply"
            else
                echo "❌ Inference failed: $CASE" >&2
                FAILED=1
            fi
            if [ "$FORCE" = "1" ]; then
                echo "⚠️  FORCE=1 只重跑缺失输出；已有 PLY 需先备份移除" >&2
            fi
        fi
    fi
    if [ "$STAGE" = "compare" ] || [ "$STAGE" = "all" ]; then
        if [ ! -s "$OUT/inference/splatanything/gaussians.ply" ]; then
            echo "❌ 无完整 SA PLY，跳过渲染" >&2
            FAILED=1
            continue
        fi
        if [ "$SKIP_EXISTING_VIDEO" = "1" ] && [ -s "$OUT/comparison.json" ] && \
           python - "$OUT/comparison.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    report = json.load(f)
raise SystemExit(0 if report.get("protocol_version") == 2 else 1)
PY
        then
            if [ "$SA_ONLY" = "1" ] || [ -s "$OUT/comparison.mp4" ]; then
                echo "⏭️  视频已完成；如需重渲，SKIP_EXISTING_VIDEO=0"
                continue
            fi
        fi
        # 只在渲染阶段配置 nvcc/glm/编译缓存，不重装推理环境。
        # shellcheck disable=SC1091
        source "$SCRIPT_DIR/_gsplat_env.sh"
        ARGS=(--case "$CASE" --frames "$FRAMES" --fps "$FPS" --size "$RENDER_SIZE")
        [ "$SA_ONLY" = "1" ] && ARGS+=(--sa-only)
        [ "$SKIP_EXISTING_VIDEO" = "1" ] && ARGS+=(--skip-existing-video)
        python "$SCRIPT_DIR/37_compare_hisilicon.py" "${ARGS[@]}" \
            > "$OUT/logs/comparison.log" 2>&1 || FAILED=1
        tail -n 8 "$OUT/logs/comparison.log"
    fi
done
if [ "$FAILED" != "0" ]; then
    echo "❌ At least one stage failed; inspect each case's splatanything/logs/" >&2
    exit 1
fi
echo "🎉 All selected cases completed: $STAGE"
