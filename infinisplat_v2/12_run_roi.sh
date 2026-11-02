#!/usr/bin/env bash
# 12_run_roi.sh — 裁手/脸 ROI 提升局部像素密度 → InfiniSplat v2，带显存/耗时测量。
#
# 为什么需要它：整图提分辨率低效（Step 2 实测 2.27x 预算只买 1.2x 锐度），因为反投影
# 是"每像素一个高斯"、高斯半径由 pixel footprint 定。把手部区域的像素密度提上去才是
# 正确杠杆 —— 本脚本从原图（1440x1920）裁 ROI，不经过整图下采样、不插值放大。
#
# 两个 07 没有的处理：ROI 框定位（SAM3 手部 bbox 或 anchor 投影）、内参三处同步修正
# （去畸变裁切 / ROI 裁切 / resize）。
#
# 用法:
#   RAW_DIR=".../01e954de...-佳佳抱枕" MASK_MANIFEST=.../hand_32v/masks_manifest.json \
#     OUT_NAME=jiajia_roi030 N_VIEWS=32 ROI_FRAC=0.30 bash 12_run_roi.sh
#
# Env:
#   RAW_DIR=          # 采集包路径（必须含 transforms.json + image/）
#   OUT_NAME=         # 输出名（默认 roi_<8位>）
#   MASK_MANIFEST=    # SAM3 masks_manifest.json（ROI_SOURCE=mask 时必填）
#   ROI_SOURCE=mask   # mask=SAM3 手部 bbox 定位；anchor=anchor_point 投影（零依赖，较粗）
#   ROI_FRAC=0.30     # 框边长占图比例（0.30 → 手部像素密度 x3.3）
#   N_VIEWS=32
#   TARGET_LONG=518   # 转换输出长边；推理 LONG_SIDE 默认跟它一致
#   OFFSET_U=0        # 框中心归一化偏移（微调用）
#   OFFSET_V=0
#   SKIP_INFER=0      # 1=只转换
#   RENDER=0          # 1=额外跑自洽渲染评测
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${N_VIEWS:=32}"
: "${ROI_SOURCE:=mask}"
: "${ROI_FRAC:=0.30}"
: "${TARGET_LONG:=518}"
: "${LONG_SIDE:=$TARGET_LONG}"
: "${OFFSET_U:=0}"
: "${OFFSET_V:=0}"
: "${SKIP_INFER:=0}"
: "${RENDER:=0}"
: "${FIXED_CENTER:=0}"   # 1=所有帧用同一框中心（消融：区分"裁切"与"框随帧漂移"）

if [ -z "${RAW_DIR:-}" ]; then
    echo "❌ ERROR: 必须指定 RAW_DIR" >&2
    exit 1
fi
if [ ! -f "$RAW_DIR/transforms.json" ] || [ ! -d "$RAW_DIR/image" ]; then
    echo "❌ ERROR: 采集包不完整（需要 transforms.json + image/）: $RAW_DIR" >&2
    exit 1
fi
if [ "$ROI_SOURCE" = "mask" ] && [ ! -f "${MASK_MANIFEST:-}" ]; then
    echo "❌ ERROR: ROI_SOURCE=mask 需要 MASK_MANIFEST（先跑 vggt_human/sam3_face_masks_worker.py）" >&2
    exit 1
fi
: "${OUT_NAME:=roi_$(basename "$RAW_DIR" | cut -c1-8)}"

CAP_DIR="$RESULTS_DIR/_captures/$OUT_NAME"
mkdir -p "$RESULTS_DIR/_captures"

echo "🔍 [12] 手部 ROI → InfiniSplat v2"
echo "  📂 输入: $RAW_DIR"
echo "  🎯 定位: $ROI_SOURCE   ROI_FRAC=$ROI_FRAC"
[ "$ROI_SOURCE" = "mask" ] && echo "  🖐️ mask: $MASK_MANIFEST"
echo "  🖼️  抽帧: $N_VIEWS     📐 长边: $LONG_SIDE"
echo "  💾 转换: $CAP_DIR"

MARGS=(--roi-source "$ROI_SOURCE" --roi-frac "$ROI_FRAC"
       --offset-u "$OFFSET_U" --offset-v "$OFFSET_V" --target-long "$TARGET_LONG")
[ "$ROI_SOURCE" = "mask" ] && MARGS+=(--mask-manifest "$MASK_MANIFEST")
[ "$FIXED_CENTER" = "1" ] && MARGS+=(--fixed-center)

python "$SCRIPT_DIR/12_make_roi_capture.py" --raw-dir "$RAW_DIR" --out-dir "$CAP_DIR" \
    --n-views "$N_VIEWS" "${MARGS[@]}"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: ROI 转换失败" >&2
    exit 1
fi

if [ "$SKIP_INFER" = "1" ]; then
    echo "⏭️  SKIP_INFER=1，只做转换"
    exit 0
fi

echo ""
echo "🚀 [12] 推理 + 采样显存"
BENCH_JSON="$RESULTS_DIR/_captures/${OUT_NAME}_bench_${N_VIEWS}v_${LONG_SIDE}.json"
IMAGES_DIR="$CAP_DIR/images" CAMERAS="$CAP_DIR/cameras.npz" \
    OUT_NAME="$OUT_NAME" LONG_SIDE="$LONG_SIDE" MAX_VIEWS="$N_VIEWS" \
    python "$SCRIPT_DIR/bench_run.py" --out "$BENCH_JSON" -- bash "$SCRIPT_DIR/02_run_inference.sh"
RC=$?
if [ $RC -ne 0 ]; then
    echo "❌ FAILED: 推理失败 (exit $RC)" >&2
    exit 1
fi

RESULT=$(ls -td "$RESULTS_DIR/${OUT_NAME}"* 2>/dev/null | head -1)
echo ""
echo "📊 测量结果"
python - "$BENCH_JSON" "$RESULT" <<'PY'
import json, sys
from pathlib import Path
b = json.loads(Path(sys.argv[1]).read_text())
d = Path(sys.argv[2])
st = json.loads((d / "result.json").read_text())["stats"]
grow = (b["peak_mb"] - b["base_mb"]) / 1024
print(f"  ⏱️  端到端 {b['sec']:.2f}s   （显存采样 {b['samples']} 次）")
print(f"  🎮 显存 峰值 {b['peak_mb']} MB / 基线 {b['base_mb']} MB → 净增 {grow:.2f} GB")
print(f"  🔵 高斯数 {st['final_count']:,}（raw {st['raw_count']:,} → merge {st['merged_count']:,}）")
print(f"  📄 scene.ply {(d/'scene.ply').stat().st_size/1e6:.1f} MB")
print(f"  📷 相机 {len(json.loads((d/'result.json').read_text())['inputs'])}")
PY

if [ "$RENDER" = "1" ]; then
    echo ""
    echo "🎨 [12] 自洽渲染评测"
    RESULT="$RESULT" IMAGES_DIR="$CAP_DIR/images" MODE=rerender LONG_SIDE="$LONG_SIDE" \
        bash "$SCRIPT_DIR/03_eval_render.sh"
fi

echo "🎉 Done. 结果: $RESULT"
