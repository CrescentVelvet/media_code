#!/usr/bin/env bash
# 07_run_capture.sh — 华为 Remy 采集包（transforms.json + image/*.heic + pcd.ply）
# 直接跑 InfiniSplat v2，带显存/耗时测量。
#
# 与 02 的区别：02 吃"目录里已有 images + cameras.npz"，本脚本从采集包一路做到指标，
# 处理三件 02 不管的事：HEIC 解码、OpenGL→OpenCV 位姿、径向畸变去畸变（裁剪同步内参）。
#
# 用法:
#   RAW_DIR="/mnt/d/dataset/手部微动样例/xxx-佳佳抱枕" OUT_NAME=capture_jiajia bash 07_run_capture.sh
#   RAW_DIR=... OUT_NAME=... N_VIEWS=16 LONG_SIDE=518 bash 07_run_capture.sh
#   RAW_DIR=... OUT_NAME=... N_VIEWS=50 SKIP_INFER=1 bash 07_run_capture.sh   # 只做转换
#
# Env (all optional except RAW_DIR):
#   RAW_DIR=          # 采集包路径（必须含 transforms.json + image/）
#   OUT_NAME=         # 输出名，默认 = RAW_DIR basename 的 ASCII 前缀
#   N_VIEWS=32        # 抽帧数 2..50（序列等间隔，含首末）
#   LONG_SIDE=518     # 推理长边
#   UNDISTORT=1       # 1=去畸变（默认），0=保留原始畸变
#   SKIP_INFER=0      # 1=只转换不推理
#   RENDER=0          # 1=转换后额外跑自洽渲染评测（03 的 rerender 模式）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${N_VIEWS:=32}"
: "${LONG_SIDE:=518}"
: "${UNDISTORT:=1}"
: "${SKIP_INFER:=0}"
: "${RENDER:=0}"

if [ -z "${RAW_DIR:-}" ]; then
    echo "❌ ERROR: 必须指定 RAW_DIR（采集包路径）" >&2
    exit 1
fi
if [ ! -f "$RAW_DIR/transforms.json" ] || [ ! -d "$RAW_DIR/image" ]; then
    echo "❌ ERROR: 采集包不完整（需要 transforms.json + image/）: $RAW_DIR" >&2
    exit 1
fi
: "${OUT_NAME:=$(basename "$RAW_DIR" | cut -c1-8)}"

CAP_DIR="$RESULTS_DIR/_captures/$OUT_NAME"
mkdir -p "$RESULTS_DIR/_captures"

echo "🎬 [07] 采集包 → InfiniSplat v2"
echo "  📂 输入: $RAW_DIR"
echo "  🖼️  抽帧: $N_VIEWS     📐 长边: $LONG_SIDE"
echo "  💾 转换: $CAP_DIR"

UARGS=()
[ "$UNDISTORT" = "1" ] || UARGS+=(--no-undistort)
python "$SCRIPT_DIR/07_capture_to_npz.py" --raw-dir "$RAW_DIR" --out-dir "$CAP_DIR" \
    --n-views "$N_VIEWS" "${UARGS[@]}"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: 转换失败" >&2
    exit 1
fi

if [ "$SKIP_INFER" = "1" ]; then
    echo "⏭️  SKIP_INFER=1，只做转换"
    exit 0
fi

# 推理 + 显存/墙钟测量（bench_run 线程内采样，避免 shell 后台采样的竞态）
echo ""
echo "🚀 [07] 推理 + 采样显存"
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
    echo "🎨 [07] 自洽渲染评测"
    RESULT="$RESULT" IMAGES_DIR="$CAP_DIR/images" MODE=rerender LONG_SIDE="$LONG_SIDE" \
        bash "$SCRIPT_DIR/03_eval_render.sh"
fi

echo "🎉 Done. 结果: $RESULT"
