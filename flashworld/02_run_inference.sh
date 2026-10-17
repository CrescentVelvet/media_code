#!/usr/bin/env bash
# 02_run_inference.sh — FlashWorld 推理：输入 json（单图/文本 + 相机轨迹）→ 3DGS。
#
# 官方入口：cli.py --input_dir <json 目录> --output_dir <输出目录> [--ply] [--spz] [--video]
#   - 目录里每个 *.json 跑一次，产物落 <output_dir>/<json 同名>/
#   - PLY 用未压缩高斯参数（官方 README 说 CLI 渲染质量优于 web 界面）
#   - --video 会用 (N-1)*fps+1 个插值相机再渲染一遍序列，VAE 解码很吃显存，默认关
#
# 显存（官方在 A800 上实测）：默认 51 GB / --offload_t5 30 GB / 再加
#   --offload_transformer_during_vae 24 GB / 都加 --offload_vae 9 GB（但慢到 10 min）。
# 本机 3090 24 GB → 默认开前两个 offload；REALLY_LOW_VRAM=1 再开 --offload_vae。
#
# ⚠️ 3090 (sm_86) 无原生 FP8：quant.py 里 native_fp8_support 判 `capability >= (9,0)`，
#    8.6 会走 `F.linear` bf16 反量化分支 —— 能跑，但比 H100 慢、也更吃显存带宽。
#
# 用法：
#   GPU=0 INPUT_DIR=~/output/flashworld_results/02_input_json \
#         bash flashworld/02_run_inference.sh
#   GPU=0 INPUT_DIR=... VIDEO=1 PLY=1 SPZ=1 bash flashworld/02_run_inference.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

INPUT_DIR="${INPUT_DIR:-$RESULTS_DIR/02_input_json}"
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/03_gaussians}"
PLY="${PLY:-1}"
SPZ="${SPZ:-0}"
VIDEO="${VIDEO:-0}"
VIDEO_FPS="${VIDEO_FPS:-15}"
REALLY_LOW_VRAM="${REALLY_LOW_VRAM:-0}"

# ── 前置检查 ───────────────────────────────────────────────────────────────
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ ERROR: INPUT_DIR 不存在: $INPUT_DIR" >&2
    echo "   先跑: IMAGE=/path/to/img.png bash flashworld/02_make_input.sh" >&2
    exit 1
fi
if ! ls "$INPUT_DIR"/*.json >/dev/null 2>&1; then
    echo "❌ ERROR: $INPUT_DIR 里没有 json" >&2
    exit 1
fi
if [ ! -f "${FW_CKPT:-}" ]; then
    echo "❌ ERROR: FlashWorld ckpt 未找到（HF cache 里没有 model.ckpt）" >&2
    echo "   先跑: bash flashworld/01_download_models.sh" >&2
    echo "   或：迅雷下完 → bash flashworld/01b_link_local_weights.sh" >&2
    exit 1
fi
if [ -z "${WAN_SNAPSHOT:-}" ]; then
    echo "⚠️ WARNING: HF cache 里没找到 Wan2.2-TI2V-5B-Diffusers 快照" >&2
    echo "            推理会在 from_pretrained 处联网（大概率失败），先跑 01_download_models.sh" >&2
fi
if [ ! -d "$FLASHWORLD_DIR" ]; then
    echo "❌ ERROR: 官方仓不存在: $FLASHWORLD_DIR（先跑 00a_setup_env.sh）" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

ARGS=(--input_dir "$INPUT_DIR" --output_dir "$OUT_DIR" --ckpt "$FW_CKPT")
# 24 GB 卡：官方给的 24 GB 档 = offload_t5 + offload_transformer_during_vae
ARGS+=(--offload_t5 --offload_transformer_during_vae)
[ "$REALLY_LOW_VRAM" = "1" ] && ARGS+=(--offload_vae)
[ "$PLY" = "1" ]   && ARGS+=(--ply)
[ "$SPZ" = "1" ]   && ARGS+=(--spz)
[ "$VIDEO" = "1" ] && ARGS+=(--video --video_fps "$VIDEO_FPS")

echo "🚀 [02] FlashWorld 推理"
echo "  📁 输入:   $INPUT_DIR  ($(ls "$INPUT_DIR"/*.json | wc -l) 个 json)"
echo "  💾 输出:   $OUT_DIR"
echo "  🏋️ ckpt:   $FW_CKPT"
echo "  🧠 基座:   ${WAN_SNAPSHOT:-<缺>}"
echo "  🎮 GPU:    ${CUDA_VISIBLE_DEVICES:-all}  offload_t5+transformer_during_vae$([ "$REALLY_LOW_VRAM" = "1" ] && echo "+vae")"
echo "  📦 产物:   ply=$PLY spz=$SPZ video=$VIDEO"
echo ""

export HF_HOME HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$FLASHWORLD_DIR:${PYTHONPATH:-}"

cd "$FLASHWORLD_DIR" || exit 1
python cli.py "${ARGS[@]}"
rc=$?
if [ $rc -ne 0 ]; then
    echo "❌ FAILED (exit $rc)" >&2
    exit 1
fi

echo ""
echo "=== 产物 ==="
find "$OUT_DIR" -maxdepth 2 -type f \( -name "*.ply" -o -name "*.spz" -o -name "*.mp4" \) 2>/dev/null |
while read -r f; do
    printf "  %-58s %8.1f MB\n" "$f" "$(du -m "$f" | cut -f1)"
done
echo ""
echo "🎉 Done."
