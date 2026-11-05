#!/usr/bin/env bash
# 02_run_da3_inference.sh — 对棚拍多视角数据跑 DA3：每视角深度 + 位姿 + 内参 + 置信度
# 输出: $RESULTS_DIR/da3_inference/<SCENE_NAME>/（mini_npz + colmap 稀疏模型 + 深度可视化）
# 用法: bash da3_human/02_run_da3_inference.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

export INPUT_DIR="${INPUT_DIR:-$HOME/output/vggt_human_results/source/images}"
export SCENE_NAME="${SCENE_NAME:-studio_human}"
export DA3_MODEL_REPO="${DA3_MODEL_REPO:-depth-anything/DA3NESTED-GIANT-LARGE-1.1}"
export PROCESS_RES="${PROCESS_RES:-504}"
export MAX_VIEWS="${MAX_VIEWS:-0}"            # 0 = 全部视角（显存不足时调小，如 60）
export EXPORT_FORMAT="${EXPORT_FORMAT:-mini_npz-colmap-depth_vis}"
export COLMAP_REF_DIR="${COLMAP_REF_DIR:-}"   # 可选：COLMAP sparse/0 参考，做位姿对齐校验
export OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/da3_inference/$SCENE_NAME}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ ERROR: 输入图像目录不存在: $INPUT_DIR" >&2
    exit 1
fi

echo "🚀 [02] DA3 推理: $SCENE_NAME"
echo "  🖼️ 输入: $INPUT_DIR (max_views=$MAX_VIEWS, process_res=$PROCESS_RES)"
echo "  🤖 模型: $DA3_MODEL_REPO"
echo "  💾 输出: $OUTPUT_DIR"
echo "  🎮 GPU: ${GPU:-all}"

# HF 直连不通时回退 hf-mirror（from_pretrained 也要走网络）
if [ -z "${HF_ENDPOINT:-}" ]; then
    if ! curl -sI --max-time 8 -o /dev/null "https://huggingface.co/api/models/$DA3_MODEL_REPO"; then
        echo "⚠️ huggingface.co API 直连失败，回退 hf-mirror.com"
        export HF_ENDPOINT="https://hf-mirror.com"
    fi
fi

mkdir -p "$OUTPUT_DIR"
python "$SCRIPT_DIR/run_da3_inference.py"
if [ $? -ne 0 ]; then echo "❌ FAILED" >&2; exit 1; fi
echo "🎉 Done. 深度/位姿产物在 $OUTPUT_DIR"
