#!/usr/bin/env bash
# 01_download_models.sh — 下载 Gen3R 官方 checkpoint（HF: JaceyH919/Gen3R，37.7GB）
#
# 本机 huggingface.co 直连不通（curl 000），走 hf-mirror.com（实测可用，12 连接 ~2MB/s）。
# 服务器若可直连 HF，设 HF_ENDPOINT=https://huggingface.co 走官方端点。
#
# Env:
#   HF_ENDPOINT   镜像端点，默认 https://hf-mirror.com
#   MODEL_DIR     权重目录，默认 ~/model/gen3r
#   WORKERS       并行块数，默认 12
#   ONLY          只下路径含该子串的文件（调试用）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
WORKERS="${WORKERS:-6}"   # hf-mirror 并发 >8 会返 429
ONLY="${ONLY:-}"

echo "🚀 [01] 下载 Gen3R 权重"
echo "  🌐 endpoint: $HF_ENDPOINT"
echo "  💾 目标:     $MODEL_DIR"
echo "  🔧 workers:  $WORKERS"

mkdir -p "$MODEL_DIR"
ARGS=(--dest "$MODEL_DIR" --endpoint "$HF_ENDPOINT" --workers "$WORKERS")
[ -n "$ONLY" ] && ARGS+=(--only "$ONLY")

python "$SCRIPT_DIR/download_models.py" "${ARGS[@]}"
if [ $? -ne 0 ]; then
    echo "❌ ERROR: 权重下载失败" >&2
    exit 1
fi

echo "🎉 Done: $MODEL_DIR"
