#!/usr/bin/env bash
# 01_download_models.sh — 下载 FlashWorld 推理权重（~54 GB）到 HF cache。
#
# 两坨：
#   Wan-AI/Wan2.2-TI2V-5B-Diffusers   34.2 GB  基座（vae 2.8G / transformer 20G / text_encoder 11.4G）
#   imlixinyang/FlashWorld            19.5 GB  model.ckpt（transformer + recon_decoder）
#
# ⚠️ 本机实测（2026-10-17）：huggingface.co 直连 RST；hf-mirror.com ~1.9 MB/s；
#    modelscope.cn ~2.3 MB/s → 单线程约 7 小时。要快就用迅雷下（download_urls.md），
#    下完跑 01b_link_local_weights.sh 挂进 HF cache。
#
# 用法：
#   bash flashworld/01_download_models.sh                       # 全部（后台跑更稳）
#   nohup bash flashworld/01_download_models.sh > ~/fw_dl.log 2>&1 &
#   SKIP_WAN=1 bash flashworld/01_download_models.sh            # 只补 ckpt
#   CHECK_ONLY=1 bash flashworld/01_download_models.sh          # 只校验
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [01] 下载 FlashWorld 权重"
echo "  📁 目标:   $HF_HOME"
echo "  📡 镜像:   ${HF_ENDPOINT:-https://hf-mirror.com}"
echo "  🏋️ ckpt:   ${FW_CKPT:-<未下载>}"
echo ""

mkdir -p "$HF_HOME"
# 让 huggingface_hub 走镜像；xet 关掉（镜像不代理 xet）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

python "$SCRIPT_DIR/01_download_models.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED（可重跑，已下部分会断点续传）" >&2
    exit 1
fi
echo "🎉 Done."
