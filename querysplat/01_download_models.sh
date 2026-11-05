#!/usr/bin/env bash
# 01_download_models.sh — QuerySplat 权重 + VGGT-Omega 骨干权重
# QuerySplat 本体: inspatio/querysplat（公开，Apache 2.0）
# VGGT-Omega 骨干: facebook/VGGT-Omega（⚠️ gated，需 HF 申请授权，自动审核）
# 两者都落到官方仓 checkpoints/（yaml 的 vggt_checkpoint 是相对路径，别挪位置）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

QS_FILE="querysplat_vggto_1B_512_8192.safetensors"
VGGT_FILE="vggt_omega_1b_512.pt"
mkdir -p "$QS_CHECKPOINT_DIR"

# HF 直连不通时回退 aifasthub（2026-11 实测 hf-mirror 已废）
if [ -z "${HF_ENDPOINT:-}" ]; then
    if ! curl -sI --max-time 8 -o /dev/null "https://huggingface.co/api/models/inspatio/querysplat"; then
        echo "⚠️ huggingface.co API 直连失败，回退 aifasthub.com 镜像"
        export HF_ENDPOINT="https://aifasthub.com"
    fi
fi
echo "  🌐 HF_ENDPOINT: ${HF_ENDPOINT:-huggingface.co (direct)}"

echo "🚀 [01] 下载 QuerySplat 权重"
hf download inspatio/querysplat "$QS_FILE" --local-dir "$QS_CHECKPOINT_DIR"
if [ $? -ne 0 ]; then
    echo "❌ QuerySplat 权重下载失败（镜像不通时换 HF_ENDPOINT 或走迅雷，见 README_wsl.md）" >&2
    exit 1
fi
echo "✅ $QS_FILE 已就绪"

echo "🚀 [01] 检查 VGGT-Omega 骨干权重（$VGGT_FILE）"
if [ -f "$QS_CHECKPOINT_DIR/$VGGT_FILE" ]; then
    echo "✅ 已存在: $QS_CHECKPOINT_DIR/$VGGT_FILE"
else
    hf download facebook/VGGT-Omega "$VGGT_FILE" --local-dir "$QS_CHECKPOINT_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ VGGT-Omega 下载失败——该仓是 gated 模型，必须先申请授权:" >&2
        echo "   https://huggingface.co/facebook/VGGT-Omega 点 \"Request access\"（自动审核），" >&2
        echo "   并确认本机已 hf login（令牌有 read 权限）。" >&2
        exit 1
    fi
fi

echo "🔍 SHA256 校验"
( cd "$QUERYSPLAT_REPO" && grep -E "$QS_FILE|$VGGT_FILE" SHA256SUMS | sha256sum -c - )
if [ $? -ne 0 ]; then echo "⚠️ 校验失败：文件与官方 SHA256SUMS 不符，别直接用" >&2; exit 1; fi
echo "🎉 Done. 权重在 $QS_CHECKPOINT_DIR"
