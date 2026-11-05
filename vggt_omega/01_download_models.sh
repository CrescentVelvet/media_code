#!/usr/bin/env bash
# 01_download_models.sh — VGGT-Omega 权重（⚠️ HF gated，需先申请授权 + hf login）
# 权重落在 ~/model/VGGT-Omega/（与 vggt_human 的 MODEL_DIR 默认约定一致，两边共用）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

VARIANT="${VARIANT:-1b_512}"
case "$VARIANT" in
    1b_512) CKPT="vggt_omega_1b_512.pt" ;;
    1b_256_text) CKPT="vggt_omega_1b_256_text.pt" ;;
    *) echo "❌ ERROR: VARIANT must be 1b_512 | 1b_256_text" >&2; exit 1 ;;
esac

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/$CKPT" ]; then
    echo "⏭️ 权重已存在: $MODEL_DIR/$CKPT ($(du -h "$MODEL_DIR/$CKPT" | cut -f1))"
    exit 0
fi

echo "🚀 [01] 下载 VGGT-Omega 权重（gated）"
echo "  🤖 facebook/VGGT-Omega ($VARIANT)"
echo "  💾 目标: $MODEL_DIR"
if [ -z "${HF_ENDPOINT:-}" ]; then
    if ! curl -sI --max-time 8 -o /dev/null "https://huggingface.co/api/models/facebook/VGGT-Omega"; then
        echo "⚠️ huggingface.co API 直连失败，回退 aifasthub.com 镜像"
        export HF_ENDPOINT="https://aifasthub.com"
    fi
fi

hf download facebook/VGGT-Omega "$CKPT" --local-dir "$MODEL_DIR"
if [ $? -ne 0 ]; then
    echo "❌ 下载失败。gated 模型的两个前置条件:" >&2
    echo "   1) 浏览器打开 https://huggingface.co/facebook/VGGT-Omega 点 \"Request access\"" >&2
    echo "      （自动审核，通常很快通过）" >&2
    echo "   2) 本机 hf login（access token 需 read 权限）" >&2
    echo "   另：QuerySplat 仓也需要这份权重，可软链到其 checkpoints/:" >&2
    echo "   ln -s $MODEL_DIR/$CKPT ~/repos/QuerySplat/checkpoints/$CKPT" >&2
    exit 1
fi
echo "🎉 Done. $MODEL_DIR/$CKPT"
