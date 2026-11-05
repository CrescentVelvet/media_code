#!/usr/bin/env bash
# 01_download_models.sh — VGGT-Omega 权重（⚠️ HF gated，需先申请授权 + hf login）
# 复用顺序（峰哥约定 2026-11-05）：
#   1) D:\wheel 复用池已有 → 校验 SHA256 → 软链到 MODEL_DIR
#   2) MODEL_DIR 已有 → 直接用
#   3) 都没有 → hf download 落到 D:\wheel 复用池，再软链到 MODEL_DIR
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

VARIANT="${VARIANT:-1b_512}"
case "$VARIANT" in
    1b_512) CKPT="vggt_omega_1b_512.pt"
            # 官方 SHA256（QuerySplat/SHA256SUMS 同款文件）
            CKPT_SHA256="c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934" ;;
    1b_256_text) CKPT="vggt_omega_1b_256_text.pt"
            CKPT_SHA256="" ;;  # 暂无官方校验和，跳过校验
    *) echo "❌ ERROR: VARIANT must be 1b_512 | 1b_256_text" >&2; exit 1 ;;
esac

mkdir -p "$MODEL_DIR" "$WHEEL_VGGT_OMEGA_DIR"

verify_sha256() {  # $1=文件路径；无校验和时跳过，有则必须匹配
    [ -z "$CKPT_SHA256" ] && return 0
    local actual
    actual="$(sha256sum "$1" | cut -d' ' -f1)"
    if [ "$actual" != "$CKPT_SHA256" ]; then
        echo "❌ SHA256 不符: $1" >&2
        echo "   期望 $CKPT_SHA256" >&2
        echo "   实际 $actual" >&2
        return 1
    fi
    echo "✅ SHA256 校验通过"
}

# 1) D:\wheel 复用池
if [ -f "$WHEEL_VGGT_OMEGA_DIR/$CKPT" ]; then
    echo "⏭️ D:\\wheel 复用池已有: $WHEEL_VGGT_OMEGA_DIR/$CKPT"
    verify_sha256 "$WHEEL_VGGT_OMEGA_DIR/$CKPT" || exit 1
    ln -sfn "$WHEEL_VGGT_OMEGA_DIR/$CKPT" "$MODEL_DIR/$CKPT"
    echo "🔗 软链到 $MODEL_DIR/$CKPT"
    exit 0
fi

# 2) MODEL_DIR 已有（可能是软链）
if [ -f "$MODEL_DIR/$CKPT" ]; then
    echo "⏭️ 权重已存在: $MODEL_DIR/$CKPT ($(du -h "$MODEL_DIR/$CKPT" | cut -f1))"
    exit 0
fi

echo "🚀 [01] 下载 VGGT-Omega 权重（gated）→ 落盘 D:\\wheel 复用池"
if [ -z "${HF_ENDPOINT:-}" ]; then
    if ! curl -sI --max-time 8 -o /dev/null "https://huggingface.co/api/models/facebook/VGGT-Omega"; then
        echo "⚠️ huggingface.co API 直连失败，回退 aifasthub.com 镜像"
        export HF_ENDPOINT="https://aifasthub.com"
    fi
fi

hf download facebook/VGGT-Omega "$CKPT" --local-dir "$WHEEL_VGGT_OMEGA_DIR"
if [ $? -ne 0 ]; then
    echo "❌ 下载失败。gated 模型的两个前置条件:" >&2
    echo "   1) 浏览器打开 https://huggingface.co/facebook/VGGT-Omega 点 \"Request access\"" >&2
    echo "      （自动审核，通常很快通过）" >&2
    echo "   2) 本机 hf login（access token 需 read 权限）" >&2
    exit 1
fi
verify_sha256 "$WHEEL_VGGT_OMEGA_DIR/$CKPT" || exit 1
ln -sfn "$WHEEL_VGGT_OMEGA_DIR/$CKPT" "$MODEL_DIR/$CKPT"
echo "🔗 软链到 $MODEL_DIR/$CKPT"
echo "🎉 Done."
