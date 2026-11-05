#!/usr/bin/env bash
# 01_download_models.sh — QuerySplat 权重 + VGGT-Omega 骨干权重
# QuerySplat 本体: inspatio/querysplat（公开，Apache 2.0）
# VGGT-Omega 骨干: facebook/VGGT-Omega（⚠️ gated，需 HF 申请授权，自动审核）
# 消费位置：官方仓 checkpoints/（yaml 的 vggt_checkpoint 是相对路径，别挪位置）
#
# 复用顺序（峰哥约定 2026-11-05）：
#   1) D:\wheel 复用池已有 → 软链到 checkpoints/
#   2) checkpoints/ 已有实体文件而 wheel 没有 → 收编进 wheel（mv + 软链回指）
#   3) 都没有 → hf download 落盘 D:\wheel，再软链
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

QS_FILE="querysplat_vggto_1B_512_8192.safetensors"
VGGT_FILE="vggt_omega_1b_512.pt"
mkdir -p "$QS_CHECKPOINT_DIR" "$WHEEL_QUERYSPLAT_DIR" "$WHEEL_VGGT_OMEGA_DIR"

# HF 直连不通时回退 aifasthub（2026-11-05 实测 hf 直连已断、hf-mirror 证书废）
if [ -z "${HF_ENDPOINT:-}" ]; then
    if ! curl -sI --max-time 8 -o /dev/null "https://huggingface.co/api/models/inspatio/querysplat"; then
        echo "⚠️ huggingface.co API 直连失败，回退 aifasthub.com 镜像"
        export HF_ENDPOINT="https://aifasthub.com"
    fi
fi
echo "  🌐 HF_ENDPOINT: ${HF_ENDPOINT:-huggingface.co (direct)}"

# ---- 通用：从 wheel 复用池软链（或收编已下载的实体文件） ----
# link_from_wheel <文件名> <wheel 子目录>
link_from_wheel() {
    local fname="$1" wdir="$2" dest="$QS_CHECKPOINT_DIR/$fname"
    if [ -L "$dest" ]; then
        echo "✅ 已是软链: $dest -> $(readlink "$dest")"; return 0
    fi
    if [ -f "$wdir/$fname" ]; then
        if [ -f "$dest" ]; then
            # 消费侧有实体文件（如早前直接 hf download 到这里）：收编进 wheel 复用池
            local a b
            a="$(sha256sum "$wdir/$fname" | cut -d' ' -f1)"
            b="$(sha256sum "$dest" | cut -d' ' -f1)"
            if [ "$a" != "$b" ]; then
                echo "❌ $dest 与 wheel 复用池同名文件内容不同，先人工确认" >&2; return 1
            fi
            rm "$dest"
            echo "📦 收编已有下载进 wheel 复用池: $wdir/$fname"
        fi
        ln -sfn "$wdir/$fname" "$dest"
        echo "🔗 软链: $dest -> $wdir/$fname"; return 0
    fi
    if [ -f "$dest" ]; then
        # 消费侧有实体文件而 wheel 没有：收编进 wheel，软链回指（下次别的仓可复用）
        mv "$dest" "$wdir/$fname"
        ln -sfn "$wdir/$fname" "$dest"
        echo "📦 收编已有下载进 wheel 复用池: $wdir/$fname"; return 0
    fi
    return 1   # 都没有，需要下载
}

echo "🚀 [01] QuerySplat 本体权重（$QS_FILE）"
if ! link_from_wheel "$QS_FILE" "$WHEEL_QUERYSPLAT_DIR"; then
    hf download inspatio/querysplat "$QS_FILE" --local-dir "$WHEEL_QUERYSPLAT_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ QuerySplat 权重下载失败（镜像不通时换 HF_ENDPOINT 或走迅雷，见 README_wsl.md）" >&2
        exit 1
    fi
    ln -sfn "$WHEEL_QUERYSPLAT_DIR/$QS_FILE" "$QS_CHECKPOINT_DIR/$QS_FILE"
    echo "🔗 软链: $QS_CHECKPOINT_DIR/$QS_FILE"
fi

echo "🚀 [01] VGGT-Omega 骨干权重（$VGGT_FILE）"
if ! link_from_wheel "$VGGT_FILE" "$WHEEL_VGGT_OMEGA_DIR"; then
    hf download facebook/VGGT-Omega "$VGGT_FILE" --local-dir "$WHEEL_VGGT_OMEGA_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ VGGT-Omega 下载失败——该仓是 gated 模型，必须先申请授权:" >&2
        echo "   https://huggingface.co/facebook/VGGT-Omega 点 \"Request access\"（自动审核），" >&2
        echo "   并确认本机已 hf login（令牌有 read 权限）。" >&2
        exit 1
    fi
    ln -sfn "$WHEEL_VGGT_OMEGA_DIR/$VGGT_FILE" "$QS_CHECKPOINT_DIR/$VGGT_FILE"
    echo "🔗 软链: $QS_CHECKPOINT_DIR/$VGGT_FILE"
fi

echo "🔍 SHA256 校验（对照官方 QuerySplat/SHA256SUMS）"
( cd "$QUERYSPLAT_REPO" && grep -E "$QS_FILE|$VGGT_FILE" SHA256SUMS | sha256sum -c - )
if [ $? -ne 0 ]; then echo "⚠️ 校验失败：文件与官方 SHA256SUMS 不符，别直接用" >&2; exit 1; fi
echo "🎉 Done. 消费位置: $QS_CHECKPOINT_DIR（实体在 $WHEEL_DIR）"
