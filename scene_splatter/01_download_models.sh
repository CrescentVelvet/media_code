#!/usr/bin/env bash
# 01_download_models.sh — Scene Splatter 三份权重
#
#   flash3d v2     HF einsafutdinov/flash3d → model_re10k_v2.pth（单图前馈 3DGS 初始化）
#   viewcrafter    HF Drexubery/ViewCrafter_25 → model.ckpt（点条件视频扩散，25 帧 ckpt）
#   DUSt3R         download.europe.naverlabs.com → DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
#
# 复用顺序（峰哥约定 2026-11-05）：
#   1) D:\wheel\scene_splatter 已有 → 软链到官方仓消费位置
#   2) 消费侧已有实体文件而 wheel 没有 → 收编进 wheel（mv + 软链回指）
#   3) 都没有 → 下载落盘 wheel，再软链
#
# ⚠️ 三个官方仓都没有发布校验和，无法 SHA256 核对（对照 querysplat 有官方 SHA256SUMS）。
#
# Env:
#   HF_ENDPOINT   HF 镜像端点（本机直连 huggingface.co 不通时自动回退 aifasthub，
#                 2026-11-05 实测 hf 直连断、hf-mirror 证书废）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

FLASH3D_FILE="model_re10k_v2.pth"
VC_FILE="model.ckpt"
DUST3R_FILE="DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
DUST3R_URL="https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

mkdir -p "$WHEEL_SS_DIR" "$SS_DIR/flash3d" "$SS_DIR/viewcrafter/checkpoints"

# HF 直连探测（网络会漂移，动手前复测）
if [ -z "${HF_ENDPOINT:-}" ]; then
    if ! curl -sI --max-time 8 -o /dev/null "https://huggingface.co/api/models/einsafutdinov/flash3d"; then
        echo "⚠️ huggingface.co 直连失败，回退 aifasthub.com 镜像"
        export HF_ENDPOINT="https://aifasthub.com"
    fi
fi
echo "  🌐 HF_ENDPOINT: ${HF_ENDPOINT:-huggingface.co (direct)}"

# ---- 通用：从 wheel 复用池软链（或收编已下载的实体文件） ----
# link_from_wheel <文件名> <wheel 子目录> <消费侧完整路径>
link_from_wheel() {
    local fname="$1" wdir="$2" dest="$3"
    if [ -L "$dest" ]; then
        echo "✅ 已是软链: $dest -> $(readlink "$dest")"; return 0
    fi
    if [ -f "$wdir/$fname" ]; then
        if [ -f "$dest" ]; then
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
        mv "$dest" "$wdir/$fname"
        ln -sfn "$wdir/$fname" "$dest"
        echo "📦 收编已有下载进 wheel 复用池: $wdir/$fname"; return 0
    fi
    return 1   # 都没有，需要下载
}

echo "🚀 [01] flash3d v2 权重（$FLASH3D_FILE）"
if ! link_from_wheel "$FLASH3D_FILE" "$WHEEL_SS_DIR" "$SS_DIR/flash3d/$FLASH3D_FILE"; then
    hf download einsafutdinov/flash3d "$FLASH3D_FILE" --local-dir "$WHEEL_SS_DIR" || {
        echo "❌ flash3d 下载失败（镜像不通时换 HF_ENDPOINT 或走迅雷，见 README_wsl.md）" >&2; exit 1; }
    ln -sfn "$WHEEL_SS_DIR/$FLASH3D_FILE" "$SS_DIR/flash3d/$FLASH3D_FILE"
    echo "🔗 软链: $SS_DIR/flash3d/$FLASH3D_FILE"
fi

echo "🚀 [01] ViewCrafter_25 权重（$VC_FILE）"
if ! link_from_wheel "$VC_FILE" "$WHEEL_SS_DIR" "$SS_DIR/viewcrafter/checkpoints/$VC_FILE"; then
    hf download Drexubery/ViewCrafter_25 "$VC_FILE" --local-dir "$WHEEL_SS_DIR" || {
        echo "❌ ViewCrafter 下载失败（备选：ViewCrafter_16 的 model.ckpt，见 NOTES.md）" >&2; exit 1; }
    ln -sfn "$WHEEL_SS_DIR/$VC_FILE" "$SS_DIR/viewcrafter/checkpoints/$VC_FILE"
    echo "🔗 软链: $SS_DIR/viewcrafter/checkpoints/$VC_FILE"
fi

echo "🚀 [01] DUSt3R ViT-L 权重（$DUST3R_FILE）"
if ! link_from_wheel "$DUST3R_FILE" "$WHEEL_SS_DIR" "$SS_DIR/viewcrafter/checkpoints/$DUST3R_FILE"; then
    curl -L --retry 3 --connect-timeout 15 -o "$WHEEL_SS_DIR/$DUST3R_FILE" "$DUST3R_URL" || {
        echo "❌ DUSt3R 下载失败（naverlabs 欧洲节点，慢/断时挂 proxy.env 里的代理重试）" >&2; exit 1; }
    ln -sfn "$WHEEL_SS_DIR/$DUST3R_FILE" "$SS_DIR/viewcrafter/checkpoints/$DUST3R_FILE"
    echo "🔗 软链: $SS_DIR/viewcrafter/checkpoints/$DUST3R_FILE"
fi

echo "⚠️ 无官方 SHA256 可核对，三个文件均为首次下载（来源已注明）。"
echo "🎉 Done. 消费位置: $SS_DIR/flash3d + $SS_DIR/viewcrafter/checkpoints（实体在 $WHEEL_SS_DIR）"
