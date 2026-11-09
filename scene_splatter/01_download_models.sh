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
#   HF_ENDPOINT   HF 镜像端点（本机直连 huggingface.co 不通时自动回退 hf-mirror.com；
#                 2026-11-09 实测 aifasthub 双通道超时已死，hf-mirror 直连/代理均 200）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

FLASH3D_FILE="model_re10k_v2.pth"
VC_FILE="model.ckpt"
DUST3R_FILE="DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
DUST3R_URL="https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
PY="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"

mkdir -p "$WHEEL_SS_DIR" "$SS_DIR/flash3d" "$SS_DIR/viewcrafter/checkpoints"

# HF 直连探测（网络会漂移，动手前复测）。2026-11-09 实测：huggingface.co API 通，
# 但 LFS 实际走 us.aws.cdn.hf.co（xet-bridge）读超时——API 通不等于能下文件，
# 所以这里测的是「元数据+CDN」组合，失败即回退 hf-mirror.com（LFS 走其自建 CDN）。
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
if [ -z "${HF_ENDPOINT:-}" ]; then
    if ! curl -sI --max-time 8 -o /dev/null "https://huggingface.co/api/models/einsafutdinov/flash3d" \
         || ! curl -sI --max-time 15 -o /dev/null "https://us.aws.cdn.hf.co"; then
        echo "⚠️ huggingface.co 或其 CDN 不可达，回退 hf-mirror.com 镜像"
        if ! curl -sI --max-time 8 -o /dev/null "https://hf-mirror.com/api/models/einsafutdinov/flash3d"; then
            echo "❌ hf-mirror.com 也不可达，请挂代理或手动下载（见 README_wsl.md）" >&2; exit 1
        fi
        export HF_ENDPOINT="https://hf-mirror.com"
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
    # 用 python API 而非 hf CLI：requirements 把 huggingface-hub 钉在 0.25.2，
    # 装新版 hf CLI 会升级 hub 库、可能破坏 transformers 4.28.1 兼容。
    "$PY" -c "from huggingface_hub import hf_hub_download; \
        print(hf_hub_download('einsafutdinov/flash3d', '$FLASH3D_FILE', local_dir='$WHEEL_SS_DIR'))" || {
        echo "❌ flash3d 下载失败（690 MB，xet 存储）" >&2
        echo "   2026-11-09 实测：*.xethub.hf.co / us.aws.cdn.hf.co 直连+代理均 TLS 断连，" >&2
        echo "   hf-mirror 只镜像元数据、对 xet 文件透传死 CDN。" >&2
        echo "   → 请峰哥手动迅雷 https://hf-mirror.com/einsafutdinov/flash3d/resolve/main/$FLASH3D_FILE" >&2
        echo "     落盘 D:\\wheel\\scene_splatter\\ 后重跑本脚本（自动软链）" >&2; exit 1; }
    ln -sfn "$WHEEL_SS_DIR/$FLASH3D_FILE" "$SS_DIR/flash3d/$FLASH3D_FILE"
    echo "🔗 软链: $SS_DIR/flash3d/$FLASH3D_FILE"
fi

echo "🚀 [01] ViewCrafter_25 权重（$VC_FILE）"
if ! link_from_wheel "$VC_FILE" "$WHEEL_SS_DIR" "$SS_DIR/viewcrafter/checkpoints/$VC_FILE"; then
    "$PY" -c "from huggingface_hub import hf_hub_download; \
        print(hf_hub_download('Drexubery/ViewCrafter_25', '$VC_FILE', local_dir='$WHEEL_SS_DIR'))" || {
        echo "❌ ViewCrafter 下载失败（10.4 GB，xet 存储，CDN 死局同上）" >&2
        echo "   → 请峰哥手动迅雷 https://hf-mirror.com/Drexubery/ViewCrafter_25/resolve/main/$VC_FILE" >&2
        echo "     落盘 D:\\wheel\\scene_splatter\\ 后重跑本脚本（自动软链）" >&2
        echo "   备选：ViewCrafter_16 的 model.ckpt（小一号，见 NOTES.md）" >&2; exit 1; }
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
