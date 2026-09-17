#!/usr/bin/env bash
# 01_download_models.sh — 下载 SHARP 权重（唯一一个文件）。
#
# 权重：sharp_2572gikvuh.pt — 2,809,738,232 bytes（≈2.62 GiB），
#       含完整模型（Depth-Pro 编码器 + Gaussian 回归头），非 gated，无需 token。
# 官方 CDN：https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt
#
# 迅雷下载（推荐，国内满速）：
#   1. 迅雷新建任务，粘贴上面这条官方 CDN 直链（支持 Range 断点续传）
#   2. 下完把文件放到 $SHARP_CKPT（见下方回显的实际路径），或：
#        SRC=/mnt/d/downloads/sharp_2572gikvuh.pt bash sharp/01_download_models.sh
#
# 用法：
#   bash sharp/01_download_models.sh                 # 从 Apple CDN 下
#   SRC=/mnt/d/downloads/sharp_2572gikvuh.pt bash sharp/01_download_models.sh   # 导入迅雷下好的文件
#   FORCE=1 bash sharp/01_download_models.sh         # 重下（忽略已存在）
#   SHARP_CKPT_URL=<url> bash sharp/01_download_models.sh   # 换源（如 HF 镜像）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SRC="${SRC:-}"
FORCE="${FORCE:-0}"
DST="$SHARP_CKPT"
EXPECT_SIZE="$SHARP_CKPT_SIZE"

echo "🚀 [01] 准备 SHARP 权重"
echo "  🏋️ 目标:  $DST"
echo "  🌐 源:    ${SRC:-$SHARP_CKPT_URL}"
echo "  📐 期望大小: $EXPECT_SIZE bytes (≈2.62 GiB)"
echo ""

mkdir -p "$(dirname "$DST")"

# ── 校验函数：字节数精确匹配才算完整 ─────────────────────────────────────
_verify() {
    local f="$1"
    [ -f "$f" ] || return 1
    local sz
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$sz" = "$EXPECT_SIZE" ]; then
        echo "  ✅ size OK: $sz bytes"
        return 0
    fi
    echo "  ⚠️  size mismatch: $sz != $EXPECT_SIZE（文件不完整或下错版本）" >&2
    return 1
}

# ── 情况 A：导入外部已下载的文件（迅雷 / 手工）────────────────────────────
if [ -n "$SRC" ]; then
    if [ ! -f "$SRC" ]; then
        echo "❌ ERROR: SRC 文件不存在: $SRC" >&2
        exit 1
    fi
    if ! _verify "$SRC"; then
        echo "❌ ERROR: 源文件大小不符，拒绝导入" >&2
        exit 1
    fi
    if [ "$(readlink -f "$SRC")" = "$(readlink -f "$DST" 2>/dev/null)" ]; then
        echo "⏭️  SRC 与目标同一文件，无需搬运"
    else
        echo "📦 搬运 -> $DST"
        cp -f "$SRC" "$DST.part" && mv -f "$DST.part" "$DST" || {
            echo "❌ ERROR: 搬运失败（磁盘空间？）" >&2
            exit 1
        }
        echo "  ✅ done"
    fi
    echo ""
    echo "🎉 [01] Done. 下一步："
    echo "  GPU=0 INPUT=/path/to/images bash sharp/02_run_inference.sh"
    exit 0
fi

# ── 情况 B：目标已存在且完整，直接跳过 ────────────────────────────────────
if [ "$FORCE" != "1" ] && _verify "$DST"; then
    echo "⏭️  权重已存在且完整，跳过下载（FORCE=1 可强制重下）"
    echo ""
    echo "🎉 [01] Done."
    exit 0
fi

# ── 情况 C：网络下载（curl 优先，wget 兜底，均支持断点续传）──────────────
_tmp="$DST.part"
if command -v curl >/dev/null 2>&1; then
    echo "📦 curl 下载中（断点续传，Ctrl+C 可中断，重跑自动续传）..."
    curl -L -C - --retry 5 --retry-delay 3 --connect-timeout 30 \
        -o "$_tmp" "$SHARP_CKPT_URL" || {
        echo "❌ curl 下载失败" >&2
        echo "  备用方案：迅雷下 $SHARP_CKPT_URL_HF" >&2
        echo "           再 SRC=<文件路径> bash sharp/01_download_models.sh" >&2
        exit 1
    }
elif command -v wget >/dev/null 2>&1; then
    echo "📦 wget 下载中（断点续传）..."
    wget --no-check-certificate -c -O "$_tmp" "$SHARP_CKPT_URL" || {
        echo "❌ wget 下载失败" >&2
        exit 1
    }
else
    echo "❌ ERROR: 既没有 curl 也没有 wget" >&2
    exit 1
fi

if ! _verify "$_tmp"; then
    echo "❌ ERROR: 下载不完整，保留 $_tmp 供续传（重跑本脚本）" >&2
    exit 1
fi
mv -f "$_tmp" "$DST"
echo "  ✅ saved: $DST"

echo ""
echo "🎉 [01] Done. 下一步："
echo "  GPU=0 INPUT=/path/to/images bash sharp/02_run_inference.sh"
