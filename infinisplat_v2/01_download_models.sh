#!/usr/bin/env bash
# 01_download_models.sh — 下载 InfiniSplat v2 推理权重与官方示例。
#
# 输出：
#   $WEIGHTS_DIR/{config.json, model.safetensors}   (≈6.6 GB, SHA256 校验)
#   $INFINISPLAT_V2/examples/data/{hypersim_provided,dtu_provided,hypersim_da3_pose}
#   examples-png.zip 解压到 $INFINISPLAT_V2/
#
# Env (all optional):
#   HF_BASE_URL=https://aifasthub.com   # 镜像站；官方 huggingface.co 在本机不可达
#   DOWNLOAD_JOBS=8                     # 分块并发数
#   SKIP_EXAMPLES=1                     # 只下权重
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

REPO_ID="Edisoneh/InfiniSplat-v2"
SHA_WEIGHTS="6d23f9cef753fb7f72ecd88fa8e074a458de5116591276da85129909e73a66e8"

echo "🚀 [01] 下载 InfiniSplat v2 权重"
echo "  📁 仓库: $INFINISPLAT_V2"
echo "  💾 权重: $WEIGHTS_DIR"
echo "  🌐 镜像: $HF_BASE_URL"
echo "  🔢 并发: ${DOWNLOAD_JOBS:-8}"

if [ ! -d "$INFINISPLAT_V2" ]; then
    echo "❌ ERROR: 官方仓不在 $INFINISPLAT_V2（先跑 00_setup_env.sh）" >&2
    exit 1
fi

mkdir -p "$WEIGHTS_DIR"

# 1) config.json（小文件，先落盘：里面就有 weights_sha256）
python "$SCRIPT_DIR/hf_download.py" "$REPO_ID" "$WEIGHTS_DIR" config.json
if [ $? -ne 0 ]; then echo "❌ config.json 下载失败" >&2; exit 1; fi

# 2) model.safetensors（6.6 GB，分块并发 + SHA256 校验）
python "$SCRIPT_DIR/hf_download.py" "$REPO_ID" "$WEIGHTS_DIR" model.safetensors --sha256 "$SHA_WEIGHTS"
if [ $? -ne 0 ]; then echo "❌ model.safetensors 下载/校验失败" >&2; exit 1; fi

if [ "${SKIP_EXAMPLES:-0}" = "1" ]; then
    echo "⏭️  SKIP_EXAMPLES=1，跳过示例下载"
    echo "🎉 Done."
    exit 0
fi

# 3) 官方示例（114 张 PNG + 相机文件，90 MB）
python "$SCRIPT_DIR/hf_download.py" "$REPO_ID" "$INFINISPLAT_V2" examples-png.zip
if [ $? -ne 0 ]; then echo "❌ examples-png.zip 下载失败" >&2; exit 1; fi

echo "📦 解压 examples-png.zip"
python -m zipfile -e "$INFINISPLAT_V2/examples-png.zip" "$INFINISPLAT_V2/"
if [ $? -ne 0 ]; then echo "❌ 解压失败" >&2; exit 1; fi

echo "🔍 示例目录检查"
for d in hypersim_provided dtu_provided hypersim_da3_pose; do
    n=$(find "$INFINISPLAT_V2/examples/data/$d/images" -maxdepth 1 -type f 2>/dev/null | wc -l)
    printf "  %-22s %s 张图\n" "$d" "$n"
done

echo "🎉 Done."
