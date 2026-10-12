#!/usr/bin/env bash
# 01_download_models.sh — 把迅雷下载好的权重/数据从暂存区（Windows 盘）搬入 Linux fs。
#
# 为什么不自动下载：本机 huggingface.co 直连被 RST、hf-mirror.com 不稳定（2026-10-10 实测），
# 大文件由峰哥用迅雷下到 D:\wheel\gavs_ms\（WSL 里 /mnt/d/wheel/gavs_ms）。
#
# 期望的暂存区布局（链接清单见 README_wsl.md）：
#   /mnt/d/wheel/gavs_ms/
#   ├── model_re10k_v2.pth                  # Flash3D 预训练权重（必需，~690MB）
#   ├── unidepth-v1-vitl14-pytorch_model.bin # UniDepth v1 权重（必需，~1.3GB）
#   └── dataset/
#       ├── dynamic_dance.zip               # 场景数据（至少下一个，~0.7-2GB/个）
#       └── ...（其余 14 场景可选）
#
# 用法：
#   bash gavs/01_download_models.sh              # 全部检查 + 搬运 + 解压
#   SKIP_UNZIP=1 bash gavs/01_download_models.sh # 只搬权重不解压数据
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [01] 搬运 GaVS 权重与数据"
echo "  📥 暂存区: $GAVS_STAGING_DIR"
echo "  🏋️ 数据根: $GAVS_DATA_ROOT"

if [ ! -d "$GAVS_STAGING_DIR" ]; then
    echo "❌ 暂存区不存在: $GAVS_STAGING_DIR" >&2
    echo "   请先用迅雷按 README_wsl.md 的链接下载到 D:\\wheel\\gavs_ms\\" >&2
    exit 1
fi

mkdir -p "$GAVS_DATA_ROOT/re10k_v2_checkpoints" "$GAVS_DATA_ROOT/dataset"
_missing=0

# ── 1. Flash3D 预训练权重 ─────────────────────────────────────────────────
CKPT="$GAVS_DATA_ROOT/re10k_v2_checkpoints/model_re10k_v2.pth"
if [ -s "$CKPT" ]; then
    echo "⏭️  Flash3D 权重已在: $CKPT"
elif [ -s "$GAVS_STAGING_DIR/model_re10k_v2.pth" ]; then
    echo "📦 搬运 model_re10k_v2.pth (~690MB)..."
    cp "$GAVS_STAGING_DIR/model_re10k_v2.pth" "$CKPT"
    echo "  ✅ done"
else
    echo "❌ 缺 model_re10k_v2.pth（迅雷链接见 README_wsl.md）" >&2
    _missing=1
fi

# ── 2. UniDepth v1 权重 → 种进 HF 缓存（_env.sh 已设 HF_HUB_OFFLINE=1）────
# hubconf.py: hf_hub_download(repo_id="lpiccinelli/unidepth-v1-vitl14",
#                             filename="pytorch_model.bin")
# 离线模式下按 snapshots/* 扫描，目录名任意，这里用 "local"。
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
UNIDEPTH_REPO_CACHE="$HF_CACHE/models--lpiccinelli--unidepth-v1-vitl14"
UNIDEPTH_SNAP="$UNIDEPTH_REPO_CACHE/snapshots/local"
if [ -s "$UNIDEPTH_SNAP/pytorch_model.bin" ]; then
    echo "⏭️  UniDepth 权重已种入 HF 缓存"
else
    _src=""
    for _cand in "$GAVS_STAGING_DIR/unidepth-v1-vitl14-pytorch_model.bin" \
                 "$GAVS_STAGING_DIR/pytorch_model.bin"; do
        [ -s "$_cand" ] && _src="$_cand" && break
    done
    if [ -n "$_src" ]; then
        echo "📦 种入 UniDepth 权重 -> HF 缓存 (~1.3GB)..."
        mkdir -p "$UNIDEPTH_SNAP"
        cp "$_src" "$UNIDEPTH_SNAP/pytorch_model.bin"
        echo "  ✅ done: $UNIDEPTH_SNAP"
    else
        echo "❌ 缺 UniDepth 权重（迅雷链接见 README_wsl.md）" >&2
        _missing=1
    fi
fi
# hf_hub_download 离线解析需要 refs/main -> snapshots/local 指针（新版 huggingface_hub
# 不再扫 snapshots 兜底，缺 refs 会报 LocalEntryNotFound；2026-10-12 实测踩坑）
if [ -s "$UNIDEPTH_SNAP/pytorch_model.bin" ] && [ ! -s "$UNIDEPTH_REPO_CACHE/refs/main" ]; then
    mkdir -p "$UNIDEPTH_REPO_CACHE/refs"
    printf "local" > "$UNIDEPTH_REPO_CACHE/refs/main"
    echo "🔧 补建 HF 缓存 refs/main -> local"
fi

# ── 3. 场景数据解压 ───────────────────────────────────────────────────────
if [ "${SKIP_UNZIP:-0}" != "1" ]; then
    shopt -s nullglob
    # 3a. staging 里已是解压目录的（峰哥手动解压过）：直接搬到 Linux fs
    for _dir in "$GAVS_STAGING_DIR/dataset"/*/; do
        _name="$(basename "$_dir")"
        if [ -d "$GAVS_DATA_ROOT/dataset/$_name" ]; then
            echo "⏭️  场景已在数据根: $_name"
            continue
        fi
        echo "📦 搬运已解压场景 $_name (~$(du -sh "$_dir" | cut -f1)，走 9p 较慢)..."
        mkdir -p "$GAVS_DATA_ROOT/dataset/$_name"
        cp -r "$_dir". "$GAVS_DATA_ROOT/dataset/$_name/" && echo "  ✅ $_name" || {
            echo "❌ copy failed: $_dir" >&2; _missing=1; }
    done
    # 3b. staging 里的 zip：解压归位
    _zips=("$GAVS_STAGING_DIR/dataset"/*.zip)
    for _zip in "${_zips[@]}"; do
        _name="$(basename "$_zip" .zip)"
        if [ -d "$GAVS_DATA_ROOT/dataset/$_name" ]; then
            echo "⏭️  场景已解压: $_name"
            continue
        fi
        echo "📦 解压 $_name.zip ..."
        _tmp="$GAVS_DATA_ROOT/dataset/.tmp_$_name"
        rm -rf "$_tmp"; mkdir -p "$_tmp"
        unzip -q -o "$_zip" -d "$_tmp" || {
            echo "❌ unzip failed: $_zip" >&2; _missing=1; rm -rf "$_tmp"; continue; }
        # zip 内层结构不确定（可能带顶层目录、也可能平铺），统一归位到 dataset/<scene>/
        if [ -d "$_tmp/$_name" ]; then
            mv "$_tmp/$_name" "$GAVS_DATA_ROOT/dataset/$_name"
            rm -rf "$_tmp"
        else
            mkdir -p "$GAVS_DATA_ROOT/dataset/$_name"
            mv "$_tmp"/* "$GAVS_DATA_ROOT/dataset/$_name/" 2>/dev/null
            rm -rf "$_tmp"
        fi
        echo "  ✅ $_name"
    done
    # 3a/3b 都没产出任何场景 → 数据真的缺
    if ! ls -d "$GAVS_DATA_ROOT/dataset"/*/ >/dev/null 2>&1; then
        echo "⚠️ 暂存区没有场景数据（$GAVS_STAGING_DIR/dataset/ 下既无 zip 也无解压目录）" >&2
        _missing=1
    fi
    shopt -u nullglob
fi

# ── 4. 验证 ───────────────────────────────────────────────────────────────
echo ""
echo "--- verification ---"
[ -s "$CKPT" ] && echo "  ✅ Flash3D 权重" || echo "  [MISS] Flash3D 权重"
[ -s "$UNIDEPTH_SNAP/pytorch_model.bin" ] && echo "  ✅ UniDepth 权重（HF 缓存）" || echo "  [MISS] UniDepth 权重"
ls -d "$GAVS_DATA_ROOT/dataset"/*/ 2>/dev/null | while read -r d; do
    echo "  ✅ 场景: $(basename "$d")"
done
[ -d "$GAVS_DATA_ROOT/dataset/$SCENE" ] || echo "  [MISS] 默认场景 $SCENE（02 需要）"

if [ "$_missing" = "1" ]; then
    echo ""
    echo "⚠️ 有缺失项，02 跑不起来。补齐后重跑本脚本。"
    exit 1
fi
echo ""
echo "🎉 [01] Done. 下一步: SCENE=$SCENE bash gavs/02_run_finetune.sh"
