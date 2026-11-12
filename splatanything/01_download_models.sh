#!/usr/bin/env bash
# 01_download_models.sh — 下载 SplatAnything 推理权重（3 个文件，16.6 GB）。
#
# 输出（$SPLATANYTHING_MODEL_DIR）：
#   worldmirror2.safetensors   5.05 GB  WorldMirror-2 几何权重（teacher）
#   SplatAnything.ckpt         11.3 GB  完整高斯学生权重（stage-1）
#   skyseg.onnx                176 MB   可选，--sky_seg 才用
# 并在官方仓 checkpoints/ 建同名软链（configs/inference.yaml 按 repo 相对路径找）。
#
# Env (all optional):
#   HF_BASE_URL=https://aifasthub.com   # HF 官方站本机直连被 reset，走镜像
#   DOWNLOAD_JOBS=8                     # 每个文件的分块并发数
#   SKIP_SKYSEG=1                       # 不装可选的天空分割模型
#   SKIP_SHA=1                          # 跳过 SHA256 校验（不建议）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

REPO_ID="ritianyu/SplatAnything"
SHA_WM2="0e37dcf97576b1d8d9ef16faf3f6fb57bfd1d02e287255d5540117e229077795"
SHA_CKPT="acbb2544cc213a78ecad3ad1f4b88f4a751831a056433331738c30d7b13b3c5c"
SHA_SKY="ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39"

echo "🚀 [01] 下载 SplatAnything 权重（16.6 GB）"
echo "  💾 权重: $SPLATANYTHING_MODEL_DIR"
echo "  🌐 镜像: $HF_BASE_URL"
echo "  🔢 并发: ${DOWNLOAD_JOBS:-8}"

mkdir -p "$SPLATANYTHING_MODEL_DIR" || exit 1

fetch() {  # fetch <filename> <sha256>
    local name="$1" sha="$2"
    local args=("$REPO_ID" "$SPLATANYTHING_MODEL_DIR" "$name")
    [ "${SKIP_SHA:-0}" = "1" ] || args+=(--sha256 "$sha")
    python "$SCRIPT_DIR/hf_download.py" "${args[@]}" || return 1
}

# 小文件先下：链路验证 + 后续可选功能
if [ "${SKIP_SKYSEG:-0}" = "1" ]; then
    echo "⏭️  SKIP_SKYSEG=1，跳过 skyseg.onnx"
else
    fetch skyseg.onnx "$SHA_SKY" || { echo "❌ skyseg.onnx 失败" >&2; exit 1; }
fi

fetch worldmirror2.safetensors "$SHA_WM2" || { echo "❌ worldmirror2.safetensors 失败" >&2; exit 1; }
fetch SplatAnything.ckpt "$SHA_CKPT" || { echo "❌ SplatAnything.ckpt 失败" >&2; exit 1; }

# 官方仓 checkpoints/ 软链（configs/inference.yaml 的相对路径锚在 repo root）
if [ -d "$SPLATANYTHING_DIR" ]; then
    mkdir -p "$SPLATANYTHING_DIR/checkpoints"
    for f in worldmirror2.safetensors SplatAnything.ckpt skyseg.onnx; do
        src="$SPLATANYTHING_MODEL_DIR/$f"
        dst="$SPLATANYTHING_DIR/checkpoints/$f"
        [ -f "$src" ] || continue
        if [ -L "$dst" ] || [ -f "$dst" ]; then
            echo "⏭️  已存在: $dst"
        else
            ln -s "$src" "$dst" && echo "🔗 $dst -> $src"
        fi
    done
else
    echo "⚠️  官方仓不在 $SPLATANYTHING_DIR，跳过软链（先跑 00_setup_env.sh）"
fi

echo "🔍 权重清单"
ls -lh "$SPLATANYTHING_MODEL_DIR" | sed 's/^/  /'
echo "🎉 Done. 下一步: bash $SCRIPT_DIR/02_run_inference.sh"
