#!/usr/bin/env bash
# structsplat/01_download_models.sh — 下载三份权重并软链进官方仓 ckpts/
#
# 权重池约定（2026-11-05）：新下载落盘 D:\wheel\<项目>_ms/，消费侧软链不复制。
#
# ⚠️ 通道选择（2026-11-19 实测，本机 WSL）：
#   HuggingFace 直连 → curl(35) Connection reset，完全不通
#   HuggingFace 走 v2rayN 代理 → 约 90 KB/s（5GB 要 15 小时，不可用）
#   Modelscope 直连 → 约 2 MB/s（5GB 约 42 分钟）✅ 首选
#   所以默认把 Modelscope 放第一位，HF 只作兜底。
#
# 三份权重：
#   1. facebook/VGGT-1B                                  model.pt            ≈ 4.68 GB
#   2. facebook/dinov3-convnext-large-pretrain-lvd1689m  全部文件            ≈ 749 MB
#   3. DazzlingSun/structsplat                           pytorch_model.bin   ≈ 3.50 GB
#
# 官方 config/dl3dv.yaml 里 path 写的是相对路径 ckpts/vggt、ckpts/dinov3_convnext_large，
# ckpt 写的是 ckpts/structsplat/pytorch_model.bin → 必须软链到这三个位置，不能挪走。
#
# 用法：
#   bash structsplat/01_download_models.sh            # 全部
#   bash structsplat/01_download_models.sh vggt       # 只下 VGGT-1B
#   bash structsplat/01_download_models.sh dino ss    # 指定多个
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

mkdir -p "$WHEEL_STRUCTSPLAT_DIR" "$WHEEL_VGGT1B_DIR" "$WHEEL_DINOV3CNX_DIR" "$SS_CKPT_DIR"

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(vggt dino ss)
has() { for t in "${TARGETS[@]}"; do [ "$t" = "$1" ] && return 0; done; return 1; }

# ------------------------------------------------------------
# 下载器
#   $1=url  $2=输出文件  $3=期望最少字节数（0=不校验）  $4=是否绕过代理(direct/proxy)
# ------------------------------------------------------------
dl() {
    local url="$1" out="$2" minbytes="${3:-0}" mode="${4:-direct}" name
    name="$(basename "$out")"
    if [ -f "$out" ]; then
        local sz; sz=$(stat -c%s "$out")
        if [ "$minbytes" -gt 0 ] && [ "$sz" -ge "$minbytes" ]; then
            echo "   ⏭️  已存在且体积达标: $name ($(numfmt --to=iec "$sz"))"
            return 0
        fi
        echo "   ⚠️  已存在但不完整 ($(numfmt --to=iec "$sz"))，续传"
    fi
    echo "   ⬇️  $name  [$mode]"
    local noproxy=()
    [ "$mode" = "direct" ] && noproxy=(--noproxy '*')
    # -C - 续传；--retry 抗网络漂移；-L 跟重定向；-# 简洁进度条
    curl -fL -C - --retry 6 --retry-delay 2 --retry-all-errors \
         --connect-timeout 30 "${noproxy[@]}" -# \
         -o "$out" "$url" || return 1
    local sz; sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$minbytes" -gt 0 ] && [ "$sz" -lt "$minbytes" ]; then
        echo "   ❌ 体积不足: $name 实际 $(numfmt --to=iec "$sz") < 期望 $(numfmt --to=iec "$minbytes")"
        return 1
    fi
    echo "   ✅ $name ($(numfmt --to=iec "$sz"))"
    return 0
}

# Modelscope 优先（本机实测 ~2MB/s），HF 兜底（~90KB/s 或直接不通）
# $1=ms_repo  $2=hf_repo  $3=file  $4=out  $5=minbytes
dl_weight() {
    local ms_repo="$1" hf_repo="$2" file="$3" out="$4" minbytes="$5"
    dl "https://www.modelscope.cn/models/$ms_repo/resolve/master/$file" \
       "$out" "$minbytes" direct && return 0
    echo "   ↪️  Modelscope 失败，改走 HF 代理"
    dl "https://huggingface.co/$hf_repo/resolve/main/$file" \
       "$out" "$minbytes" proxy
}

# 软链原子替换：$1=目标链接  $2=真实路径
link_to() {
    local dst="$1" src="$2"
    if [ -L "$dst" ] && [ "$(readlink -f "$dst")" = "$(readlink -f "$src")" ]; then
        echo "   ⏭️  软链已正确: $dst -> $src"
        return 0
    fi
    if [ -e "$dst" ] && [ ! -L "$dst" ]; then
        echo "   ⚠️  $dst 是实体目录/文件，移到 .bak"
        mv "$dst" "$dst.bak.$(date +%Y%m%d%H%M%S)"
    fi
    ln -sfn "$src" "$dst"
    echo "   🔗 $dst -> $src"
}

# ============================================================
# 1. VGGT-1B —— geo_encoder
# ============================================================
# model.pt → model.safetensors（幂等：已存在就跳过）
# 见调用处的长注释，这是 VGGT.from_pretrained 在本地目录下能跑通的必要条件。
he_vggt_safetensors() {
    local src="$WHEEL_VGGT1B_DIR/model.pt"
    local dst="$WHEEL_VGGT1B_DIR/model.safetensors"
    if [ ! -f "$src" ]; then
        echo "   ⚠️  跳过转换：$src 不存在"
        return 0
    fi
    if [ -f "$dst" ]; then
        local ds; ds=$(stat -c%s "$dst")
        if [ "$ds" -gt 1000000000 ]; then
            echo "   ⏭️  model.safetensors 已存在 ($(numfmt --to=iec "$ds"))"
            return 0
        fi
    fi
    echo "   🔄 model.pt → model.safetensors（huggingface_hub 本地分支只认 safetensors）"
    python - "$src" "$dst" <<'PY'
import sys, os, torch
from safetensors.torch import save_file
src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu", weights_only=True)
if not isinstance(sd, dict):
    raise SystemExit(f"❌ {type(sd).__name__} 不是 state_dict，无法直接转 safetensors")
# safetensors 要求连续内存且不共享存储
sd = {k: (v.contiguous().clone() if isinstance(v, torch.Tensor) else v)
      for k, v in sd.items()}
save_file(sd, dst)
print(f"   ✅ {os.path.getsize(dst):,} B  ({len(sd)} tensors)")
PY
}

if has vggt; then
    echo ""
    echo "📦 [1/3] VGGT-1B (geo_encoder) → $WHEEL_VGGT1B_DIR"
    # 必须整仓结构（config.json + model.pt），VGGT.from_pretrained 会读 config
    # config.json 实测只有 62 字节（embed_dim/img_size/patch_size 三字段）
    dl_weight "facebook/VGGT-1B" "facebook/VGGT-1B" "config.json" \
              "$WHEEL_VGGT1B_DIR/config.json" 20
    dl_weight "facebook/VGGT-1B" "facebook/VGGT-1B" "model.pt" \
              "$WHEEL_VGGT1B_DIR/model.pt" 4900000000
    # ⚠️ 必须转成 model.safetensors（2026-11-19 实测踩坑）：
    #    VGGT 继承 huggingface_hub.PyTorchModelHubMixin，而本机 huggingface_hub 的
    #    _from_pretrained 在 model_id 是**本地目录**时硬编码只读 model.safetensors：
    #        if os.path.isdir(model_id):
    #            model_file = os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE)
    #    只有从 hub 下载才 fallback 到 pytorch_model.bin。HF 上 VGGT-1B 只给 model.pt，
    #    于是 VGGT.from_pretrained("ckpts/vggt") 直接：
    #        FileNotFoundError: No such file or directory: ckpts/vggt/model.safetensors
    #    报错里看不出真因（像是路径写错，其实是格式不匹配），故在下载侧就地转掉。
    he_vggt_safetensors
    link_to "$SS_CKPT_DIR/vggt" "$WHEEL_VGGT1B_DIR"
fi

# ============================================================
# 2. DINOv3-ConvNeXt-Large —— sem_encoder
# ============================================================
if has dino; then
    echo ""
    echo "📦 [2/3] DINOv3-ConvNeXt-Large (sem_encoder) → $WHEEL_DINOV3CNX_DIR"
    # 注：本机这份权重已由用户从 HF 手动下载好（784,957,976 B，实测可加载）。
    # Modelscope 无此 repo，走了 HF；若已有文件会自动跳过。
    dl_weight "facebook/dinov3-convnext-large-pretrain-lvd1689m" \
              "facebook/dinov3-convnext-large-pretrain-lvd1689m" \
              "config.json" "$WHEEL_DINOV3CNX_DIR/config.json" 200
    dl_weight "facebook/dinov3-convnext-large-pretrain-lvd1689m" \
              "facebook/dinov3-convnext-large-pretrain-lvd1689m" \
              "model.safetensors" "$WHEEL_DINOV3CNX_DIR/model.safetensors" 700000000
    dl_weight "facebook/dinov3-convnext-large-pretrain-lvd1689m" \
              "facebook/dinov3-convnext-large-pretrain-lvd1689m" \
              "preprocessor_config.json" "$WHEEL_DINOV3CNX_DIR/preprocessor_config.json" 50 \
              || echo "   ⚠️  可选文件 preprocessor_config.json 缺失，继续"
    link_to "$SS_CKPT_DIR/dinov3_convnext_large" "$WHEEL_DINOV3CNX_DIR"
fi

# ============================================================
# 3. StructSplat 官方 ckpt —— 主权重
# ============================================================
if has ss; then
    echo ""
    echo "📦 [3/3] StructSplat ckpt → $WHEEL_STRUCTSPLAT_DIR"
    SS_CKPT_FILE="$WHEEL_STRUCTSPLAT_DIR/pytorch_model.bin"
    # Modelscope 上没有这个 repo（API 返回 record not found），只能走 HF。
    # HF 单流经代理仅 ~90 KB/s（3.3GB 要 10 小时），必须用多并发分块：
    # 实测 16 并发聚合 1.0~1.1 MB/s。00c 支持断点续传（.part/NNNNNN 分块）。
    echo "   注：Modelscope 无 DazzlingSun/structsplat，走 HF 多并发分块"
    SS_EXPECT=3497564858
    SS_CUR=0
    [ -f "$SS_CKPT_FILE" ] && SS_CUR=$(stat -c%s "$SS_CKPT_FILE")
    if [ "$SS_CUR" -ge "$SS_EXPECT" ]; then
        echo "   ⏭️  已完整: $(numfmt --to=iec "$SS_CUR")"
    else
        python "$SCRIPT_DIR/00c_par_download.py" \
            "https://huggingface.co/DazzlingSun/structsplat/resolve/main/pytorch_model.bin" \
            "$SS_CKPT_FILE" \
            --chunk-mb 8 --workers 16 --expect-size "$SS_EXPECT" \
            || echo "   ⚠️  下载未完成（可重跑本脚本续传）"
    fi
    # 官方 yaml 期望 ckpts/structsplat/pytorch_model.bin
    mkdir -p "$SS_CKPT_DIR/structsplat"
    link_to "$SS_CKPT_DIR/structsplat/pytorch_model.bin" "$SS_CKPT_FILE"
fi

# ============================================================
# 汇总
# ============================================================
echo ""
echo "🔍 权重就位检查"
check() {
    local p="$1" label="$2"
    if [ -e "$p" ]; then
        local real; real="$(readlink -f "$p")"
        local sz; sz=$(stat -Lc%s "$p" 2>/dev/null || echo 0)
        printf "  ✅ %-32s %s  (%s)\n" "$label" "$(numfmt --to=iec "$sz")" "$real"
    else
        printf "  ❌ %-32s 缺失\n" "$label"
    fi
}
check "$SS_CKPT_DIR/vggt/config.json"                        "vggt/config.json"
check "$SS_CKPT_DIR/vggt/model.pt"                           "vggt/model.pt"
check "$SS_CKPT_DIR/dinov3_convnext_large/config.json"       "dinov3/config.json"
check "$SS_CKPT_DIR/dinov3_convnext_large/model.safetensors" "dinov3/model.safetensors"
check "$SS_CKPT_DIR/structsplat/pytorch_model.bin"           "structsplat/pytorch_model.bin"

echo ""
echo "🎉 [01] 权重下载完成"
echo "   下一步: bash structsplat/02_run_inference.sh"
