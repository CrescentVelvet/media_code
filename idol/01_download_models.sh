#!/usr/bin/env bash
# 01_download_models.sh —— 下载 idol 全部权重到 D:\wheel（HF 缓存复用，不重复下载）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

idol_check_hf_token || exit 1

echo "🚀 [01] 下载 Idol 权重"
echo "  📁 HF_HOME:  $HF_HOME"
echo "  🌐 代理:     ${https_proxy:-<无>}"
echo "  🏋️ 主体:     speridlabs/idol          (~3.0 GB)"
echo "  🏋️ 编码器:   facebook/dinov3-vith16plus-pretrain-lvd1689m (~1.7 GB, gated)"
echo "  🏋️ VAE:      black-forest-labs/FLUX.2-dev  仅 vae/ 子目录 (~0.2 GB, gated)"
echo "  ⚠️  FLUX.2-dev 全仓 30GB+，这里只取 vae，勿用 snapshot_download 全量"

# hf_transfer 未安装时关掉开关，否则 huggingface_hub 直接抛错。
# 默认已在 _env.sh 里设为 0（代理环境下会卡死，见 _env.sh 注释）。
python -c "import hf_transfer" 2>/dev/null || export HF_HUB_ENABLE_HF_TRANSFER=0
echo "  ⚡ hf_transfer: $([ "$HF_HUB_ENABLE_HF_TRANSFER" = "1" ] && echo 启用 || echo 关闭/未装)"

mkdir -p "$HF_HOME" "$IDOL_MODELS"

python - <<'PY'
import json, os, sys, time
from huggingface_hub import snapshot_download

HF_HOME = os.environ["HF_HOME"]
MODELS = os.environ["IDOL_MODELS"]

JOBS = [
    ("speridlabs/idol", None,
     ["config.json", "denoiser.safetensors", "decoder.safetensors",
      "decoder_gs.safetensors", "birefnet.safetensors"]),
    ("facebook/dinov3-vith16plus-pretrain-lvd1689m", None, None),   # 全仓（AutoModel 需 config 全量）
    ("black-forest-labs/FLUX.2-dev", None,                          # 只要 vae 子目录
     ["vae/*", "model_index.json"]),
]

for repo, _, patterns in JOBS:
    t0 = time.time()
    print(f"\n📦 {repo}")
    try:
        path = snapshot_download(repo, allow_patterns=patterns, max_workers=4)
    except Exception as e:
        print(f"❌ {repo} 下载失败: {type(e).__name__}: {str(e)[:400]}")
        sys.exit(1)
    # snapshot 目录里全是软链（指向 blobs/），getsize 会跟随软链，别用 islink 过滤否则统计成 0
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(path) for f in fs
               if os.path.exists(os.path.join(dp, f)))
    print(f"  ✅ {path}")
    print(f"  📐 {size/1e9:.2f} GB   ⏱️ {time.time()-t0:.0f}s")

# ---- 校验主体权重齐全 ----
main = [p for p in os.listdir(os.path.join(HF_HOME, "hub"))
        if p.startswith("models--speridlabs--idol")]
if not main:
    sys.exit("❌ 没找到 speridlabs/idol 的缓存目录")
snap_root = os.path.join(HF_HOME, "hub", main[0], "snapshots")
snap = os.path.join(snap_root, os.listdir(snap_root)[0])
need = ["config.json", "denoiser.safetensors", "decoder.safetensors",
        "decoder_gs.safetensors", "birefnet.safetensors"]
missing = [f for f in need if not os.path.exists(os.path.join(snap, f))]
if missing:
    sys.exit(f"❌ 主体权重缺文件: {missing}")

# ---- 导出稳定软链目录，供 --weights 直接用（避开 snapshot 哈希名） ----
os.makedirs(MODELS, exist_ok=True)
for f in need:
    dst = os.path.join(MODELS, f)
    src = os.path.join(snap, f)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)
print(f"\n✅ 权重软链: {MODELS}")
for f in need:
    print(f"   {f}  {os.path.getsize(os.path.join(MODELS, f))/1e9:.2f} GB")

cfg = json.load(open(os.path.join(MODELS, "config.json")))
print(f"\n📋 config: image_size={cfg['image_size']} max_voxel_level={cfg['max_voxel_level']}")
print(f"📋 encoders: {cfg['image_encoders']}")
PY
if [ $? -ne 0 ]; then
    echo "❌ 下载/校验失败" >&2
    exit 1
fi

echo "💾 D:\\wheel 占用: $(du -sh "$HF_HOME" 2>/dev/null | cut -f1)"
echo "🎉 [01] Done. 下一步: bash 02_run_inference.sh"
