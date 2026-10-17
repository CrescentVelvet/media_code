#!/usr/bin/env python3
"""下载 FlashWorld 推理权重到 HF cache 布局（app.py 里 model_id 是字符串，靠 HF_HOME 解析）。

两坨权重：
  1. Wan-AI/Wan2.2-TI2V-5B-Diffusers —— 基座（vae / transformer / text_encoder / tokenizer
     / scheduler），约 34.2 GB。跳过 assets、examples 这些文档图。
  2. imlixinyang/FlashWorld —— model.ckpt 单文件约 19.5 GB（transformer + recon_decoder）。

⚠️ 本机实测（2026-10-17）：
  - huggingface.co 直连 RST；hf-mirror.com 可通但只有 ~1.9 MB/s（单线程）
  - modelscope.cn ~2.3 MB/s
  → 想快就走 迅雷（见 download_urls.md），手动下完用 01b_link_local_weights.sh 挂进 cache。

Env vars:
  HF_ENDPOINT     镜像地址（proxy.env 默认 hf-mirror.com）
  SKIP_WAN=1      跳过 Wan 基座
  SKIP_CKPT=1     跳过 FlashWorld ckpt
  CHECK_ONLY=1    只检查已下载完整性，不下载
"""
import os
import sys
import time

from huggingface_hub import snapshot_download, hf_hub_download

WAN_REPO = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
FW_REPO = os.environ.get("FW_REPO_ID", "imlixinyang/FlashWorld")
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# 文档图 / 示例图不是推理依赖，跳过能省 ~1.5 MB（主要是不让 hf 枚举无关文件）
WAN_IGNORE = ["assets/*", "examples/*", "*.png", "*.JPG", "*.jpg"]
WAN_REQUIRED = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer.json",
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]

GB = 1e9


def human(n):
    return f"{n / GB:.2f} GB"


def find_snapshot(repo_id):
    """在 HF_HOME 里找某 repo 的快照目录（不联网）。"""
    tag = "models--" + repo_id.replace("/", "--")
    root = os.path.join(HF_HOME, "hub", tag, "snapshots")
    if not os.path.isdir(root):
        return None
    snaps = sorted(os.listdir(root))
    return os.path.join(root, snaps[-1]) if snaps else None


def report_wan():
    snap = find_snapshot(WAN_REPO)
    if snap is None:
        print(f"  ❌ {WAN_REPO}: 未下载")
        return False
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(snap) for f in fs)
    missing = [p for p in WAN_REQUIRED
               if not os.path.exists(os.path.join(snap, p))
               or os.path.getsize(os.path.join(snap, p)) == 0]
    print(f"  {'✅' if not missing else '⚠️'} {WAN_REPO}: {human(size)}"
          + (f"，缺 {len(missing)} 个必需文件: {missing[:3]}..." if missing else ""))
    return not missing


def report_ckpt():
    snap = find_snapshot(FW_REPO)
    if snap is None:
        print(f"  ❌ {FW_REPO}: 未下载")
        return False
    p = os.path.join(snap, "model.ckpt")
    if not os.path.exists(p):
        print(f"  ❌ {FW_REPO}: 快照目录在但 model.ckpt 缺失")
        return False
    size = os.path.getsize(p)
    ok = size > 19 * GB
    print(f"  {'✅' if ok else '⚠️'} {FW_REPO}/model.ckpt: {human(size)}")
    return ok


def main():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print("=== [01] FlashWorld 权重下载 ===")
    print(f"  📡 endpoint: {os.environ['HF_ENDPOINT']}")
    print(f"  📁 HF_HOME:  {HF_HOME}")
    print("")

    if os.environ.get("CHECK_ONLY") == "1":
        ok1 = report_wan()
        ok2 = report_ckpt()
        sys.exit(0 if (ok1 and ok2) else 1)

    t0 = time.time()
    if os.environ.get("SKIP_WAN") != "1":
        print(f"📦 {WAN_REPO}  (~34 GB, 断点续传)")
        snapshot_download(repo_id=WAN_REPO, ignore_patterns=WAN_IGNORE,
                          max_workers=int(os.environ.get("MAX_WORKERS", "4")))
        print(f"  ⏱️  {time.time() - t0:.0f}s")
    if os.environ.get("SKIP_CKPT") != "1":
        print(f"📦 {FW_REPO}/model.ckpt  (~19.5 GB)")
        hf_hub_download(repo_id=FW_REPO, filename="model.ckpt")
        print(f"  ⏱️  {time.time() - t0:.0f}s")

    print("")
    print("=== verify ===")
    ok1 = report_wan()
    ok2 = report_ckpt()
    if not (ok1 and ok2):
        sys.exit("❌ 权重不完整，见上方报告")
    print(f"🎉 Done. 总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
