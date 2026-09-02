#!/usr/bin/env python3
"""09c — 离线解码 09b 存盘的 latent（不重跑 37 分钟去噪）。

起因：run#5 去噪 50/50 全部完成，却因为 09b 钩子里一个 NameError 崩在
VAE 解码前，37 分钟成果全丢。此后 09b 会在每次 vae.decode 前把输入 latent
落盘（latent_decode*.pt）。这个脚本负责把那份 latent 离线解码成 mp4：
即使 09b 的解码/存片环节再出任何问题，去噪成果都能救回来。

只加载 VAE（fp32 ~10.4GB，约 2-3 分钟），不碰 transformer/text_encoder。

用法（WSL）：
  cd /mnt/c/code/media_code && conda activate minimax_h3
  MODEL_PATH=/mnt/d/wheel/minimaxh3_ms \
  LATENT=/mnt/d/output/minimaxh3_rotate_results/diag_baseline/latent_decode1.pt \
  OUTPUT_DIR=/mnt/d/output/minimaxh3_rotate_results/diag_baseline \
  python minimax_h3/09c_decode_latent.py

Env vars:
  MODEL_PATH  默认 /mnt/d/wheel/minimaxh3_ms
  LATENT      要解码的 latent_decode*.pt 路径（必填）
  DEVICE      默认 cuda:0
  OUTPUT_DIR  默认 latent 同目录
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/d/wheel/minimaxh3_ms")
LATENT = os.environ.get("LATENT", "")
DEVICE = os.environ.get("DEVICE", "cuda:0")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "")


def stats(name, t):
    import torch
    t = t.detach().float()
    finite = torch.isfinite(t)
    if int(finite.sum()) == 0:
        print(f"  {name:<28} 全部 NaN/Inf !!")
        return
    print(
        f"  {name:<28} shape={tuple(t.shape)} "
        f"mean={t[finite].mean():+.4f} std={t[finite].std():.4f} "
        f"min={t[finite].min():+.3f} max={t[finite].max():+.3f} "
        f"nan/inf={int((~finite).sum())}"
    )


def to_uint8(t):
    """[-1,1] float tensor (C,T,H,W) -> (T,H,W,C) uint8 numpy."""
    import torch
    x = t.clamp(0, 1)
    x = (x * 255).round().to(torch.uint8)
    return x.permute(1, 2, 3, 0).cpu().numpy()


def main():
    import torch
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.utils.export_utils import encode_video

    if not LATENT:
        print("❌ 必须设置 LATENT=/path/to/latent_decode*.pt（由 09b 存盘）")
        raise SystemExit(1)
    out_dir = OUTPUT_DIR or os.path.dirname(LATENT)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 68)
    print("🔬 09c — 离线解码 09b 存盘的 latent")
    print(f"  latent : {LATENT}")
    print(f"  model  : {MODEL_PATH}")
    print(f"  device : {DEVICE}")
    print("=" * 68)

    ckpt = torch.load(LATENT, map_location="cpu", weights_only=False)
    z = ckpt["z"].to(DEVICE)
    meta = ckpt.get("meta", {})
    fps = int(meta.get("fps", 24))
    print(f"\n  meta   : {meta}")
    stats("z (decode 收到的原始输入)", z)
    lm = torch.tensor(ckpt.get("latents_mean") or [0.0] * z.shape[1],
                      device=DEVICE).view(1, -1, 1, 1, 1)
    ls = torch.tensor(ckpt.get("latents_std") or [1.0] * z.shape[1],
                      device=DEVICE).view(1, -1, 1, 1, 1)
    stats("z * std + mean (反归一化)", z * ls + lm)

    print("\n📦 loading VAE (fp32, ~10.4GB, 约 2-3 分钟)...")
    t0 = time.time()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(MODEL_PATH, subfolder="vae")
    vae.to(DEVICE)
    vae.eval()
    print(f"  ✅ VAE ready in {time.time() - t0:.1f}s  "
          f"(tiling={vae.use_tiling} slicing={vae.use_slicing})")

    print("\n🔄 decoding...")
    t0 = time.time()
    with torch.no_grad():
        sample = vae.decode(z).sample
    print(f"  ✅ decoded in {(time.time() - t0) / 60:.1f} min")
    stats("decoded sample", sample)

    # 两种像素约定各存一份（和 09a 一致，避免猜错去归一化方向）：
    #   standard: sample/2 + 0.5（imagenet 风格反归一）
    #   direct : sample 直接 clamp（若 decode 输出已是 [0,1]）
    variants = {
        "standard": sample / 2 + 0.5,
        "direct": sample,
    }
    for tag, pix in variants.items():
        path = os.path.join(out_dir, f"redecode_{tag}.mp4")
        encode_video(to_uint8(pix[0]), fps=fps, output_path=path)
        print(f"  💾 {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")

    print(f"\n{'=' * 68}")
    print("👉 判读（和 09b 钩子打印的 z 统计对照）：")
    print("  · z 的 std≈1、|max|<10、无 NaN，且 redecode_*.mp4 干净")
    print("      → transformer 没问题；若 06c/06d 出马赛克，根因在它们的加载配方")
    print("  · z 本身幅值爆炸 / 含 NaN")
    print("      → transformer 侧坏了（int8 量化 / offload / scheduler）")
    print("  · z 正常但 redecode_*.mp4 仍是马赛克")
    print("      → VAE 解码路径问题（09a 却是干净的 → 查 pipeline 给 VAE 的输入约定）")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
