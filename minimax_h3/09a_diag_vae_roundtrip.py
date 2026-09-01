#!/usr/bin/env python3
"""09a — VAE 编解码自检（不跑去噪，2-3 分钟定位「马赛克平色块」在哪一侧）。

为什么先跑这个：「整齐方格 + 每格纯色 + 格间不连续」是解码端数值爆炸的指纹，
而 MiniMax-H3 的 VAE 默认开着空间 tiling（见 AutoencoderKLMiniMaxH3 类注释：
"spatial tiling is on by default"），tile 各自饱和 → 接缝处断开。
所以先用一张已知图案走一遍 encode→decode，VAE 这一侧是好是坏直接见分晓。

本脚本做三件事：
  1. 合成一段可判别的图案（水平/垂直渐变 + 移动方块），任何空间错乱一眼可见
  2. encode → 打印 latent 逐通道 mean/std，与 config 里的 latents_mean/latents_std 比对
     （这一步验证「归一化到底该乘还是该除、有没有被重复应用」）
  3. 分别用「应用归一化」和「不应用归一化」两种方式 decode，各存一个 mp4
      ——哪个干净，就说明 pipeline 该用哪种约定

Env vars:
  MODEL_PATH  默认 /mnt/d/wheel/minimaxh3_ms
  DEVICE      默认 cuda:0
  NUM_FRAMES  默认 124（必须满足 17*n+5；可选 22/39/56/73/90/107/124）
  WIDTH/HEIGHT 默认 960x544
  OUTPUT_DIR  默认 /mnt/d/output/minimaxh3_rotate_results/diag_vae
  FIRST_FRAME 可选，用真实图片替代合成图案（会做成静止视频）
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/d/wheel/minimaxh3_ms")
DEVICE = os.environ.get("DEVICE", "cuda:0")
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "124"))
WIDTH = int(os.environ.get("WIDTH", "960"))
HEIGHT = int(os.environ.get("HEIGHT", "544"))
FPS = int(os.environ.get("FPS", "24"))
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", "/mnt/d/output/minimaxh3_rotate_results/diag_vae")
FIRST_FRAME = os.environ.get("FIRST_FRAME", "")

# MiniMax-H3 的像素约定是 ImageNet 归一化，值域基数是 [0,1] 而不是 [-1,1]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def stats(name, t):
    """打印张量统计，一眼看出数值是否爆炸。"""
    t = t.detach().float()
    finite = torch.isfinite(t)
    print(
        f"  {name:<26} shape={tuple(t.shape)} "
        f"mean={t[finite].mean():+.4f} std={t[finite].std():.4f} "
        f"min={t[finite].min():+.3f} max={t[finite].max():+.3f} "
        f"nan/inf={int((~finite).sum())}"
    )


def make_pattern(num_frames, height, width, device):
    """合成判别图案：R 水平渐变、G 垂直渐变、B 里一个移动的方块。"""
    ys = torch.linspace(0, 1, height, device=device).view(1, 1, height, 1)
    xs = torch.linspace(0, 1, width, device=device).view(1, 1, 1, width)
    frames = []
    for f in range(num_frames):
        img = torch.zeros(3, height, width, device=device)
        img[0] = xs.expand(1, 1, height, width)          # R: 左右渐变
        img[1] = ys.expand(1, 1, height, width)          # G: 上下渐变
        box = (f * 4) % max(1, width - 64)               # B: 移动方块
        img[2, height // 4:height // 2, box:box + 64] = 1.0
        frames.append(img)
    return torch.stack(frames, dim=1).unsqueeze(0)       # [1,3,F,H,W]


def main():
    if (NUM_FRAMES - 5) % 17 != 0:
        sys.exit(f"❌ NUM_FRAMES={NUM_FRAMES} 不满足 17*n+5（可选 22/39/56/73/90/107/124）")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"{'=' * 64}")
    print("🔬 09a — MiniMax-H3 VAE encode/decode 自检")
    print(f"  model : {MODEL_PATH}")
    print(f"  device: {DEVICE}")
    print(f"  video : {NUM_FRAMES}f {WIDTH}x{HEIGHT}  →  "
          f"{(NUM_FRAMES - 5) // 17 * 5 + 2} latent frames")
    print(f"  output: {OUTPUT_DIR}")
    print(f"{'=' * 64}")

    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.utils.export_utils import encode_video

    print("\n📦 loading VAE (fp32, ~10.4GB)...")
    t0 = time.time()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(MODEL_PATH, subfolder="vae")
    vae.to(DEVICE)
    vae.eval()
    print(f"  ✅ loaded in {time.time() - t0:.0f}s  "
          f"tiling={vae.use_tiling} slicing={vae.use_slicing}")
    print(f"  dtype={next(vae.parameters()).dtype}  "
          f"(diffusers 用 _keep_in_fp32_modules 强制钉在 fp32)")

    mean = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1, 1)

    # ── 输入 ────────────────────────────────────────────────────────────
    if FIRST_FRAME:
        from diffusers.utils import load_image
        import torchvision.transforms.functional as TF
        pil = load_image(FIRST_FRAME).convert("RGB").resize((WIDTH, HEIGHT))
        frame = TF.to_tensor(pil).to(DEVICE).unsqueeze(1)          # [3,1,H,W]
        pixel = frame.unsqueeze(0).repeat(1, 1, NUM_FRAMES, 1, 1)  # 静止视频
        print(f"\n🖼️  输入: {FIRST_FRAME}（静止重复 {NUM_FRAMES} 帧）")
    else:
        pixel = make_pattern(NUM_FRAMES, HEIGHT, WIDTH, DEVICE)
        print(f"\n🖼️  输入: 合成图案（R 横渐变 / G 纵渐变 / B 移动方块）")
    stats("input pixel [0,1]", pixel)

    x = (pixel - mean) / std
    stats("input (imagenet-normalized)", x)

    # ── encode ──────────────────────────────────────────────────────────
    print("\n🔄 encoding...")
    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        raw = posterior.mode()                    # 确定性取 mode，排除采样噪声
    stats("raw latent (before norm)", raw)

    lm = torch.tensor(vae.config.latents_mean, device=DEVICE).view(1, -1, 1, 1, 1)
    ls = torch.tensor(vae.config.latents_std, device=DEVICE).view(1, -1, 1, 1, 1)

    # 关键判据：raw latent 的逐通道统计应当接近 config 里的 latents_mean/std。
    # 若吻合 → 归一化约定确认无误；若差一个数量级 → pipeline 的归一化方向反了或被重复应用。
    got_mean = raw.float().mean(dim=[0, 2, 3, 4])
    got_std = raw.float().std(dim=[0, 2, 3, 4])
    print("\n  📊 逐通道比对（前 8 通道）:  raw  vs  config")
    print(f"  {'ch':<4}{'raw_mean':>10}{'cfg_mean':>10}{'raw_std':>10}{'cfg_std':>10}")
    for c in range(min(8, raw.shape[1])):
        print(f"  {c:<4}{got_mean[c]:>10.3f}{lm.view(-1)[c]:>10.3f}"
              f"{got_std[c]:>10.3f}{ls.view(-1)[c]:>10.3f}")
    mean_cos = torch.nn.functional.cosine_similarity(
        got_mean, lm.view(-1), dim=0).item()
    std_cos = torch.nn.functional.cosine_similarity(
        got_std, ls.view(-1), dim=0).item()
    print(f"\n  cosine 相似度: mean={mean_cos:+.3f}  std={std_cos:+.3f}")
    print("  → 两者都应接近 +1.0。若偏离很大，说明归一化约定对不上，")
    print("    这就是马赛克的来源（latent 幅值爆炸 → tile 各自饱和）。")

    lat_norm = (raw - lm) / ls
    stats("normalized latent", lat_norm)

    # ── decode 两种约定 ─────────────────────────────────────────────────
    print("\n🔄 decoding (两种归一化约定各存一份)...")
    from diffusers.utils.export_utils import export_to_video, encode_video  # noqa: F401

    def pt_to_numpy(t):
        """tensor [0,1] float -> numpy uint8（内联实现，不依赖 diffusers 版本）"""
        return (t * 255).round().to(torch.uint8).cpu().numpy()

    for tag, z in (("norm", lat_norm), ("raw", raw)):
        with torch.no_grad():
            dec = vae.decode(z).sample
        stats(f"decode[{tag}] output", dec)
        pix = dec * std + mean
        stats(f"pixel[{tag}] after de-imagenet", pix)
        clamped = pix.clamp(0, 1)
        frac_clip = float((pix < 0).float().mean() + (pix > 1).float().mean())
        print(f"  ⚠️  {tag}: 越界被 clamp 的像素比例 = {frac_clip * 100:.1f}%"
              f"  （>5% 说明幅值不对）")

        frames = pt_to_numpy(clamped.squeeze(0).permute(1, 2, 3, 0).float())
        out = os.path.join(OUTPUT_DIR, f"vae_roundtrip_{tag}.mp4")
        encode_video(frames, fps=FPS, output_path=out)
        print(f"  💾 {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")

    print(f"\n{'=' * 64}")
    print("👉 判读：")
    print("  · vae_roundtrip_norm.mp4 干净  → VAE 与归一化都没问题，锅在 transformer 侧")
    print("  · vae_roundtrip_raw.mp4 干净   → pipeline 的归一化被重复应用了（多做了一次）")
    print("  · 两个都是马赛克               → VAE 权重或 diffusers 的 VAE 实现有问题")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
