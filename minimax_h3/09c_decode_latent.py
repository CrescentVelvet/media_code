#!/usr/bin/env python3
"""09c — 离线解码 09b 存盘的 latent（不重跑 37 分钟去噪）。

起因：run#5 去噪 50/50 全部完成，却因为 09b 钩子里一个 NameError 崩在
VAE 解码前，37 分钟成果全丢。此后 09b 会在每次 vae.decode 前把输入 latent
落盘（latent_decode*.pt）。这个脚本负责把那份 latent 离线解码成 mp4：
即使 09b 的解码/存片环节再出任何问题，去噪成果都能救回来。

2026-09-02 升级：09b 出片仍是马赛克后，本脚本成为「VAE offload 解码」
与「latent 内容」的二分实验：
  · 09b 里 VAE 是 leaf offload + fp32 逐层搬入解码；09a 证明常驻 VAE 对
    偏离分布的 latent 也只输出「低对比度」，不出马赛克（合成图案版实测）。
    → 若本脚本（常驻 VAE）解同一份 latent 是干净的，锅就是 offload 解码路径。
  · 若仍马赛克 → latent 内容本身坏（transformer/量化侧），下一步 09e 对照。

新增三步诊断（decode 前后各一）：
  ① latent 块状检测：空间差分在 tile 边界（960x544 → latent 列 11/22/33/44、
     行 9/18）是否异常跳变 —— 回答「latent 是不是按网格断开的」
  ② 常驻 VAE decode（与 09b 的 offload 解码同输入对比）
  ③ 输出像素的网格性检测：行/列差分在像素 tile 边界
     （列 176/352/528/704、行 144/288）是否异常 —— 回答「马赛克网格
     是否与 VAE tile 对齐」

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

import torch

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/d/wheel/minimaxh3_ms")
LATENT = os.environ.get("LATENT", "")
DEVICE = os.environ.get("DEVICE", "cuda:0")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "")

# 与 09a 相同的像素约定（MiniMax-H3 用 ImageNet 归一化）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def stats(name, t):
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


def tile_boundaries(sample_w, sample_h, compression,
                    tile_px=256, overlap_px=64):
    """复现 AutoencoderKLMiniMaxH3._split_tiles 的 tile 起始索引（像素 & latent）。

    逻辑照抄 autoencoder_kl_minimax_h3.py：num_tiles = ceil(len/tile)，
    不足则 +1；剩余量按 compression 步长摊到 overlap 上。
    """
    def split(length):
        if tile_px >= length:
            return [0]
        n = -(-length // tile_px)  # ceil
        while tile_px * n - overlap_px * (n - 1) - length < 0:
            n += 1
        overlaps = [overlap_px] * (n - 1)
        remaining = tile_px * n - sum(overlaps) - length
        step = 0
        while remaining > 0:
            overlaps[step % (n - 1)] += compression
            remaining -= compression
            step += 1
        starts = [0]
        for i in range(n - 1):
            starts.append(starts[-1] + tile_px - overlaps[i])
        return starts
    return split(sample_w), split(sample_h)


def latent_block_check(z, starts_w_px, starts_h_px, compression):
    """① latent 块状检测：空间差分在 tile 边界处是否异常。

    z: [1,C,T,H,W] raw latent。返回 (打印)——若 latent 内容按 tile 网格
    断开，边界列/行的差分均值会是普通位置的数倍。
    """
    zz = z[0].float()                       # [C,T,H,W]
    # 每条「列边界 i→i+1」的平均差分（对所有通道、帧、行平均）
    dcol = (zz[:, :, :, 1:] - zz[:, :, :, :-1]).abs().mean(dim=(0, 1, 2))
    drow = (zz[:, :, 1:, :] - zz[:, :, :-1, :]).abs().mean(dim=(0, 1, 3))
    bcols = [s // compression for s in starts_w_px if s > 0]
    brows = [s // compression for s in starts_h_px if s > 0]
    print(f"  tile 边界（latent 列）: {bcols}   （latent 行）: {brows}")
    print(f"  全部列边界差分: mean={dcol.mean():.4f} max={dcol.max():.4f}")
    if bcols:
        vals = [dcol[c].item() for c in bcols if c < dcol.numel()]
        base = dcol.median().item()
        print(f"  tile 边界列差分: {[f'{v:.4f}' for v in vals]}"
              f"  （中位普通列={base:.4f}）")
        ratio = max(vals) / max(base, 1e-8)
        print(f"  → 边界/普通 比值 = {ratio:.1f}x"
              + ("  ⚠️ latent 在 tile 边界断裂！" if ratio > 3
                 else "  ✅ latent 无网格断裂"))
    if brows:
        vals = [drow[r].item() for r in brows if r < drow.numel()]
        base = drow.median().item()
        print(f"  tile 边界行差分: {[f'{v:.4f}' for v in vals]}"
              f"  （中位普通行={base:.4f}）")
    return dcol, drow


def pixel_grid_check(sample, starts_w_px, starts_h_px):
    """③ 输出像素网格性检测：解码帧的行/列差分在 tile 边界是否异常。"""
    s = sample[0].float().mean(dim=0)       # [T,H,W] 亮度平均（跨通道）
    dcol = (s[:, :, 1:] - s[:, :, :-1]).abs().mean(dim=(0, 1))   # [W-1]
    drow = (s[:, 1:, :] - s[:, :-1, :]).abs().mean(dim=(0, 2))   # [H-1]
    print(f"  列差分: mean={dcol.mean():.4f}  max={dcol.max():.4f} @列{int(dcol.argmax())}")
    print(f"  行差分: mean={drow.mean():.4f}  max={drow.max():.4f} @行{int(drow.argmax())}")
    for tag, d, starts in (("列", dcol, starts_w_px), ("行", drow, starts_h_px)):
        vals = [d[s_ - 1].item() for s_ in starts if 0 < s_ <= d.numel()]
        base = d.median().item()
        if vals:
            print(f"  tile 边界{tag}差分: {[f'{v:.4f}' for v in vals]}"
                  f"（中位={base:.4f}）")
            ratio = max(vals) / max(base, 1e-8)
            print(f"  → {tag}边界/普通 比值 = {ratio:.1f}x"
                  + ("  ⚠️ 马赛克网格与 VAE tile 对齐！" if ratio > 3
                     else "  ✅ 无 tile 对齐的网格断裂"))


def to_uint8(t):
    """[0,1] float tensor (C,T,H,W) -> (T,H,W,C) uint8 numpy。"""
    x = t.clamp(0, 1)
    x = (x * 255).round().to(torch.uint8)
    return x.permute(1, 2, 3, 0).cpu().numpy()


def main():
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.utils.export_utils import encode_video

    if not LATENT:
        print("❌ 必须设置 LATENT=/path/to/latent_decode*.pt（由 09b 存盘）")
        raise SystemExit(1)
    out_dir = OUTPUT_DIR or os.path.dirname(LATENT)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 68)
    print("🔬 09c — 离线解码 latent（常驻 VAE，对照 09b 的 offload 解码）")
    print(f"  latent : {LATENT}")
    print(f"  model  : {MODEL_PATH}")
    print(f"  device : {DEVICE}")
    print("=" * 68)

    ckpt = torch.load(LATENT, map_location="cpu", weights_only=True)
    z = ckpt["z"].to(DEVICE)
    meta = ckpt.get("meta", {})
    fps = int(meta.get("fps", 24))
    width, height = int(meta.get("width", 960)), int(meta.get("height", 544))
    print(f"\n  meta   : {meta}")
    stats("z (decode 收到的原始输入)", z)
    lm = torch.tensor(ckpt.get("latents_mean") or [0.0] * z.shape[1],
                      device=DEVICE).view(1, -1, 1, 1, 1)
    ls = torch.tensor(ckpt.get("latents_std") or [1.0] * z.shape[1],
                      device=DEVICE).view(1, -1, 1, 1, 1)

    # ── ① latent 块状检测（decode 之前，纯数学）────────────────────────
    print("\n🧩 ① latent 块状检测")
    sw, sh = tile_boundaries(width, height, compression=16)
    print(f"  VAE tile 起始（像素）: 横向 {sw}  纵向 {sh}")
    latent_block_check(z, sw, sh, compression=16)

    # ── ② 常驻 VAE decode ──────────────────────────────────────────────
    print("\n📦 loading VAE (fp32, ~10.4GB, 约 2-3 分钟)...")
    t0 = time.time()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(MODEL_PATH, subfolder="vae")
    vae.to(DEVICE)
    vae.eval()
    print(f"  ✅ VAE ready in {time.time() - t0:.1f}s  "
          f"(tiling={vae.use_tiling} slicing={vae.use_slicing})")

    print("\n🔄 decoding（常驻 fp32，非 offload）...")
    t0 = time.time()
    with torch.no_grad():
        sample = vae.decode(z).sample
    print(f"  ✅ decoded in {(time.time() - t0) / 60:.1f} min")
    stats("decoded sample", sample)

    # ── ③ 输出像素网格性检测 ────────────────────────────────────────────
    print("\n🧩 ③ 输出像素网格性检测（与 VAE tile 对齐性）")
    pixel_grid_check(sample, sw, sh)

    # ── 存片：imagenet 约定（与 09a/09b pipeline 一致）+ direct 对照 ────
    mean = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=DEVICE).view(3, 1, 1, 1)
    variants = {
        "imagenet": sample * std + mean,
        "direct": sample,
    }
    for tag, pix in variants.items():
        path = os.path.join(out_dir, f"redecode_{tag}.mp4")
        encode_video(to_uint8(pix[0]), fps=fps, output_path=path)
        print(f"  💾 {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")

    print(f"\n{'=' * 68}")
    print("👉 判读（核心二分：这份 latent 在「常驻 VAE」下解码是否干净）：")
    print("  · redecode_imagenet.mp4 干净（对比 09b 的马赛克片）")
    print("      → 锅在 09b 的 VAE leaf-offload 解码路径，不在模型/量化。")
    print("        修法：09b 生成结束后、解码前，把 VAE 常驻 GPU（vae.to(DEVICE)），")
    print("        transformer 侧显存此时已释放，24GB 装得下 10.4GB 的 VAE。")
    print("  · redecode 仍是马赛克，且 ① 显示 latent 无网格断裂")
    print("      → latent 高频内容坏（transformer/量化侧）。")
    print("        下一步 09e：transformer bf16 + offload 对照（慢，1-3 小时）。")
    print("  · ① 就显示 latent 在 tile 边界断裂")
    print("      → latent 本身按网格断开 → transformer 侧问题（量化首嫌），")
    print("        同样进 09e。")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
