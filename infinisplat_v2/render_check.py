#!/usr/bin/env python3
"""用 gsplat 渲染 InfiniSplat v2 导出的 SH0 高斯场景，并与原图比对。

发布包刻意不带光栅化器（INSTALL.md 说明无 gsplat 依赖），所以渲染必须自接。
本脚本承担两件事：
  1) 自洽检查（re-render）：用 result 目录里导出的相机渲染输入视角，与输入图比。
     指标好看只说明"属性学对了"，因为点位置与相机来自同一套预测/对齐。
  2) NVS 指标（holdout）：用留出视角的 GT 相机渲染，与未参与重建的图比。
     这才是能对标论文表 1 的口径（论文每例 4 个 held-out 目标视角）。

Env vars: 无（全部走命令行参数）。

用法:
  python render_check.py --result <result_dir> --out <out_dir>
  python render_check.py --result <result_dir> --out <out_dir> \
      --images-dir <dir> --cameras <holdout_cameras.npz> --tag holdout
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData


def load_ply(path):
    ply = PlyData.read(str(path))["vertex"]
    names = ply.data.dtype.names
    means = np.stack([ply[n] for n in ("x", "y", "z")], 1).astype(np.float32)
    scales = np.exp(np.stack([ply[f"scale_{i}"] for i in range(3)], 1).astype(np.float32))
    opac = 1.0 / (1.0 + np.exp(-ply["opacity"].astype(np.float32)))
    quats = np.stack([ply[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    sh = np.stack([ply[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32)
    return dict(means=means, scales=scales, opacities=opac, quats=quats, sh=sh, fields=names)


def ssim(a, b, sigma=1.5, win=11):
    """标准 SSIM（高斯窗），a/b 为 [H,W,3] float32 in [0,1]。"""
    def gauss_kernel(ch):
        c = torch.arange(win, dtype=torch.float32) - win // 2
        g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).view(-1, 1)
        return (g @ g.t()).expand(ch, 1, win, win).contiguous()

    x = torch.from_numpy(a).permute(2, 0, 1)[None]
    y = torch.from_numpy(b).permute(2, 0, 1)[None]
    ch = x.shape[1]
    k = gauss_kernel(ch)
    pad = win // 2
    mu_x = torch.nn.functional.conv2d(x, k, padding=pad, groups=ch)
    mu_y = torch.nn.functional.conv2d(y, k, padding=pad, groups=ch)
    sxx = torch.nn.functional.conv2d(x * x, k, padding=pad, groups=ch) - mu_x ** 2
    syy = torch.nn.functional.conv2d(y * y, k, padding=pad, groups=ch) - mu_y ** 2
    sxy = torch.nn.functional.conv2d(x * y, k, padding=pad, groups=ch) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return float(s.mean())


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float("inf") if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def render(g, c2w, K, w, h, sh_degree=0, eps2d=0.01, near=0.01, far=200.0,
           background=1.0, device="cuda"):
    """返回 (rgb[C,H,W,3], alpha[C,H,W])。

    ⚠️ gsplat 1.5.2 的 rasterization 输出**已含** 3DGS 的 `+0.5` 直流偏置
    （实测 f_dc=0 + alpha≈0.98 时像素≈0.49），这里不能再加一次 0.5。
    colors 必须给成 [C,N,K,3]（K=1，对应 sh_degree=0）。
    """
    from gsplat import rasterization

    viewmats = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(device)  # [C,4,4] w2c
    Ks = torch.from_numpy(K.astype(np.float32)).to(device)                          # [C,3,3]
    C = viewmats.shape[0]
    colors = torch.from_numpy(g["sh"]).to(device)[None, :, None, :].expand(C, -1, -1, -1).contiguous()
    bg = torch.full((C, 3), float(background), device=device)
    out, alpha, _ = rasterization(
        means=torch.from_numpy(g["means"]).to(device),
        quats=torch.from_numpy(g["quats"]).to(device),
        scales=torch.from_numpy(g["scales"]).to(device),
        opacities=torch.from_numpy(g["opacities"]).to(device),
        colors=colors, viewmats=viewmats, Ks=Ks, width=w, height=h,
        sh_degree=sh_degree, eps2d=eps2d, near_plane=near, far_plane=far,
        backgrounds=bg, render_mode="RGB")
    return out.clamp(0, 1).cpu().numpy(), alpha.cpu().numpy()


def contact_sheet(pairs, out_path, max_items=6):
    """GT / render / |diff| 三列拼图。"""
    tiles = []
    for gt, rd in pairs[:max_items]:
        diff = np.abs(gt.astype(np.float32) - rd.astype(np.float32))
        diff = np.clip(diff * 4, 0, 1)
        tiles.append(np.concatenate([gt, rd, (diff * 255).astype(np.uint8)], axis=1))
    rows = [np.concatenate(tiles[i:i + 2], axis=0) for i in range(0, len(tiles), 2)]
    wid = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, wid - r.shape[1]), (0, 0))) for r in rows]
    Image.fromarray(np.concatenate(rows, axis=0)).save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="02 输出目录（含 scene.ply / cameras.npz / result.json）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--images-dir", help="原图目录；缺省用 result.json 记录的输入图路径")
    ap.add_argument("--cameras", help="另给相机 npz（holdout 评测用）")
    ap.add_argument("--tag", default="rerender")
    ap.add_argument("--eps2d", type=float, default=0.01, help="论文值 1e-2")
    ap.add_argument("--max-views", type=int, default=32)
    ap.add_argument("--background", type=float, default=1.0, help="渲染背景亮度（1=白，常见 3DGS 评测口径）")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    res = Path(args.result)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((res / "result.json").read_text())
    g = load_ply(res / "scene.ply")
    cams_path = Path(args.cameras) if args.cameras else res / "cameras.npz"
    cams = np.load(cams_path, allow_pickle=False)
    c2w_all, K_all = cams["c2w"], cams["intrinsics"]

    records = meta.get("inputs", [])
    if args.images_dir:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        imgs = sorted(p for p in Path(args.images_dir).iterdir()
                      if p.is_file() and p.suffix.lower() in exts)
    else:
        # 记录里没有绝对路径，仅用于自洽模式：按 result.json 的 sha 找不到图时要求显式给目录
        imgs = []

    n = min(len(c2w_all), args.max_views, len(imgs) if imgs else len(c2w_all))
    if len(imgs) and len(imgs) != len(c2w_all):
        print(f"⚠️  图像数({len(imgs)}) 与相机数({len(c2w_all)}) 不一致，按较小值取前 {n} 个")
    if not imgs:
        raise SystemExit("❌ 需要 --images-dir（用于计算指标与拼图）")

    print(f"🚀 渲染 {n} 个视角 | 高斯 {len(g['means']):,} | eps2d={args.eps2d}")
    ps, ss = [], []
    pairs = []
    for i in range(n):
        rec = records[i] if i < len(records) else {}
        w, h = rec.get("processed_wh", [0, 0])
        if not (w and h):
            with Image.open(imgs[i]) as im:
                w, h = im.size
        rd, al = render(g, c2w_all[i], K_all[i], int(w), int(h), eps2d=args.eps2d,
                        background=args.background, device=args.device)
        rd = rd[0]
        gt = np.asarray(Image.open(imgs[i]).convert("RGB").resize((int(w), int(h)), Image.BICUBIC))
        p, s = psnr(gt, (rd * 255).round()), ssim(gt / 255.0, rd)
        ps.append(p); ss.append(s); pairs.append((gt, (rd * 255).round().astype(np.uint8)))
        Image.fromarray((rd * 255).round().astype(np.uint8)).save(out / f"render_{i:03d}.png")
        cov = float(al[0].mean())
        print(f"  [{i:02d}] {imgs[i].name[:40]:42s} {w}x{h}  PSNR {p:6.3f}  SSIM {s:.4f}  alpha {cov:.3f}")

    summary = dict(tag=args.tag, views=n, gaussians=int(len(g["means"])), eps2d=args.eps2d,
                   psnr_mean=float(np.mean(ps)), ssim_mean=float(np.mean(ss)),
                   per_view=dict(psnr=ps, ssim=ss))
    (out / f"metrics_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    contact_sheet(pairs, out / f"compare_{args.tag}.jpg")
    print(f"✅ mean PSNR {summary['psnr_mean']:.3f} | mean SSIM {summary['ssim_mean']:.4f}")
    print(f"✅ 输出: {out}/metrics_{args.tag}.json, compare_{args.tag}.jpg")


if __name__ == "__main__":
    main()
