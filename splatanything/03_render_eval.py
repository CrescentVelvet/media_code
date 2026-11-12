#!/usr/bin/env python3
"""用 gsplat 渲染 SplatAnything 导出的高斯，与模型实际输入的 rgb/ 比对（自洽口径）。

为什么必须自接：官方发布包刻意不带光栅化器（`rasterization.py` 注释：
"no rasterization or training"），所以"能不能渲出来"得自己验。

口径说明：
  自洽 re-render 用导出的相机 + 处理后的 rgb/ 图。指标高只说明"高斯属性学对了"，
  因为点位置、相机、GT 图来自同一次前向。要评 NVS 必须另留视角（02 的 INPUT 只给
  训练视角，留出的视角要自己用相机 JSON 加进来）。

坐标约定（v3 比 v2 简单，不需要 Sim(3) 对齐）：
  gaussians.ply 在"导出云坐标系"；cameras.json 的 extrinsics 是同一坐标系下的
  OpenCV c2w（`export_camera_poses`，已过 GSN 缩放），intrinsics 对应 rgb/ 的
  处理分辨率。所以直接 inv(c2w) 就能渲，不要再去拟合 Sim3。

Args:
  --result   02 的输出目录（含 gaussians.ply / cameras.json / rgb/）
  --out      评测输出目录（默认 RESULT/eval）
  --eps2d    屏幕空间膨胀，默认 0.01（论文值 1e-2）
  --max-views 最多评几个视角，0 = 全评
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData


def load_ply(path):
    """读标准 3DGS PLY（SH DC / log scale / logit opacity / wxyz）。"""
    ply = PlyData.read(str(path))['vertex']
    means = np.stack([ply[n] for n in ('x', 'y', 'z')], 1).astype(np.float32)
    scales = np.exp(np.stack([ply[f'scale_{i}'] for i in range(3)], 1).astype(np.float32))
    opacities = 1.0 / (1.0 + np.exp(-ply['opacity'].astype(np.float32)))
    quats = np.stack([ply[f'rot_{i}'] for i in range(4)], 1).astype(np.float32)
    sh = np.stack([ply[f'f_dc_{i}'] for i in range(3)], 1).astype(np.float32)
    return dict(means=means, scales=scales, opacities=opacities, quats=quats, sh=sh)


def load_cameras(path):
    """cameras.json → (c2w[N,4,4], K[N,3,3], ids[N])，按 extrinsics 顺序。"""
    data = json.loads(Path(path).read_text())
    extr = {e['camera_id']: np.asarray(e['matrix'], np.float64) for e in data['extrinsics']}
    intr = {e['camera_id']: np.asarray(e['matrix'], np.float32) for e in data['intrinsics']}
    ids = [e['camera_id'] for e in data['extrinsics'] if e['camera_id'] in intr]
    c2w = np.stack([extr[i] for i in ids])
    K = np.stack([intr[i] for i in ids])
    return c2w, K, ids


def render(g, c2w, K, w, h, sh_degree=0, eps2d=0.01, near=0.01, far=400.0,
           background=0.0, device='cuda'):
    """返回 (rgb[C,H,W,3] float32 in [0,1], alpha[C,H,W])。

    ⚠️ gsplat 1.5.x 的 rasterization 输出**已含** 3DGS 的 +0.5 直流偏置，
    colors 传 f_dc（[C,N,1,3]，sh_degree=0）即可，不要再加 0.5。
    """
    from gsplat import rasterization

    c2w = np.asarray(c2w, np.float64).reshape(-1, 4, 4)
    K = np.asarray(K, np.float32).reshape(-1, 3, 3)
    viewmats = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(device)
    Ks = torch.from_numpy(K).to(device)
    C = viewmats.shape[0]
    colors = torch.from_numpy(g['sh']).to(device)[None, :, None, :].expand(C, -1, -1, -1).contiguous()
    bg = torch.full((C, 3), float(background), device=device)
    out, alpha, _ = rasterization(
        means=torch.from_numpy(g['means']).to(device),
        quats=torch.from_numpy(g['quats']).to(device),
        scales=torch.from_numpy(g['scales']).to(device),
        opacities=torch.from_numpy(g['opacities']).to(device),
        colors=colors, viewmats=viewmats, Ks=Ks, width=w, height=h,
        sh_degree=sh_degree, eps2d=eps2d, near_plane=near, far_plane=far,
        backgrounds=bg, render_mode='RGB')
    return out.clamp(0, 1).cpu().numpy(), alpha.cpu().numpy()


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float('inf') if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def ssim(a, b, sigma=1.5, win=11):
    """标准 SSIM（高斯窗），a/b 为 [H,W,3] float32 in [0,1]。"""
    def kernel(ch):
        c = torch.arange(win, dtype=torch.float32) - win // 2
        g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).view(-1, 1)
        return (g @ g.t()).expand(ch, 1, win, win).contiguous()

    x = torch.from_numpy(np.ascontiguousarray(a, np.float32)).permute(2, 0, 1)[None]
    y = torch.from_numpy(np.ascontiguousarray(b, np.float32)).permute(2, 0, 1)[None]
    ch = x.shape[1]
    k = kernel(ch)
    mu_x = torch.nn.functional.conv2d(x, k, padding=win // 2, groups=ch)
    mu_y = torch.nn.functional.conv2d(y, k, padding=win // 2, groups=ch)
    sx = torch.nn.functional.conv2d(x * x, k, padding=win // 2, groups=ch) - mu_x ** 2
    sy = torch.nn.functional.conv2d(y * y, k, padding=win // 2, groups=ch) - mu_y ** 2
    sxy = torch.nn.functional.conv2d(x * y, k, padding=win // 2, groups=ch) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2))
    return float(s.mean())


def contact_sheet(pairs, out_path, max_items=6):
    """GT / render / |diff|×4 三列拼图。"""
    tiles = []
    for gt, rd in pairs[:max_items]:
        diff = np.clip(np.abs(gt.astype(np.float32) - rd.astype(np.float32)) * 4, 0, 1)
        tiles.append(np.concatenate([gt, rd, (diff * 255).astype(np.uint8)], axis=1))
    if not tiles:
        return
    rows = [np.concatenate(tiles[i:i + 2], axis=0) for i in range(0, len(tiles), 2)]
    wid = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, wid - r.shape[1]), (0, 0))) for r in rows]
    Image.fromarray(np.concatenate(rows, axis=0)).save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--eps2d', type=float, default=0.01)
    ap.add_argument('--max-views', type=int, default=0)
    ap.add_argument('--background', type=float, default=None,
                    help='渲染背景灰度；缺省按 GT 四边像素中位数自动估（GT 黑就黑、白就白）')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    res = Path(args.result)
    out = Path(args.out) if args.out else res / 'eval'
    out.mkdir(parents=True, exist_ok=True)

    ply_path = res / 'gaussians.ply'
    cam_path = res / 'cameras.json'
    rgb_dir = res / 'rgb'
    for p in (ply_path, cam_path):
        if not p.exists():
            raise SystemExit(f'❌ 缺 {p}（先跑 02_run_inference.sh）')
    if not rgb_dir.is_dir():
        raise SystemExit(f'❌ 缺 {rgb_dir}（02 默认导出；用了 NO_DEPTH_RGB=1 就会缺）')

    g = load_ply(ply_path)
    c2w, K, ids = load_cameras(cam_path)
    print(f'🚀 渲染 {len(ids)} 视角 | 高斯 {len(g["means"]):,} | eps2d={args.eps2d}')

    # 用第一张 GT 的尺寸为准（所有视角同 resolution）
    with Image.open(rgb_dir / f'{ids[0]}.png') as im:
        w, h = im.size

    sel = range(len(ids)) if args.max_views <= 0 else range(min(args.max_views, len(ids)))
    (out / 'render').mkdir(exist_ok=True)

    # 背景色必须与 GT 一致，否则 PSNR 会被整片背景吃掉（实测白底 vs 黑底只有 6.75 dB，
    # 而几何和颜色其实都对）。没显式给就从 GT 四边像素的中位数估。
    if args.background is None:
        first = np.asarray(Image.open(rgb_dir / f'{ids[0]}.png').convert('RGB'), np.float32) / 255.0
        edge = np.concatenate([first[0], first[-1], first[:, 0], first[:, -1]], axis=0)
        bg = float(np.median(edge))
    else:
        bg = float(args.background)
    print(f'  背景灰度 {bg:.3f}（{"自动" if args.background is None else "指定"}）')

    rows, pairs = [], []
    for i in sel:
        gt = np.asarray(Image.open(rgb_dir / f'{ids[i]}.png').convert('RGB'), np.float32) / 255.0
        rd, al = render(g, c2w[i:i + 1], K[i:i + 1], w, h, eps2d=args.eps2d,
                        background=bg, device=args.device)
        rd, al = rd[0], al[0]
        Image.fromarray((rd * 255).astype(np.uint8)).save(out / 'render' / f'{ids[i]}.png')
        p, s = psnr(gt, rd), ssim(gt, rd)
        rows.append(dict(id=ids[i], psnr=p, ssim=s,
                         alpha_mean=float(al.mean()),
                         coverage=float((al > 0.01).mean())))
        pairs.append(((gt * 255).astype(np.uint8), (rd * 255).astype(np.uint8)))
        print(f'  {ids[i]:<20s} PSNR {p:6.2f}  SSIM {s:.4f}  alpha {al.mean():.3f}')

    contact_sheet(pairs, out / 'contact_sheet.png')
    summary = dict(result=str(res), views=len(rows), gaussians=int(len(g['means'])),
                   resolution=[w, h], eps2d=args.eps2d, background=bg,
                   psnr_mean=float(np.mean([r['psnr'] for r in rows])),
                   ssim_mean=float(np.mean([r['ssim'] for r in rows])),
                   alpha_mean=float(np.mean([r['alpha_mean'] for r in rows])),
                   views_detail=rows)
    (out / 'eval_summary.json').write_text(json.dumps(summary, indent=2))

    print(f'\n📊 自洽口径：PSNR {summary["psnr_mean"]:.2f}  SSIM {summary["ssim_mean"]:.4f}  '
          f'alpha {summary["alpha_mean"]:.3f}')
    print(f'🎉 Done: {out}')


if __name__ == '__main__':
    main()
