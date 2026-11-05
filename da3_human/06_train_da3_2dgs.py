#!/usr/bin/env python3
"""06_train_da3_2dgs.py — 阶段 C 核心：anchor + 有界 offset 双分支 2DGS per-scene 训练器。

方案（EXPERIMENT_PLAN.md 阶段 C）：
  中心 = anchor + R_c2w · (tanh(offset_raw) ⊙ bound_cam)   # 有界 offset，D1 消融入口
  几何分支（1/4 分辨率渲染）：光度(L1+SSIM) + 2DGS depth-distortion + normal-consistency
                            → 只回传 offset/scale/quat（ detach 颜色/不透明度）
  纹理分支（全分辨率渲染）：光度(L1+SSIM) → 只回传 SH/不透明度（detach 几何）

运行环境：vggt_human env（torch 2.5.1+cu121 + nvcc 12.1，gsplat JIT 已验证）。
da3_human env（torch cu130）无 nvcc，gsplat 无法编译，故训练切 vggt_human env。

Env vars:
  ANCHOR_DIR    05 产物目录（anchors.npz + anchor_summary.json）
  DA3_INFER_DIR 02 的 DA3 导出目录（exports/mini_npz/results.npz，取位姿/内参）
  IMAGES_DIR    原图目录（125 jpg，sorted 顺序与 DA3 视角一致）
  OUTPUT_DIR    输出目录
  ITERATIONS=30000  OFFSET_PIX=1.0  OFFSET_DEPTH=0.02   # 0=锁死；大值≈全开(tanh 饱和)
  SH_DEGREE=3  GEO_RES=4  LR_GEO=1e-4  LR_TEX=5e-3  LR_OPAC=5e-3
  LAMBDA_SSIM=0.2  LAMBDA_DIST=0.1  LAMBDA_NORM=0.1  EPS2D=0.3
  PRUNE_OPACITY=0     # 训练后剪枝阈值（0=关；0.01 为 D4 消融档）
  SMOKE=0             # 1=300 iters 冒烟（8 视角，不评测）
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------- 数据加载
def load_scene(anchor_dir: Path, infer_dir: Path, images_dir: Path):
    """返回 anchors(M,3)/rgb/conf/src_view、w2c[125,4,4]、K_full[125,3,3]、图像路径列表。"""
    a = np.load(anchor_dir / "anchors.npz")
    z = np.load(infer_dir / "exports" / "mini_npz" / "results.npz")
    ext = z["extrinsics"]  # [N,3,4] w2c
    bottom = np.tile(np.array([[[0.0, 0.0, 0.0, 1.0]]], np.float32), (len(ext), 1, 1))
    w2c = np.concatenate([ext, bottom], axis=1)
    k504 = z["intrinsics"]  # [N,3,3] @504p
    # 深度 [N,504,378]（H,W）→ 全分辨率 H=1920,W=1440，比例两边一致
    H504, W504 = z["depth"].shape[1], z["depth"].shape[2]
    H_full, W_full = 1920, 1440
    assert abs(H_full / H504 - W_full / W504) < 1e-3, "深度与全分辨率长宽比不一致"
    s = H_full / H504
    K_full = k504 * s
    K_full[:, 2, 2] = 1.0
    imgs = sorted(Path(images_dir).glob("*.jpg")) + sorted(Path(images_dir).glob("*.png"))
    assert len(imgs) == len(w2c), f"图像数 {len(imgs)} != 视角数 {len(w2c)}"
    return (a["xyz"].astype(np.float32), a["rgb"].astype(np.float32),
            a["conf"].astype(np.float32), a["src_view"].astype(np.int64),
            w2c.astype(np.float32), K_full.astype(np.float32), imgs, float(s))


def load_image(path: Path, device):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"❌ 图像读取失败: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).to(device)  # [3,H,W]


# ---------------------------------------------------------------- SSIM（与 gaussian-splatting 同实现，保证口径一致）
def _gaussian_window(size, sigma, device):
    g = torch.exp(-((torch.arange(size, dtype=torch.float32, device=device) - size // 2) ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def ssim(a, b):
    """a,b [3,H,W] in [0,1]，与 gaussian-splatting utils.loss_utils.ssim 同公式（window=11）"""
    C = a.shape[0]
    w = _gaussian_window(11, 1.5, a.device)
    w = w.view(1, 1, 11) * w.view(1, 11, 1)
    w = w.expand(C, 1, 11, 11).contiguous()
    a, b = a.unsqueeze(0), b.unsqueeze(0)
    mu1 = F.conv2d(a, w, padding=5, groups=C)
    mu2 = F.conv2d(b, w, padding=5, groups=C)
    mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(a * a, w, padding=5, groups=C) - mu1s
    s2 = F.conv2d(b * b, w, padding=5, groups=C) - mu2s
    s12 = F.conv2d(a * b, w, padding=5, groups=C) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1s + mu2s + c1) * (s1 + s2 + c2))
    return m.mean()


def photometric_loss(render_rgb, gt, lambda_ssim):
    l1 = (render_rgb - gt).abs().mean()
    return (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - ssim(render_rgb, gt)), l1


# ---------------------------------------------------------------- 模型
class DualBranch2DGS(torch.nn.Module):
    """anchor 固定；offset 有界（src 相机系 tanh 界）；SH + 地板不透明度。"""

    def __init__(self, xyz, rgb, src_view, w2c, K_full, voxel,
                 offset_pix, offset_depth, sh_degree):
        super().__init__()
        M = len(xyz)
        self.sh_degree = sh_degree
        device = torch.device("cuda")
        w2c_t = torch.from_numpy(w2c).to(device)
        c2w_t = torch.linalg.inv(w2c_t)

        self.register_buffer("anchor", torch.from_numpy(xyz).to(device))
        # 每个 anchor 的源视角旋转（c2w）与深度（w2c 第三行点积）
        src_t = torch.from_numpy(src_view).to(device)
        self.register_buffer("src_R", c2w_t[src_t, :3, :3])  # [M,3,3]
        xyz_h = torch.cat([self.anchor, torch.ones(M, 1, device=device)], 1)
        z_cam = (w2c_t[src_t] @ xyz_h[:, :, None])[:, 2, 0]  # [M]
        fx = torch.from_numpy(K_full[:, 0, 0]).to(device)[src_t]
        bp = offset_pix * z_cam / fx            # 横向界：N 像素 × 米/像素
        bd = offset_depth * z_cam               # 沿视线界：深度百分比
        self.register_buffer("bound_cam", torch.stack([bp, bp, bd], 1))  # [M,3]

        self.offset_raw = torch.nn.Parameter(torch.zeros(M, 3, device=device))
        s0 = math.log(voxel)
        self.scales_raw = torch.nn.Parameter(torch.full((M, 3), s0, device=device))
        with torch.no_grad():
            self.scales_raw[:, 2] = math.log(voxel * 0.3)  # 2DGS z 厚度初始薄
        self.quats = torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=device).repeat(M, 1))
        # SH：DC 项由 anchor 颜色初始化，其余 0
        n_sh = (sh_degree + 1) ** 2
        sh = torch.zeros(M, n_sh, 3, device=device)
        with torch.no_grad():
            sh[:, 0] = (torch.from_numpy(rgb).to(device) - 0.5) / SH_C0
        self.sh = torch.nn.Parameter(sh)
        # QuerySplat 地板正则：opacity = sigmoid(raw - 2.0)，初值 α≈0.119
        self.opac_raw = torch.nn.Parameter(torch.zeros(M, device=device))

    def means(self):
        off_cam = torch.tanh(self.offset_raw) * self.bound_cam
        return self.anchor + torch.einsum("mij,mj->mi", self.src_R, off_cam)

    def forward_params(self):
        return (self.means(), F.normalize(self.quats, dim=-1), self.scales_raw.exp(),
                torch.sigmoid(self.opac_raw - 2.0), self.sh)


# ---------------------------------------------------------------- 训练
def main():
    anchor_dir = Path(os.environ.get("ANCHOR_DIR", ""))
    infer_dir = Path(os.environ.get("DA3_INFER_DIR", ""))
    images_dir = Path(os.environ.get("IMAGES_DIR", ""))
    out_dir = Path(os.environ.get("OUTPUT_DIR", ""))
    if not all([str(anchor_dir), str(infer_dir), str(images_dir), str(out_dir)]):
        sys.exit("❌ ANCHOR_DIR / DA3_INFER_DIR / IMAGES_DIR / OUTPUT_DIR 均需经 06 脚本注入")

    iters = int(os.environ.get("ITERATIONS", "30000"))
    smoke = os.environ.get("SMOKE", "0") == "1"
    if smoke:
        iters = 300
    offset_pix = float(os.environ.get("OFFSET_PIX", "1.0"))
    offset_depth = float(os.environ.get("OFFSET_DEPTH", "0.02"))
    sh_degree = int(os.environ.get("SH_DEGREE", "3"))
    geo_res = int(os.environ.get("GEO_RES", "4"))
    lr_geo = float(os.environ.get("LR_GEO", "1e-4"))
    lr_tex = float(os.environ.get("LR_TEX", "5e-3"))
    lr_opac = float(os.environ.get("LR_OPAC", "5e-3"))
    lambda_ssim = float(os.environ.get("LAMBDA_SSIM", "0.2"))
    lambda_dist = float(os.environ.get("LAMBDA_DIST", "0.1"))
    lambda_norm = float(os.environ.get("LAMBDA_NORM", "0.1"))
    eps2d = float(os.environ.get("EPS2D", "0.3"))
    prune = float(os.environ.get("PRUNE_OPACITY", "0"))
    seed = int(os.environ.get("SEED", "0"))
    device = torch.device("cuda")

    torch.manual_seed(seed)
    np.random.seed(seed)

    xyz, rgb, conf, src_view, w2c, K_full, img_paths, _ = load_scene(anchor_dir, infer_dir, images_dir)
    summary = json.loads((anchor_dir / "anchor_summary.json").read_text(encoding="utf-8"))
    voxel = float(summary["voxel"])
    N_views = len(img_paths)
    H, W = 1920, 1440
    print(f"🚀 [06] 双分支 2DGS 训练: {len(xyz):,} anchors × {N_views} 视角")
    print(f"  有界 offset: 横向 ±{offset_pix:g} 像素 · 沿视线 ±{offset_depth:g}×深度 | "
          f"SH{sh_degree} | 几何分支 1/{geo_res}")
    print(f"  lr geo={lr_geo:g} tex={lr_tex:g} opac={lr_opac:g} | "
          f"λ ssim={lambda_ssim} dist={lambda_dist} norm={lambda_norm}")

    from gsplat import rasterization_2dgs

    model = DualBranch2DGS(xyz, rgb, src_view, w2c, K_full, voxel,
                           offset_pix, offset_depth, sh_degree).to(device)
    opt = torch.optim.Adam([
        {"params": [model.offset_raw, model.scales_raw, model.quats], "lr": lr_geo},
        {"params": [model.sh], "lr": lr_tex},
        {"params": [model.opac_raw], "lr": lr_opac},
    ])

    w2c_t = torch.from_numpy(w2c).to(device)
    K_t = torch.from_numpy(K_full).to(device)
    Kq_t = K_t.clone()
    Kq_t[:, :2, :] /= geo_res
    black4 = torch.zeros(1, 3, device=device)  # RGB+ED 模式 backgrounds 只给 RGB，深度通道由 gsplat 内部补零
    black3 = torch.zeros(1, 3, device=device)

    views = list(range(N_views))
    if smoke:
        views = views[:8]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "loss_curve.csv"
    with open(log_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("iter,tex_loss,geo_loss,distortion,normal_consist\n")

    t0 = time.time()
    for it in range(1, iters + 1):
        vi = views[np.random.randint(len(views))]
        gt_full = load_image(img_paths[vi], device)
        gt_geo = F.avg_pool2d(gt_full.unsqueeze(0), geo_res).squeeze(0) if geo_res > 1 else gt_full

        means, quats, scales, opac, sh = model.forward_params()

        # ---- 纹理分支：全分辨率，共享几何梯度（v3 去 detach）
        # v1 曾 detach opac → 冻结 0.119 叠化糊(14.88)；v2 detach means/quats/scales 仍拖 ~3dB
        # v3：双路共享几何梯度，几何质量靠 1/4 分辨率正则（dist/norm）约束
        rend_tex, alpha_tex, _, _, _, _, _ = rasterization_2dgs(
            means, quats, scales, opac, sh,
            w2c_t[vi:vi + 1], K_t[vi:vi + 1], W, H,
            render_mode="RGB", sh_degree=sh_degree, backgrounds=black3,
            eps2d=eps2d, near_plane=0.01, far_plane=100.0, packed=False)
        tex_loss, tex_l1 = photometric_loss(rend_tex[0].permute(2, 0, 1)[:3], gt_full, lambda_ssim)

        # ---- 几何分支：1/4 分辨率，detach 颜色/不透明度 → 只学 offset/scale/法线
        # 颜色用冻结 DC 项（sh_degree=None 要求 [N,3] RGB；颜色归纹理分支学，几何分支只管几何）
        Hq, Wq = H // geo_res, W // geo_res
        colors_geo = (sh[:, 0] * SH_C0 + 0.5).clamp(0, 1).detach()[None]  # [C,N,3] 非 packed 要求带相机维
        rend_geo, alpha_geo, rn_geo, sn_geo, distort, _, _ = rasterization_2dgs(
            means, quats, scales, opac.detach(), colors_geo,
            w2c_t[vi:vi + 1], Kq_t[vi:vi + 1], Wq, Hq,
            render_mode="RGB+ED", sh_degree=None, backgrounds=black4,  # noqa: E501
            eps2d=eps2d, near_plane=0.01, far_plane=100.0, packed=False)
        geo_l1_ssim, geo_l1 = photometric_loss(rend_geo[0, :, :, :3].permute(2, 0, 1), gt_geo, lambda_ssim)
        mask = (alpha_geo[0, :, :, 0] > 0.5)
        dist_loss = (distort[0, :, :, 0] * mask).sum() / mask.sum().clamp_min(1)
        # surf_normals 在 C=1 时不带相机维（gsplat 1.5.3 实测 [H,W,3]），防御两种形状
        sn = sn_geo[0] if sn_geo.dim() == 4 else sn_geo
        cos = (rn_geo[0] * sn).sum(-1)
        norm_loss = ((1.0 - cos) * mask).sum() / mask.sum().clamp_min(1)
        geo_loss = geo_l1_ssim + lambda_dist * dist_loss + lambda_norm * norm_loss

        loss = tex_loss + geo_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % 250 == 0 or it == 1:
            el = time.time() - t0
            print(f"  it {it:6d}/{iters} | tex {tex_loss.item():.4f} (L1 {tex_l1.item():.4f}) | "
                  f"geo {geo_loss.item():.4f} (dist {dist_loss.item():.4f} norm {norm_loss.item():.4f}) | "
                  f"{el / it * 1000:.0f} ms/it")
            with open(log_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(f"{it},{tex_loss.item():.6f},{geo_loss.item():.6f},"
                        f"{dist_loss.item():.6f},{norm_loss.item():.6f}\n")

    print(f"🏋️ 训练完成 {time.time() - t0:.0f}s，开始评测（全分辨率渲染 {len(views)} 视角）")
    with torch.no_grad():
        _o = torch.sigmoid(model.opac_raw - 2.0)
        print(f"  🔍 opacity 终态: mean {_o.mean():.4f} std {_o.std():.4f} "
              f"max {_o.max():.4f}（std≈0 说明没学到，检查 detach）")

    # ---------------------------------------------------------------- 评测（与 03/04 同协议：in-sample 全视角）
    model.eval()
    renders_dir = out_dir / "renders"
    gt_dir = out_dir / "gt"
    renders_dir.mkdir(exist_ok=True)
    gt_dir.mkdir(exist_ok=True)
    import cv2

    with torch.no_grad():
        means, quats, scales, opac, sh = model.forward_params()
        if prune > 0:
            keep = opac >= prune
            print(f"✂️  opacity 剪枝: {int(keep.sum()):,}/{len(opac):,}（阈值 {prune}）")
            means, quats, scales, opac, sh = (means[keep], quats[keep], scales[keep],
                                              opac[keep], sh[keep])
        psnrs, ssims = [], []
        for vi in views:
            rend, _, _, _, _, _, _ = rasterization_2dgs(
                means, quats, scales, opac, sh,
                w2c_t[vi:vi + 1], K_t[vi:vi + 1], W, H,
                render_mode="RGB", sh_degree=sh_degree, backgrounds=black3,
                eps2d=eps2d, near_plane=0.01, far_plane=100.0, packed=False)
            r = rend[0, :, :, :3].permute(2, 0, 1).clamp(0, 1)
            g = load_image(img_paths[vi], device)
            mse = ((r - g) ** 2).mean().item()
            psnrs.append(-10.0 * math.log10(max(mse, 1e-12)))
            ssims.append(ssim(r, g).item())
            arr = (r.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(str(renders_dir / f"{img_paths[vi].stem}.png"), arr)
            garr = (g.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(str(gt_dir / f"{img_paths[vi].stem}.png"), garr)

    # LPIPS：与基线同实现（gaussian-splatting 仓 lpipsPyTorch, net=vgg）
    lpipss = []
    try:
        sys.path.insert(0, os.path.expanduser("~/repos/gaussian-splatting"))
        from lpipsPyTorch import lpips as gsv_lpips  # noqa: E402
        for vi in views:
            r = torch.from_numpy(cv2.imread(str(renders_dir / f"{img_paths[vi].stem}.png"))[:, :, ::-1]
                                 .astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
            g = load_image(img_paths[vi], device)
            lpipss.append(gsv_lpips(r, g, net_type="vgg").item())
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ LPIPS 评测失败（{e}），仅报 PSNR/SSIM")

    metrics = {
        "n_views_eval": len(views), "protocol": "in-sample train views, full res, black bg",
        "PSNR": float(np.mean(psnrs)), "SSIM": float(np.mean(ssims)),
        "LPIPS": float(np.mean(lpipss)) if lpipss else None,
        "iterations": iters, "offset_pix": offset_pix, "offset_depth": offset_depth,
        "sh_degree": sh_degree, "geo_res": geo_res,
        "lr_geo": lr_geo, "lr_tex": lr_tex, "lr_opac": lr_opac,
        "lambda_ssim": lambda_ssim, "lambda_dist": lambda_dist, "lambda_norm": lambda_norm,
        "eps2d": eps2d, "prune_opacity": prune, "seed": seed,
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    # checkpoint（含 anchor 绑定关系，供 D 阶段继续训练/导出）
    np.savez_compressed(out_dir / "checkpoint_final.npz",
                        means=means.detach().cpu().numpy(), quats=quats.detach().cpu().numpy(),
                        scales=scales.detach().cpu().numpy(), opacities=opac.detach().cpu().numpy(),
                        sh=sh.detach().cpu().numpy(), anchor=xyz, src_view=src_view)
    print(f"🎉 Done. PSNR {metrics['PSNR']:.3f} / SSIM {metrics['SSIM']:.4f}"
          + (f" / LPIPS {metrics['LPIPS']:.4f}" if lpipss else ""))
    print(f"   产物: {out_dir}/renders | gt | metrics.json | checkpoint_final.npz | loss_curve.csv")


if __name__ == "__main__":
    main()
