#!/usr/bin/env python3
"""30_ff_multiview_eval.py — 通用 FF（前馈直出）多视角评测 + Phase 0 数据登记。

D1（29_d1_diag_3view.py）只支持固定 3 视角且绑 FT 口径；本脚本面向任意视角数：
  - 输入视角自一致性：逐视角 PSNR / SSIM / LPIPS / 锐度 / 饱和度 / 前景 PSNR
  - 覆盖：全图/前景/背景 alpha 空洞率（与 D1 同口径的 depth 粗 mask）
  - Phase 0 登记：逐视角方位角/仰角/半径、相邻视角角距（不得写成“均匀环”）
  - 汇总图：前 min(N,12) 视角 render|GT 拼接 + 逐视角 PSNR 柱状图

Env:
  CAPTURE_DIR  采集包（images/，GT 来源；view_indices 存在时按索引取帧）
  FF_DIR       08 导出目录（gs_max/0000.ply + cameras.npz，正式口径：不裁剪+全 SH）
  OUT_DIR      输出目录
  EVAL_W/H     统一评测分辨率（默认 378×504，与 D1 同口径）
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

EVAL_W = int(os.environ.get("EVAL_W", "378"))
EVAL_H = int(os.environ.get("EVAL_H", "504"))


def log(m):
    print(m, flush=True)


def read_ply(path):
    """手工解析 binary_little_endian PLY（全部 float 字段，3DGS/DA3 通用）。"""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.read(1)
        lines = header.decode("ascii").splitlines()
        names = [ln.split()[-1] for ln in lines if ln.startswith("property float")]
        n = int([ln for ln in lines if ln.startswith("element vertex")][0].split()[-1])
        data = np.frombuffer(f.read(n * 4 * len(names)), dtype="<f4").reshape(n, len(names))
    return {nm: data[:, i].copy() for i, nm in enumerate(names)}


def gsplat_render(d, viewmats, Ks, w, h, sh_degree):
    """调用约定对齐 29_d1_diag_3view.py（gsplat 1.5.3）：scales 传 exp 后、opacity 传
    sigmoid 后、colors 形状 (N,K,3)、逐视角光栅化。"""
    from gsplat import rasterization
    n = d["x"].shape[0]
    xyz = np.stack([d["x"], d["y"], d["z"]], -1)
    quats = np.stack([d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"]], -1)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-12
    a = 1.0 / (1.0 + np.exp(-d["opacity"]))  # PLY 存 logit
    dc = np.stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]], -1)  # (N,3)
    k_rest = max(0, (sh_degree + 1) ** 2 - 1)
    if k_rest > 0:
        fr = np.stack([d[f"f_rest_{i}"] for i in range(3 * k_rest)], -1).reshape(n, 3, k_rest)
        colors = np.transpose(np.concatenate([dc[..., None], fr], axis=-1), (0, 2, 1))
    else:
        colors = dc[:, None, :]
    means = torch.from_numpy(xyz).float().cuda()
    quats_t = torch.from_numpy(quats).float().cuda()
    scales_t = torch.exp(torch.from_numpy(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], -1)).float().cuda())
    opa_t = torch.from_numpy(a).float().cuda()
    colors_t = torch.from_numpy(colors).float().cuda()
    rgbs, alphas = [], []
    with torch.no_grad():
        for i in range(len(viewmats)):
            img, alpha, _ = rasterization(
                means, quats_t, scales_t, opa_t, colors_t,
                torch.from_numpy(viewmats[i]).float().cuda()[None],
                torch.from_numpy(Ks[i]).float().cuda()[None],
                w, h, sh_degree=sh_degree,
            )
            rgbs.append(img[0].clamp(0, 1).mul(255).byte().cpu().numpy())
            alphas.append(alpha[0, ..., 0].cpu().numpy())
    return np.stack(rgbs), np.stack(alphas)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def sharpness(img):
    import cv2
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def mean_sat(img, mask):
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return round(float(hsv[..., 1][mask].mean()), 1) if mask.any() else 0.0


def main():
    from PIL import Image

    img_dir = Path(os.path.expanduser(os.environ["CAPTURE_DIR"])) / "images"
    ff_dir = Path(os.path.expanduser(os.environ["FF_DIR"]))
    out = Path(os.path.expanduser(os.environ["OUT_DIR"]))
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    log(f"🖼️ 采集目录 {len(imgs)} 张")

    # ---------- 相机 ----------
    npz = np.load(ff_dir / "cameras.npz")
    ext = np.asarray(npz["extrinsics"])
    N = ext.shape[0]
    if ext.shape[-2:] == (3, 4):
        ext = np.concatenate(
            [ext, np.tile(np.array([[0, 0, 0, 1]], np.float32), (N, 1, 1))], axis=1)
    sf = float(np.asarray(npz["scale_factor"]))
    intr = np.asarray(npz["intrinsics"]).reshape(N, 3, 3)
    vm = ext.copy()
    if sf > 0:
        vm[:, :3, 3] /= sf  # E-01：高斯已乘 sf，相机平移除回
    H_npz, W_npz = np.asarray(npz["depth"]).shape[-2:]
    view_indices = np.asarray(npz["view_indices"]) if "view_indices" in npz.files \
        else np.arange(len(imgs))
    log(f"📷 相机 {N} 个 @ {W_npz}x{H_npz} | sf={sf:.4f} | 抽帧索引 {view_indices.tolist()}")

    # ---------- Phase 0：逐视角方位角/仰角/半径 + 相邻角距 ----------
    centers = np.stack([-vm[i, :3, :3].T @ vm[i, :3, 3] for i in range(N)])  # w2c 反推
    az = np.degrees(np.arctan2(centers[:, 0], centers[:, 2]))
    el = np.degrees(np.arctan2(centers[:, 1], np.linalg.norm(centers[:, [0, 2]], axis=1)))
    radius = np.linalg.norm(centers, axis=1)
    c0 = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    gaps = np.degrees(np.arccos(np.clip(np.sum(c0[1:] * c0[:-1], axis=1), -1, 1)))
    log("🧭 方位角 " + " ".join(f"{a:+.1f}" for a in az))
    log("🧭 相邻角距 " + " ".join(f"{g:.1f}" for g in gaps))

    # ---------- 人物/背景粗 mask（与 D1 同口径） ----------
    dep = np.asarray(npz["depth"]).reshape(N, H_npz, W_npz)
    med = np.median(dep)
    bg_mask_eval = np.array([Image.fromarray(
        (dep[i] > 1.5 * med).astype(np.uint8) * 255).resize((EVAL_W, EVAL_H)) for i in range(N)]) > 127
    fg_mask_eval = ~bg_mask_eval
    log(f"🩹 粗 mask：背景占 {bg_mask_eval.mean() * 100:.1f}%（depth > 1.5×{med:.2f}m）")

    # ---------- 渲染 ----------
    d = read_ply(ff_dir / "gs_max" / "0000.ply")
    k_rest = (len([k for k in d if k.startswith("f_rest_")])) // 3
    sh_degree = int(round((k_rest + 1) ** 0.5)) - 1
    t0 = time.time()
    rgb, alpha = gsplat_render(d, vm, intr, W_npz, H_npz, sh_degree)
    t_rend = time.time() - t0
    log(f"✅ 渲染 {N} 视角 {t_rend:.1f}s（{d['x'].shape[0]:,} 高斯, SH{sh_degree}）")

    alpha_eval = np.array([Image.fromarray((alpha[i] * 255).astype(np.uint8)).resize((EVAL_W, EVAL_H))
                           for i in range(N)]) / 255.0
    hole = alpha_eval < 0.5

    # ---------- LPIPS（alex，首次使用需下载权重） ----------
    lpips_fn = None
    try:
        import lpips as lpips_mod
        lpips_fn = lpips_mod.LPIPS(net="alex", verbose=False).cuda().eval()
        log("📐 LPIPS(alex) 就绪")
    except Exception as e:  # 网络受限时降级，不阻塞主指标
        log(f"⚠️ LPIPS 不可用（{type(e).__name__}），跳过")

    # ---------- 逐视角指标 ----------
    views = {}
    psnrs, psnrs_fg, lpips_v, ssims = [], [], [], []
    for i in range(N):
        gi = int(view_indices[i])
        gt_eval = np.array(Image.fromarray(np.array(Image.open(imgs[gi]).convert("RGB")))
                           .resize((EVAL_W, EVAL_H)))
        rd_eval = np.array(Image.fromarray(rgb[i]).resize((EVAL_W, EVAL_H)))
        v = {
            "src_frame": gi,
            "azimuth_deg": round(float(az[i]), 1),
            "elevation_deg": round(float(el[i]), 1),
            "radius_m": round(float(radius[i]), 2),
            "psnr": round(psnr(rd_eval, gt_eval), 2),
            "sharp_render": round(sharpness(rd_eval), 1),
            "sharp_gt": round(sharpness(gt_eval), 1),
            "sat_render": mean_sat(rd_eval, np.ones((EVAL_H, EVAL_W), bool)),
            "sat_gt": mean_sat(gt_eval, np.ones((EVAL_H, EVAL_W), bool)),
            "hole_pct_all": round(float(hole[i].mean() * 100), 2),
            "hole_pct_fg": round(float((hole[i] & fg_mask_eval[i]).sum() / max(1, fg_mask_eval[i].sum()) * 100), 2),
            "fg_psnr": round(psnr(rd_eval[fg_mask_eval[i]], gt_eval[fg_mask_eval[i]]), 2)
            if fg_mask_eval[i].sum() > 100 else None,
        }
        try:
            from skimage.metrics import structural_similarity as ssim_fn
            v["ssim"] = round(float(ssim_fn(gt_eval, rd_eval, channel_axis=2)), 4)
        except Exception:
            v["ssim"] = None
        if lpips_fn is not None:
            with torch.no_grad():
                a_t = torch.from_numpy(rd_eval).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
                b_t = torch.from_numpy(gt_eval).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
                v["lpips"] = round(float(lpips_fn(a_t, b_t).item()), 4)
        views[f"v{i}"] = v
        psnrs.append(v["psnr"])
        psnrs_fg.append(v["fg_psnr"] if v["fg_psnr"] is not None else np.nan)
        ssims.append(v["ssim"] if v["ssim"] is not None else np.nan)
        lpips_v.append(v.get("lpips", np.nan))
        # 产物（评测留档；图像不入 git，只进 output）
        Image.fromarray(rgb[i]).save(out / f"v{i:02d}_render.png")
        Image.fromarray(gt_eval).save(out / f"v{i:02d}_gt.png")
        pair = np.concatenate([rd_eval, np.full((EVAL_H, 4, 3), 255, np.uint8), gt_eval], axis=1)
        Image.fromarray(pair).save(out / f"v{i:02d}_pair.png")
        if i < 12:
            log(f"  v{i:02d}(帧{gi:02d}, az {v['azimuth_deg']:+.1f}°): PSNR {v['psnr']:.2f} "
                f"FG {v['fg_psnr']} SSIM {v['ssim']} LPIPS {v.get('lpips')} "
                f"锐度 {v['sharp_render']:.0f}/{v['sharp_gt']:.0f}")

    def agg(a):
        a = np.asarray(a, float)
        a = a[~np.isnan(a)]
        return {"mean": round(float(a.mean()), 3), "median": round(float(np.median(a)), 3),
                "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)} if len(a) else None

    report = {
        "scene": Path(os.path.expanduser(os.environ["CAPTURE_DIR"])).name,
        "n_views": N, "eval_size": [EVAL_W, EVAL_H],
        "n_gaussians": int(d["x"].shape[0]),
        "render_seconds": round(t_rend, 2),
        "view_indices": [int(x) for x in view_indices],
        "azimuth_deg": [round(float(a), 1) for a in az],
        "adjacent_gaps_deg": [round(float(g), 1) for g in gaps],
        "pairwise_gap_mean_deg": round(float(gaps.mean()), 1) if len(gaps) else 0.0,
        "pairwise_gap_max_deg": round(float(gaps.max()), 1) if len(gaps) else 0.0,
        "hole_pct_all": round(float(hole.mean() * 100), 2),
        "aggregate": {
            "psnr": agg(psnrs), "fg_psnr": agg(psnrs_fg),
            "ssim": agg(ssims), "lpips": agg(lpips_v),
            "sharp_ratio": round(float(np.nanmean([views[f'v{i}']['sharp_render'] / max(1, views[f'v{i}']['sharp_gt'])
                                                   for i in range(N)])), 3),
        },
        "views": views,
    }
    (out / "multiview_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    # ---------- 汇总图：render|GT 墙 + PSNR 柱状 ----------
    ncol = min(N, 8)
    nrow = (min(N, 12) + ncol - 1) // ncol
    tw, th = 189, 252
    wall = Image.new("RGB", (ncol * (tw * 2 + 6), nrow * (th + 22)), (255, 255, 255))
    from PIL import ImageDraw
    drw = ImageDraw.Draw(wall)
    for i in range(min(N, 12)):
        r, c = divmod(i, ncol)
        x = c * (tw * 2 + 6)
        y = r * (th + 22) + 20
        wall.paste(Image.open(out / f"v{i:02d}_pair.png").resize((tw * 2, th)), (x, y))
        drw.text((x + 4, y - 16), f"v{i} PSNR {views[f'v{i}']['psnr']:.1f}", fill=(0, 0, 0))
    wall.save(out / "summary_wall.png")

    import cv2
    bw, bh = 900, 300
    chart = np.full((bh, bw, 3), 255, np.uint8)
    x0, y0 = 50, bh - 40
    cw = (bw - 80) / N
    for i in range(N):
        hgt = int((psnrs[i] - 5) / 30 * (bh - 80))
        cv2.rectangle(chart, (int(x0 + i * cw) + 2, y0 - max(hgt, 0)),
                      (int(x0 + (i + 1) * cw) - 2, y0), (80, 140, 240), -1)
        cv2.putText(chart, f"{psnrs[i]:.1f}", (int(x0 + i * cw), y0 - max(hgt, 0) - 4),
                    0, 0.35, (0, 0, 0), 1)
        cv2.putText(chart, f"v{i}", (int(x0 + i * cw) + 3, y0 + 14), 0, 0.4, (0, 0, 0), 1)
    cv2.line(chart, (x0, y0), (bw - 20, y0), (0, 0, 0), 1)
    cv2.putText(chart, f"per-view PSNR (mean {np.mean(psnrs):.2f} / min {np.min(psnrs):.2f})",
                (x0, 20), 0, 0.6, (0, 0, 0), 2)
    cv2.imwrite(str(out / "summary_psnr.png"), chart)

    log(f"🎉 汇总：PSNR mean {np.mean(psnrs):.2f} / min {np.min(psnrs):.2f} | "
        f"FG {np.nanmean(psnrs_fg):.2f} | 空洞 {report['hole_pct_all']}% → {out}")


if __name__ == "__main__":
    main()
