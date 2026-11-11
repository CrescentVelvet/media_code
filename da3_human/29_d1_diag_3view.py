#!/usr/bin/env python3
"""29_d1_diag_3view.py — D1 诊断包渲染与统计（三视角，双口径）。

两个口径并排诊断，分离"前馈直出"与"30k 精化"各自的问题：
  FF = 前馈直出（08_bench_official_gs.py 导出的 PLY + cameras.npz）
  FT = DA3 点云 init + 30k 步 3DGS 精化（04_model_3dgs PLY + cameras.json）

每个口径 × 每个输入视角渲染 RGB + alpha，保存 PNG，统计写 JSON：
  opacity 分布 / σ 折像素 / 空洞率（alpha<0.5，分全图·人物·背景）/
  颜色统计（GT vs 渲染，人物/背景分开）/ PSNR / Laplacian 锐度。
人物/背景粗 mask 共用 FF cameras.npz 里的预测深度（depth>1.5×中位 = 背景），
两口径同 mask 才可比。所有指标在 378×504 统一口径计算（对角线规律，R-3）。

⚠️ 读 PLY 的约定（08 导出与官方 3DGS 一致，audit_11 已核）：
  opacity 列存 logit（渲染前 sigmoid）、scale 列存 log（渲染前 exp）、
  f_rest 按 channel-major 排列（f_rest_{c*15+k} → reshape(N,3,15)）。

Env:
  CAPTURE_DIR  采集转换目录（GT images/，3 张 1439×1919）
  FF_DIR       08 导出目录（gs_max/0000.ply 为正式口径 + cameras.npz）
  FT_DIR       04_model_3dgs 目录（point_cloud/iteration_30000/point_cloud.ply + cameras.json）
  OUT_DIR      诊断输出目录
  EVAL_W/H     统一评测分辨率（默认 378×504）
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
    """d = read_ply 结果；返回 renders (V,H,W,3) uint8、alphas (V,H,W) float32。
    调用约定对齐 compare_4methods/94（gsplat 1.5.3）：scales 传 exp 后、opacity 传
    sigmoid 后、colors 形状 (N,K,3)（K=SH 基个数）、逐视角光栅化。"""
    from gsplat import rasterization
    n = d["x"].shape[0]
    xyz = np.stack([d["x"], d["y"], d["z"]], -1)
    quats = np.stack([d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"]], -1)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-12
    a = 1.0 / (1.0 + np.exp(-d["opacity"]))  # PLY 存 logit（audit_11）
    dc = np.stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]], -1)  # (N,3)
    k_rest = max(0, (sh_degree + 1) ** 2 - 1)
    if k_rest > 0:
        # 08 导出是 channel-major（f_rest_{c*15+k}）→ (N,3,K-1) → 转 (N,K-1,3)
        fr = np.stack([d[f"f_rest_{i}"] for i in range(3 * k_rest)], -1).reshape(n, 3, k_rest)
        colors = np.concatenate([dc[..., None], fr], axis=-1)  # (N,3,K)
        colors = np.transpose(colors, (0, 2, 1))  # (N,K,3)
    else:
        colors = dc[:, None, :]  # (N,1,3)
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


def color_stats(img, mask):
    """mask=True 的区域统计 RGB 均值/方差/饱和度（HSV S 通道均值）。"""
    import cv2
    px = img[mask]
    if px.size == 0:
        return {}
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1]
    return {
        "mean_rgb": [round(float(x), 1) for x in px.mean(0)],
        "std_rgb": [round(float(x), 1) for x in px.std(0)],
        "mean_sat": round(float(hsv[mask].mean()), 1),
    }


def main():
    cap = Path(os.path.expanduser(os.environ["CAPTURE_DIR"]))
    img_dir = cap / "images"
    ff_dir = Path(os.path.expanduser(os.environ["FF_DIR"]))
    ft_dir = Path(os.path.expanduser(os.environ["FT_DIR"]))
    out = Path(os.path.expanduser(os.environ["OUT_DIR"]))
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    gts = [np.array(__import__("PIL.Image", fromlist=["open"]).open(p).convert("RGB")) for p in imgs]
    log(f"🖼️ GT {len(gts)} 张，尺寸 {gts[0].shape[:2]}")

    # ---------- 相机 ----------
    # FF：cameras.npz（extrinsics 为 w2c (N,3,4|4,4)，is_metric 时平移 ÷ scale_factor 回到高斯尺度）
    npz = np.load(ff_dir / "cameras.npz")
    ext = np.asarray(npz["extrinsics"])
    N = ext.shape[0]
    if ext.shape[-2:] == (3, 4):  # DA3 (N,3,4) → 补齐 (N,4,4)
        ext = np.concatenate(
            [ext, np.tile(np.array([[0, 0, 0, 1]], np.float32), (N, 1, 1))], axis=1)
    sf = float(np.asarray(npz["scale_factor"]))
    intr = np.asarray(npz["intrinsics"]).reshape(N, 3, 3)
    vm_ff = ext.copy()
    if sf > 0:
        vm_ff[:, :3, 3] /= sf  # E-01：高斯已乘 sf，相机平移除回
    H_npz, W_npz = np.asarray(npz["depth"]).shape[-2:]
    Ks_ff = intr
    log(f"📷 FF 相机 {N} 个 @ {W_npz}x{H_npz} | sf={sf:.4f}")

    # FT：cameras.json（3DGS 惯例：position + w2c rotation）
    cams = json.loads((ft_dir / "cameras.json").read_text(encoding="utf-8"))
    vm_ft, Ks_ft = [], []
    for c in cams:
        R = np.array(c["rotation"], np.float64)
        t = np.array(c["position"], np.float64)
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = -R @ t
        vm_ft.append(m)
        Ks_ft.append([[c["fx"], 0, c["width"] / 2.0], [0, c["fy"], c["height"] / 2.0], [0, 0, 1]])
    W_ft, H_ft = int(cams[0]["width"]), int(cams[0]["height"])
    log(f"📷 FT 相机 {len(cams)} 个 @ {W_ft}x{H_ft}（原分辨率渲染后降采样评测）")

    # ---------- 人物/背景粗 mask（共用 FF 预测深度） ----------
    dep = np.asarray(npz["depth"]).reshape(N, H_npz, W_npz)
    med = np.median(dep)
    bg_mask_eval = np.array([__import__("PIL.Image", fromlist=["fromarray"]).fromarray(
        (dep[i] > 1.5 * med).astype(np.uint8) * 255).resize((EVAL_W, EVAL_H)) for i in range(N)]) > 127
    fg_mask_eval = ~bg_mask_eval
    log(f"🩹 粗 mask：背景占 {bg_mask_eval.mean() * 100:.1f}%（depth > 1.5×{med:.2f}m）")

    # ---------- 口径定义 ----------
    ft_ply = ft_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    pipelines = [
        ("FF_max", ff_dir / "gs_max" / "0000.ply", np.asarray(vm_ff), np.asarray(Ks_ff), W_npz, H_npz, 2),
        ("FF_official", ff_dir / "gs_official" / "0000.ply", np.asarray(vm_ff), np.asarray(Ks_ff), W_npz, H_npz, 0),
        ("FT_30k", ft_ply, np.asarray(vm_ft), np.asarray(Ks_ft), W_ft, H_ft, 3),
    ]

    from PIL import Image
    report = {"eval_size": [EVAL_W, EVAL_H], "pipelines": {}, "timings": {}}
    for tag, ply, vm, Ks, w, h, sh_deg in pipelines:
        t0 = time.time()
        d = read_ply(ply)
        n_pts = d["x"].shape[0]
        rgb, alpha = gsplat_render(d, vm, Ks, w, h, sh_deg)
        t_rend = time.time() - t0

        # 属性统计（在渲染分辨率口径下）
        a = 1.0 / (1.0 + np.exp(-d["opacity"]))
        s = np.exp(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], -1))
        s_max = s.max(-1)
        # σ 折像素：逐相机取该相机 z 深度中位近似
        sig_px = []
        for i in range(N):
            zc = np.abs((vm[i, :3, :3] @ np.stack([d["x"], d["y"], d["z"]], 0)) + vm[i, :3, 3][:, None])[2]
            zmed = np.median(zc)
            sig_px.append(float(np.median(Ks[i][0, 0] * s_max / np.maximum(zc, 1e-3))))
        sig_px = np.array(sig_px)

        stats = {
            "n_points": int(n_pts), "render_s": round(t_rend, 2),
            "opacity_median": round(float(np.median(a)), 4),
            "opacity_lt_0.1_pct": round(float((a < 0.1).mean() * 100), 1),
            "opacity_gt_0.5_pct": round(float((a > 0.5).mean() * 100), 1),
            "sigma_px_median_per_view": [round(float(x), 3) for x in sig_px],
            "views": {},
        }
        # 空洞率需要 eval 分辨率下的 alpha：降采样
        alpha_eval = np.array([Image.fromarray((alpha[i] * 255).astype(np.uint8)).resize((EVAL_W, EVAL_H)) for i in range(N)]) / 255.0
        hole = alpha_eval < 0.5
        stats["hole_pct_all"] = round(float(hole.mean() * 100), 2)
        stats["hole_pct_bg"] = round(float((hole & bg_mask_eval).sum() / max(1, bg_mask_eval.sum()) * 100), 2)
        stats["hole_pct_fg"] = round(float((hole & fg_mask_eval).sum() / max(1, fg_mask_eval.sum()) * 100), 2)

        for i in range(N):
            gt_eval = np.array(Image.fromarray(gts[i]).resize((EVAL_W, EVAL_H)))
            rd_eval = np.array(Image.fromarray(rgb[i]).resize((EVAL_W, EVAL_H)))
            Image.fromarray(rgb[i]).save(out / f"{tag}_v{i}_render.png")
            Image.fromarray((alpha[i] * 255).astype(np.uint8)).save(out / f"{tag}_v{i}_alpha.png")
            Image.fromarray(np.abs(rd_eval.astype(int) - gt_eval.astype(int)).astype(np.uint8)).save(out / f"{tag}_v{i}_absdiff.png")
            gt_eval_b = None
            stats["views"][f"v{i}"] = {
                "psnr": round(psnr(rd_eval, gt_eval), 2),
                "sharp_render": round(sharpness(rd_eval), 1),
                "sharp_gt": round(sharpness(gt_eval), 1),
                "render_whole": color_stats(rd_eval, np.ones((EVAL_H, EVAL_W), bool)),
                "gt_whole": color_stats(gt_eval, np.ones((EVAL_H, EVAL_W), bool)),
                "render_bg": color_stats(rd_eval, bg_mask_eval[i]),
                "gt_bg": color_stats(gt_eval, bg_mask_eval[i]),
                "render_fg": color_stats(rd_eval, fg_mask_eval[i]),
                "gt_fg": color_stats(gt_eval, fg_mask_eval[i]),
            }
        report["pipelines"][tag] = stats
        log(f"✅ {tag}: {n_pts:,} 点 | 渲染 {t_rend:.1f}s | α中位 {stats['opacity_median']} | "
            f"空洞 全图 {stats['hole_pct_all']}% / 背景 {stats['hole_pct_bg']}% / 人物 {stats['hole_pct_fg']}%")

    (out / "diag_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    log(f"🎉 诊断完成 → {out / 'diag_report.json'}")


if __name__ == "__main__":
    main()
