#!/usr/bin/env python3
"""29n：E3 scale 原始值 vs 受限值 一次性因果对照（EXPERIMENT_PLAN.md E3）。

背景：D1 实测 σ 折像素中位 2.7~4.5 px，糊的一个候选根因是 scale 过大。Phase 1 停止条件
只允许一次 scale 因果对照——本脚本即该对照，规则预先固定、不看结果调参：

  封顶规则：s'_i = min(s_i, SIGMA_TARGET_PXL * z_min / f)   （对每个高斯的三轴分别施加）
    z_min = 该高斯在 3 个输入相机中的最近深度（保证任意输入视角投影 σ 都 ≤ 目标）
    f     = 相机 fx（cameras.npz 内参已是 378×504 口径，勿缩放，ERRATA #10）
    SIGMA_TARGET_PXL = 1.5（基线中位的 ~1/3，预注册常量）

  其余一切不动：center / rotation / opacity / SH / 相机 / 渲染器 / 底色。

验收判据（EXPERIMENT_PLAN E3 行）：v1/v2（远离参考视角）锐度提升，且空洞率与背景颗粒
（背景 Laplacian 方差）不恶化 → scale 受限才可保留；否则糊归因回跨视角几何不一致。

产物：<OUT>/e3_report.json + 逐视角 render/alpha/absdiff PNG + sigma_hist.png（纯条形图，
中文标注由 29o 在 Windows 侧叠加）。

用法（WSL，querysplat env）：
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate querysplat
    python /mnt/c/code/media_code/da3_human/29n_e3_scale_clamp_ablation.py
"""
import json
import os
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("EVAL_W", "378")
os.environ.setdefault("EVAL_H", "504")

# 复用 29_d1_diag 的 read_ply / gsplat_render / psnr / sharpness / color_stats
mod = types.ModuleType("x")
code = open("/mnt/c/code/media_code/da3_human/29_d1_diag_3view.py").read()
exec(compile(code, "29.py", "exec"), mod.__dict__)

FF = Path("/home/velvet/output/da3_results/佳佳抱枕_3v/d1_ff_export")
CAP = Path("/home/velvet/output/da3_results/_captures/佳佳抱枕_3v/images")
OUT = Path("/home/velvet/output/da3_results/佳佳抱枕_3v/e3_scale_clamp")

SIGMA_TARGET_PXL = 1.5  # 预注册：封顶目标投影 σ（px），不看结果调参

W, H = 378, 504


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- 相机（同 29m 口径：extrinsics=w2c 直接用，内参勿缩放） ----------
    npz = np.load(FF / "cameras.npz")
    ext = np.asarray(npz["extrinsics"]).astype(np.float64)
    if ext.ndim == 4:
        ext = ext[:, :3, :]
    if ext.shape[1:] == (3, 4):
        m = np.tile(np.eye(4), (3, 1, 1))
        m[:, :3, :] = ext
        ext = m
    sf = float(np.asarray(npz["scale_factor"]))
    viewmats = np.repeat(np.eye(4)[None], 3, 0)
    for k in range(3):
        vm = ext[k].copy()
        vm[:3, 3] = vm[:3, 3] / sf  # 高斯已乘 sf，相机平移除回（E-01）
        viewmats[k] = vm
    K = np.asarray(npz["intrinsics"]).astype(np.float64)[0]
    Ks = np.repeat(K[None], 3, 0)
    fx = float(K[0, 0])

    gts = [np.array(Image.open(p).convert("RGB").resize((W, H)))
           for p in sorted(CAP.glob("*.png"))[:3]]

    # 人物前景 mask（同 29m：预测深度阈值）
    dep = np.asarray(npz["depth"]).reshape(3, 504, 378)
    med = np.median(dep)
    fg = np.array([np.array(Image.fromarray((d <= 1.5 * med).astype(np.uint8) * 255)
                            .resize((W, H))) > 127 for d in dep])
    bg = ~fg

    # ---------- 载入正式口径 PLY，推断 SH 阶数 ----------
    d0 = mod.read_ply(str(FF / "gs_max" / "0000.ply"))
    n = d0["x"].shape[0]
    n_rest = len([k for k in d0 if k.startswith("f_rest_")])
    sh_deg = int(round((n_rest // 3 + 1) ** 0.5)) - 1 if n_rest else 0
    print(f"[29n] PLY {n:,} 点 | SH degree {sh_deg} | fx={fx:.1f}", flush=True)

    # ---------- 预计算：逐高斯三相机最近深度 z_min 与 σ 统计 ----------
    xyz = np.stack([d0["x"], d0["y"], d0["z"]], 0)  # (3,N)
    z_all = []
    for k in range(3):
        zc = np.abs((viewmats[k, :3, :3] @ xyz) + viewmats[k, :3, 3][:, None])[2]
        z_all.append(zc)
    z_all = np.stack(z_all)          # (3,N)
    z_min = z_all.min(0)             # (N,)

    def sigma_px_per_view(d):
        s = np.exp(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], -1)).max(-1)
        # 逐视角中位（与 D1 同口径：f*s_max/z，z 取逐高斯实际深度）
        out = []
        for k in range(3):
            out.append(float(np.median(fx * s / np.maximum(z_all[k], 1e-3))))
        return out

    # ---------- 受限版：三轴 scale 封顶 ----------
    cap = SIGMA_TARGET_PXL * z_min / fx            # (N,)
    d1 = {k: v.copy() for k, v in d0.items()}
    s_orig = np.exp(np.stack([d0["scale_0"], d0["scale_1"], d0["scale_2"]], -1))
    s_clamp = np.minimum(s_orig, cap[:, None])
    n_clamped = int((s_clamp < s_orig).any(-1).sum())
    for i in range(3):
        d1[f"scale_{i}"] = np.log(np.maximum(s_clamp[:, i], 1e-9)).astype(np.float32)
    print(f"[29n] 封顶: {n_clamped:,}/{n:,} 高斯被钳制 ({n_clamped / n * 100:.1f}%) | "
          f"cap 中位 {np.median(cap) * 1000:.2f} mm", flush=True)

    # ---------- 双臂渲染 + 指标 ----------
    report = {
        "sigma_target_px": SIGMA_TARGET_PXL,
        "clamp_rule": "s'_i = min(s_i, sigma_target * z_min / fx), z_min = min depth over 3 input views",
        "n_points": int(n), "n_clamped": n_clamped,
        "clamp_frac_pct": round(n_clamped / n * 100, 1),
        "arms": {},
    }
    for tag, d in [("base", d0), ("clamp", d1)]:
        t0 = time.time()
        rends, alphas = [], []
        for k in range(3):
            r, a = mod.gsplat_render(d, viewmats[k:k + 1], Ks[k:k + 1], W, H,
                                     sh_degree=sh_deg)
            rends.append(r[0])
            alphas.append(a[0])
        t_rend = time.time() - t0

        arm = {"render_s": round(t_rend, 2),
               "sigma_px_median_per_view": [round(x, 3) for x in sigma_px_per_view(d)],
               "views": {}}
        for k in range(3):
            rd, gt = rends[k], gts[k]
            a = alphas[k]
            hole50 = a < 0.5
            hole05 = a < 0.05
            arm["views"][f"v{k}"] = {
                "psnr": round(mod.psnr(rd, gt), 2),
                "sharp_whole": round(mod.sharpness(rd), 1),
                "sharp_gt": round(mod.sharpness(gt), 1),
                "sharp_fg": round(mod.sharpness(
                    np.where(fg[k][..., None], rd, 0)), 1),
                "sharp_gt_fg": round(mod.sharpness(
                    np.where(fg[k][..., None], gt, 0)), 1),
                # 颗粒哨兵：背景越平滑越好，clamp 后背景 Laplacian 方差上升 = 出颗粒
                "sharp_bg": round(mod.sharpness(
                    np.where(bg[k][..., None], rd, 0)), 1),
                "sharp_gt_bg": round(mod.sharpness(
                    np.where(bg[k][..., None], gt, 0)), 1),
                "hole_pct_all": round(float(hole50.mean() * 100), 2),
                "hole_pct_fg": round(float((hole50 & fg[k]).sum() / max(1, fg[k].sum()) * 100), 2),
                "hole_pct_bg05": round(float((hole05 & bg[k]).sum() / max(1, bg[k].sum()) * 100), 2),
                "alpha_median_fg": round(float(np.median(a[fg[k]])), 3),
                "alpha_mean_fg": round(float(a[fg[k]].mean()), 3),
                "sat_fg_render": mod.color_stats(rd, fg[k]).get("mean_sat"),
                "sat_fg_gt": mod.color_stats(gt, fg[k]).get("mean_sat"),
            }
            Image.fromarray(rd).save(OUT / f"{tag}_v{k}_render.png")
            Image.fromarray((a * 255).astype(np.uint8)).save(OUT / f"{tag}_v{k}_alpha.png")
            Image.fromarray(np.abs(rd.astype(int) - gt.astype(int)).astype(np.uint8)
                            ).save(OUT / f"{tag}_v{k}_absdiff.png")
        report["arms"][tag] = arm
        v = arm["views"]
        print(f"[29n] {tag:5s} 渲染 {t_rend:.1f}s | σ中位 {arm['sigma_px_median_per_view']} | "
              f"PSNR {[v[f'v{k}']['psnr'] for k in range(3)]} | "
              f"锐度(全图) {[v[f'v{k}']['sharp_whole'] for k in range(3)]} | "
              f"空洞(全/人/背) {[(v[f'v{k}']['hole_pct_all'], v[f'v{k}']['hole_pct_fg'], v[f'v{k}']['hole_pct_bg05']) for k in range(3)]}",
              flush=True)

    # ---------- σ 分布直方图（纯条形，中文标注由 29o 叠加） ----------
    s_b = np.exp(np.stack([d0["scale_0"], d0["scale_1"], d0["scale_2"]], -1)).max(-1)
    s_c = np.exp(np.stack([d1["scale_0"], d1["scale_1"], d1["scale_2"]], -1)).max(-1)
    sig_b = fx * s_b / np.maximum(z_min, 1e-3)   # 用 z_min（最坏视角）口径
    sig_c = fx * s_c / np.maximum(z_min, 1e-3)
    bins = np.linspace(0, 12, 49)                # 0~12 px，48 桶（>12 的尾部并入末桶）
    hb, _ = np.histogram(np.clip(sig_b, 0, 12), bins)
    hc, _ = np.histogram(np.clip(sig_c, 0, 12), bins)
    hw, hh = 960, 420
    hist = Image.new("RGB", (hw, hh), (255, 255, 255))
    dr = __import__("PIL.ImageDraw", fromlist=["Draw"]).Draw(hist)
    plot_l, plot_b, plot_t = 50, hh - 40, 20
    plot_w, plot_h = hw - 70, hh - 80
    hmax = max(np.log10(max(hb.max(), 1)), np.log10(max(hc.max(), 1)), 1e-6)
    bw = plot_w / len(hb)
    for i in range(len(hb)):
        x0 = plot_l + i * bw
        for hgt, col in [(hb[i], (150, 150, 150)), (hc[i], (60, 170, 90))]:
            if hgt <= 0:
                continue
            bhgt = plot_h * (np.log10(hgt) / hmax) if hmax > 0 else 0
            dr.rectangle([x0 + 1, plot_b - bhgt, x0 + bw - 1, plot_b], fill=col)
    dr.rectangle([plot_l, plot_t, plot_l + plot_w, plot_b], outline=(0, 0, 0))
    hist.save(OUT / "sigma_hist.png")
    report["sigma_worstview_quantiles_px"] = {
        "base": [round(float(x), 2) for x in np.quantile(sig_b, [0.5, 0.9, 0.99])],
        "clamp": [round(float(x), 2) for x in np.quantile(sig_c, [0.5, 0.9, 0.99])],
    }
    print(f"[29n] σ(worst-view) 分位 base p50/p90/p99 = {report['sigma_worstview_quantiles_px']['base']} | "
          f"clamp = {report['sigma_worstview_quantiles_px']['clamp']}", flush=True)

    (OUT / "e3_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[29n] -> {OUT / 'e3_report.json'}", flush=True)


if __name__ == "__main__":
    main()
