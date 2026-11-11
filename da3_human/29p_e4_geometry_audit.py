#!/usr/bin/env python3
"""29p：E4 几何与遮挡诊断——center 投影 / 跨视角重投影 / 逐视角层归因 / rotation 一致性。

背景：EXPERIMENT_PLAN.md E4 行——区分 center 错位、scale/rotation 不匹配、遮挡融合错误。
前置事实：32v 高斯数 6,096,384 = 32×190,512、3v = 3×190,512，PLY 是逐视角层等块拼接
（本脚本先用 3×3 在框矩阵验证块-视角归属，不成立则报错，不猜）。

四个审计块：
  A. 层归属验证 + 3×3 逐层逐视角在框率矩阵
  B. 逐层 × 逐视角单独渲染（RGB+alpha）：错位人物/幽灵由哪一层贡献 → center 错位 vs 遮挡融合权重
  C. v0 人物高斯（全层、投影进 v0 人物 mask 且深度一致）跨视角重投影：
     v1/v2 在框率 / 人物 mask 命中率 / miss distance（cv2.distanceTransform，像素级）
  D. rotation 一致性：各向异性短轴（世界系）vs 该视角预测深度图法向；scale footprint 分区域统计

判读边界（写进报告）：人物 mask 与法向都来自 DA3 自身预测深度，非独立 GT——
命中率低 ⇒ 几何不一致成立，但"命中"不代表几何正确，只代表自洽。

用法（WSL，querysplat env）：
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate querysplat
    python /mnt/c/code/media_code/da3_human/29p_e4_geometry_audit.py
产物：<OUT>/e4_report.json + 逐层渲染 PNG + 投影散点 PNG + scatter_top/side.png
"""
import json
import os
import time
import types
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("EVAL_W", "378")
os.environ.setdefault("EVAL_H", "504")

mod = types.ModuleType("x")
code = open("/mnt/c/code/media_code/da3_human/29_d1_diag_3view.py").read()
exec(compile(code, "29.py", "exec"), mod.__dict__)

FF = Path("/home/velvet/output/da3_results/佳佳抱枕_3v/d1_ff_export")
CAP = Path("/home/velvet/output/da3_results/_captures/佳佳抱枕_3v/images")
OUT = Path("/home/velvet/output/da3_results/佳佳抱枕_3v/e4_geometry")

W, H = 378, 504
LAYER_COLORS = [(230, 60, 60), (60, 180, 80), (70, 120, 240)]  # L0/L1/L2
N_LAYERS = 3


def project(xyz, vm, K):
    """世界点 (N,3) → 像素坐标与深度。返回 (u, v, z)；z<=0 表示在相机后方。"""
    xc = xyz @ vm[:3, :3].T + vm[:3, 3]
    z = xc[:, 2]
    u = K[0, 0] * xc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
    v = K[1, 1] * xc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    return u, v, z


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- 相机 / GT / mask（口径同 29n） ----------
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
        vm[:3, 3] = vm[:3, 3] / sf
        viewmats[k] = vm
    K = np.asarray(npz["intrinsics"]).astype(np.float64)[0]
    Ks = np.repeat(K[None], 3, 0)

    gts = [np.array(Image.open(p).convert("RGB").resize((W, H)))
           for p in sorted(CAP.glob("*.png"))[:3]]
    dep = np.asarray(npz["depth"]).reshape(3, H, W).astype(np.float64)
    med = np.median(dep)
    fg = np.array([d <= 1.5 * med for d in dep])          # (3,H,W) 人物/前景
    bg = ~fg

    d0 = mod.read_ply(str(FF / "gs_max" / "0000.ply"))
    n = d0["x"].shape[0]
    n_rest = len([k2 for k2 in d0 if k2.startswith("f_rest_")])
    sh_deg = int(round((n_rest // 3 + 1) ** 0.5)) - 1 if n_rest else 0
    B = n // N_LAYERS
    assert n == B * N_LAYERS, f"PLY 点数 {n} 不是 {N_LAYERS} 的整数倍"
    xyz_all = np.stack([d0["x"], d0["y"], d0["z"]], -1)
    print(f"[29p] PLY {n:,} 点 = {N_LAYERS} 层 × {B:,} | SH deg {sh_deg}", flush=True)

    report = {"n_points": int(n), "block_size": int(B), "views": {}}

    # ---------- A. 层归属验证：3×3 在框率矩阵 ----------
    inframe = np.zeros((N_LAYERS, N_LAYERS))
    for k in range(N_LAYERS):
        xyz = xyz_all[k * B:(k + 1) * B]
        for j in range(N_LAYERS):
            u, v, z = project(xyz, viewmats[j], K)
            ok = (z > 0.2) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            inframe[k, j] = ok.mean() * 100
    diag = np.mean([inframe[k, k] for k in range(N_LAYERS)])
    offd = np.mean([inframe[k, j] for k in range(N_LAYERS) for j in range(N_LAYERS) if k != j])
    report["layer_inframe_pct"] = np.round(inframe, 1).tolist()
    print(f"[29p] 在框率矩阵(行=层,列=视角)%:\n{np.round(inframe,1)}", flush=True)
    print(f"[29p] 对角 {diag:.1f}% vs 非对角 {offd:.1f}%", flush=True)
    if diag <= offd:
        print("[29p] ❌ 层-视角对角不显著，切块假设不成立，停止", flush=True)
        return

    # ---------- B. 逐层 × 逐视角渲染 + 人物区贡献份额 ----------
    layers = []
    for k in range(N_LAYERS):
        dl = {key: val[k * B:(k + 1) * B] for key, val in d0.items()}
        layers.append(dl)
    renders = {}
    t0 = time.time()
    for k in range(N_LAYERS):
        for j in range(N_LAYERS):
            r, a = mod.gsplat_render(layers[k], viewmats[j:j + 1], Ks[j:j + 1], W, H,
                                     sh_degree=sh_deg)
            renders[(k, j)] = (r[0], a[0])
            Image.fromarray(r[0]).save(OUT / f"L{k}_at_v{j}_render.png")
            Image.fromarray((a[0] * 255).astype(np.uint8)).save(OUT / f"L{k}_at_v{j}_alpha.png")
    print(f"[29p] 9 组逐层渲染 {time.time() - t0:.1f}s", flush=True)

    for j in range(N_LAYERS):
        vrep = {}
        fg_j = fg[j]
        fg_px = max(1, int(fg_j.sum()))
        for k in range(N_LAYERS):
            r, a = renders[(k, j)]
            vrep[f"L{k}"] = {
                "psnr_vs_gt": round(mod.psnr(r, gts[j]), 2),
                "alpha_mass_fg": round(float(a[fg_j].sum() / fg_px), 3),
                "alpha_mass_bg": round(float(a[bg[j]].sum() / max(1, int(bg[j].sum()))), 3),
                "sharp_whole": round(mod.sharpness(r), 1),
            }
        report["views"][f"v{j}"] = vrep
        print(f"[29p] v{j} 逐层: " + " | ".join(
            f"L{k}: PSNR {vrep[f'L{k}']['psnr_vs_gt']} 人物α量 {vrep[f'L{k}']['alpha_mass_fg']}"
            for k in range(N_LAYERS)), flush=True)

    # ---------- C. v0 人物高斯跨视角重投影 ----------
    dep0 = dep[0]
    sel_all = []       # 每层在 v0  Depth-consistent 的人物高斯索引（全局 idx）
    for k in range(N_LAYERS):
        xyz = xyz_all[k * B:(k + 1) * B]
        u, v, z = project(xyz, viewmats[0], K)
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inb = (z > 0.2) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ui_c = np.clip(ui, 0, W - 1)
        vi_c = np.clip(vi, 0, H - 1)
        on_person = inb & fg[0][vi_c, ui_c]
        zref = dep0[vi_c, ui_c]
        depth_ok = on_person & (z > 0.8 * zref) & (z < 1.25 * zref)
        sel_all.append(np.where(depth_ok)[0] + k * B)
        print(f"[29p] L{k}: v0 人物mask内 {int(on_person.sum()):,}，深度一致 "
              f"{int(depth_ok.sum()):,}（{depth_ok.sum() / B * 100:.1f}% 该层）", flush=True)
    person_idx = np.concatenate(sel_all)
    xyz_p = xyz_all[person_idx]
    print(f"[29p] v0 人物高斯合计 {len(person_idx):,}", flush=True)

    reproj = {}
    for j in (1, 2):
        u, v, z = project(xyz_p, viewmats[j], K)
        inf = (z > 0.2) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui_c = np.clip(np.round(u).astype(int), 0, W - 1)
        vi_c = np.clip(np.round(v).astype(int), 0, H - 1)
        hit = inf & fg[j][vi_c, ui_c]
        # miss distance：到最近人物像素的距离（px）
        dist = cv2.distanceTransform((~fg[j]).astype(np.uint8), cv2.DIST_L2, 3)
        miss_px = float(np.median(dist[vi_c[inf], ui_c[inf]])) if inf.any() else -1
        # 投影质心 vs 人物 mask 质心
        cx_p, cy_p = float(np.mean(u[inf])), float(np.mean(v[inf]))
        ys, xs = np.nonzero(fg[j])
        cx_m, cy_m = float(xs.mean()), float(ys.mean())
        reproj[f"v{j}"] = {
            "inframe_pct": round(float(inf.mean() * 100), 1),
            "person_hit_pct": round(float(hit.sum() / max(1, len(xyz_p)) * 100), 1),
            "miss_dist_median_px": round(miss_px, 1),
            "proj_centroid": [round(cx_p, 1), round(cy_p, 1)],
            "person_mask_centroid": [round(cx_m, 1), round(cy_m, 1)],
            "centroid_shift_px": round(float(np.hypot(cx_p - cx_m, cy_p - cy_m)), 1),
        }
        print(f"[29p] v0人物高斯→v{j}: 在框 {reproj[f'v{j}']['inframe_pct']}% | "
              f"人物命中 {reproj[f'v{j}']['person_hit_pct']}% | miss中位 {miss_px:.1f}px | "
              f"质心位移 {reproj[f'v{j}']['centroid_shift_px']}px", flush=True)
    report["person_reprojection"] = reproj
    report["person_gaussians"] = int(len(person_idx))

    # 人物高斯投影散点（v1/v2 各一张，画在 GT 上）
    for j in (1, 2):
        img = gts[j].copy()
        u, v, z = project(xyz_p, viewmats[j], K)
        inf = (z > 0.2) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        pts = np.round(np.stack([u[inf], v[inf]], -1)).astype(int)
        step = max(1, len(pts) // 20000)
        for x, y in pts[::step]:
            cv2.circle(img, (x, y), 1, (255, 60, 60), -1)
        Image.fromarray(img).save(OUT / f"person_proj_v{j}.png")

    # ---------- C2. 全中心投影散点（每视角，按层着色） ----------
    for j in range(N_LAYERS):
        img = gts[j].copy()
        for k in range(N_LAYERS):
            xyz = xyz_all[k * B:(k + 1) * B]
            u, v, z = project(xyz, viewmats[j], K)
            inf = (z > 0.2) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            pts = np.round(np.stack([u[inf], v[inf]], -1)).astype(int)
            step = max(1, len(pts) // 15000)
            col = LAYER_COLORS[k]
            for x, y in pts[::step]:
                cv2.circle(img, (x, y), 1, col, -1)
        Image.fromarray(img).save(OUT / f"proj_centers_v{j}.png")

    # ---------- C3. 3D 散点（俯视 x-z / 侧视 y-z，按层着色 + 相机位置） ----------
    # 轴向用 0.5%~99.5% 分位裁剪：p99 σ=320px 的离群高斯会把云团压成一条线
    cam_pos = np.stack([-vm[:3, :3].T @ vm[:3, 3] for vm in viewmats])  # w2c 反推中心

    def draw_scatter(name, axes, pts_sel, layer_groups=None, max_radius=8.0):
        # layer_groups=None → pts_sel 按 N_LAYERS 等块着色；否则 [(pts, color)] 显式分组
        # max_radius：只画距相机质心 max_radius m 内的点（离群高斯会把坐标轴拉飞）
        img = np.full((560, 760, 3), 255, np.uint8)
        cen = cam_pos.mean(0)
        near = np.linalg.norm(pts_sel - cen, axis=1) < max_radius
        pts_near = pts_sel[near]
        lo = pts_near[:, [axes[0], axes[1]]].min(0) - 0.2
        hi = pts_near[:, [axes[0], axes[1]]].max(0) + 0.2
        sc = np.minimum(700 / max(hi[0] - lo[0], 1e-3), 500 / max(hi[1] - lo[1], 1e-3))
        groups = ([(pts_sel[k * B:(k + 1) * B], LAYER_COLORS[k])
                   for k in range(N_LAYERS)] if layer_groups is None else layer_groups)
        for p, col in groups:
            p = p[np.linalg.norm(p - cen, axis=1) < max_radius]
            p = p[:, [axes[0], axes[1]]]
            step = max(1, len(p) // 15000)
            for a, b in p[::step]:
                x = int(30 + (a - lo[0]) * sc)
                y = int(530 - (b - lo[1]) * sc)
                if 0 <= x < 760 and 0 <= y < 560:
                    img[y, x] = col
        for cj, c in enumerate(cam_pos):
            x = int(30 + (c[axes[0]] - lo[0]) * sc)
            y = int(530 - (c[axes[1]] - lo[1]) * sc)
            cv2.circle(img, (x, y), 7, (0, 0, 0), -1)
            cv2.putText(img, f"C{cj}", (x + 9, y + 4), 0, 0.6, (0, 0, 0), 2)
        Image.fromarray(img).save(OUT / f"scatter_{name}.png")

    draw_scatter("top", (0, 2), xyz_all)
    draw_scatter("side", (1, 2), xyz_all)

    # 人物高斯（v0 选集）3D 散点：三层人物云团是否重合 → center 错位直接可视化
    person_groups = [(xyz_all[idx], LAYER_COLORS[k]) for k, idx in enumerate(sel_all)]
    draw_scatter("person_top", (0, 2), xyz_p, layer_groups=person_groups)
    draw_scatter("person_side", (1, 2), xyz_p, layer_groups=person_groups)

    # ---------- D. rotation 一致性（短轴 vs 预测深度法向）+ scale footprint ----------
    # 法向：由深度图重建点云求梯度（相机系），转世界系
    rot_rep = {}
    for k in range(N_LAYERS):
        j = k  # 主要看本层在自身视角
        zmap = dep[j]
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        xc = (us - K[0, 2]) / K[0, 0] * zmap
        yc = (vs - K[1, 2]) / K[1, 1] * zmap
        pc = np.stack([xc, yc, zmap], -1)
        gy, gx = np.gradient(pc, axis=(0, 1))
        n_c = np.cross(gx.reshape(-1, 3), gy.reshape(-1, 3))
        nl = np.linalg.norm(n_c, axis=-1, keepdims=True) + 1e-9
        n_c = (n_c / nl).reshape(H, W, 3)
        n_w = n_c @ viewmats[j][:3, :3]  # 相机系→世界系（旋转正交）
        # 各高斯短轴（世界系）
        q = np.stack([layers[k]["rot_0"], layers[k]["rot_1"],
                      layers[k]["rot_2"], layers[k]["rot_3"]], -1)
        q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
        s = np.exp(np.stack([layers[k]["scale_0"], layers[k]["scale_1"],
                             layers[k]["scale_2"]], -1))
        amin = s.argmin(-1)
        # 短轴方向 = R(q) @ e_amin（取绝对值一致性，cos 用 |.|）
        x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        # 用四元数旋转基矢 e0/e1/e2
        def rot_axis(e):
            # v = q * e * q^-1，e=(1,0,0) 等；返回 (B,3)
            ex, ey, ez = e
            tx = 2 * (y_ * ez - z_ * ey)
            ty = 2 * (z_ * ex - x_ * ez)
            tz = 2 * (x_ * ey - y_ * ex)
            rx = ex + w_ * tx + (y_ * tz - z_ * ty)
            ry = ey + w_ * ty + (z_ * tx - x_ * tz)
            rz = ez + w_ * tz + (x_ * ty - y_ * tx)
            return np.stack([rx, ry, rz], -1)
        axes = np.stack([rot_axis((1, 0, 0)), rot_axis((0, 1, 0)),
                         rot_axis((0, 0, 1))])          # (3,B,3)
        b_idx = np.arange(B)
        short_axis = axes[amin, b_idx]                   # (B,3)
        short_axis /= (np.linalg.norm(short_axis, axis=-1, keepdims=True) + 1e-12)
        # 只统计投影在帧内且深度一致的高斯
        xyz = xyz_all[k * B:(k + 1) * B]
        u, v, z = project(xyz, viewmats[j], K)
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inb = (z > 0.2) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ui_c, vi_c = np.clip(ui, 0, W - 1), np.clip(vi, 0, H - 1)
        zref = dep[j][vi_c, ui_c]
        ok = inb & (z > 0.8 * zref) & (z < 1.25 * zref)
        cos = np.abs((short_axis[ok] * n_w[vi_c[ok], ui_c[ok]]).sum(-1))
        rot_rep[f"L{k}"] = {
            "n_valid": int(ok.sum()),
            "cos_median": round(float(np.median(cos)), 3),
            "cos_frac_ge0.7_pct": round(float((cos >= 0.7).mean() * 100), 1),
        }
        print(f"[29p] L{k} rotation 一致性(本视角): 有效 {int(ok.sum()):,} | "
              f"|cos| 中位 {np.median(cos):.3f} | ≥0.7 占 {(cos >= 0.7).mean() * 100:.1f}%",
              flush=True)
    report["rotation_consistency"] = rot_rep

    # scale footprint 分区域（人物/背景 × 层）：σ 折像素中位
    foot = {}
    for k in range(N_LAYERS):
        xyz = xyz_all[k * B:(k + 1) * B]
        s_max = np.exp(np.stack([layers[k]["scale_0"], layers[k]["scale_1"],
                                 layers[k]["scale_2"]], -1)).max(-1)
        for j in range(N_LAYERS):
            u, v, z = project(xyz, viewmats[j], K)
            ui, vi = np.round(u).astype(int), np.round(v).astype(int)
            inb = (z > 0.2) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            ui_c, vi_c = np.clip(ui, 0, W - 1), np.clip(vi, 0, H - 1)
            sig = K[0, 0] * s_max / np.maximum(z, 1e-3)
            on_fg = inb & fg[j][vi_c, ui_c]
            foot[f"L{k}_v{j}"] = {
                "sigma_px_median_all": round(float(np.median(sig[inb])), 2),
                "sigma_px_median_fg": round(float(np.median(sig[on_fg])), 2) if on_fg.any() else -1,
            }
    report["scale_footprint"] = foot

    (OUT / "e4_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[29p] -> {OUT / 'e4_report.json'}", flush=True)


if __name__ == "__main__":
    main()
