#!/usr/bin/env python3
"""Step 5：轨道外外推体检 —— 精调后的高斯在新视角会不会「塌成 sprite / 露空洞」。

**为什么必须单独测**：Step 3 的 +2.11 dB 只在 32 视角环绕内部 ~5.7° 的插值上验证过。
精调的机制是「把多次低不透明度叠加平均压成最近高斯主导」（半径反而变大 1.26×），
这种收敛在**观测覆盖到的方向**上是擦糊，在**没观测到的方向**上完全可能变成薄片：
高斯被压扁去贴合某几个视角，换一个方位就只剩侧面、甚至只剩背面不可见的空壳。

**怎么在无 GT 的情况下判**（外推视角没有真值图，只能比"有 GT 的观测视角"和"外推视角"
在同一套指标上的差异，让偏差互相抵消）：
  cov    覆盖：alpha 均值。外推时掉说明侧面没铺开。
  hole%  空洞：alpha>0.3 的连通区域做 fill_holes 后多出来的那部分 / 目标面积。
         **这是"露空洞"的直接判据** —— 外推时薄片之间露底就是它。
  semi%  半透明率：alpha 落在 (0.15, 0.85) 的像素占比。幽灵/拖影/层叠薄片都会抬高它。
  sharp  边缘锐度：目标区域内亮度梯度均值。sprite 化（贴图被拉大）时它会掉。

**位姿是真造出来的，不是挑出来的**：由观测相机拟合轨道（球心 + 上轴 + 方位/俯仰/半径），
再在球面上按「超出现测范围」的角度和半径生成 look-at 位姿。位姿构造器自带自检：
把每个观测相机按 (az, el, r) 重建一遍，和它的 GT c2w 比旋转/位置误差，报上来自证没搞错约定。

用法:
  python 16_extrap_eval.py --result ~/output/.../jiajia --out ~/output/.../_extrap --tag jiajia
  # 原始 vs 精调对照（推荐）：两份结果用同一批位姿渲染
  python 16_extrap_eval.py --result <原始> --result-b <原始>/ft --out ... --tag jiajia
  python 16_extrap_eval.py --result <原始> --stats-only            # 只看轨道几何
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_check import load_ply, model_preprocess, psnr, render  # noqa: E402


# ---------------------------------------------------------------- 轨道几何
def orbit_fit(c2w):
    """从观测相机的 c2w 拟合「球心 + 上轴 + 半径」，返回 (center, up, radius)。

    球心**必须用视线交汇**求，不能用相机中心均值：手持环绕时人在转、被摄体不动，
    平均相机位置会偏到被摄体一侧（本场景实测 GT 光轴偏离均值点 35°、半径在 0.27~0.83
    之间乱跳）。求 min Σ‖(I−zzᵀ)(c−p)‖² 的闭式解，等价于所有光轴的最近点。
    """
    R, cen = c2w[:, :3, :3], c2w[:, :3, 3]
    z = R[:, :, 2]                                   # 光轴（朝前）
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for zi, pi in zip(z, cen):
        P = np.eye(3) - np.outer(zi, zi)             # 投到与光轴垂直的平面
        A += P
        b += P @ pi
    if np.linalg.cond(A) < 1e6:                      # 光轴近乎互相平行时退化 → 退回质心
        center = np.linalg.solve(A, b)
    else:
        center = cen.mean(0)
        print("   ⚠️ 视线交汇病态（光轴近平行），退回相机中心均值")
    up = -R[:, :, 1].mean(0)                         # OpenCV 相机 +y 朝下 → -y 是世界"上"
    up = up / (np.linalg.norm(up) + 1e-12)
    radius = float(np.median(np.linalg.norm(cen - center, axis=1)))
    return center, up, radius


def orbit_basis(center, up, c2w):
    """(e1, e2) 张成水平面：e1 取第一台相机方向的水平投影，e2 = up × e1。"""
    v = c2w[0, :3, 3] - center
    v = v - up * float(v @ up)
    e1 = v / (np.linalg.norm(v) + 1e-12)
    e2 = np.cross(up, e1)
    return e1, e2


def to_azel(center, up, e1, e2, p):
    v = p - center
    r = float(np.linalg.norm(v))
    u = v / (r + 1e-12)
    el = np.degrees(np.arcsin(np.clip(float(u @ up), -1, 1)))
    az = np.degrees(np.arctan2(float(u @ e2), float(u @ e1))) % 360.0
    return az, el, r


def make_c2w(center, up, e1, e2, az, el, radius):
    """球面 (az, el, radius) → look-at c2w（OpenCV：x右 y下 z前）。

    ⚠️ 两个符号坑都是实测钉死的（拿 32 台观测相机的 GT 对拍才看出来）：
    `x = cross(up, z)` 给出的是 GT x 的**反方向**（roll 差 180°），
    `y = cross(z, x)` 给出的又是世界"上"（图上下一翻）。两处都反，单看渲染"有图"是看不出来的。
    正确写法：x = cross(z, up)，y = cross(z, x) → (x, y, z) 右手、z 朝前、y 朝下。
    """
    a, e = np.radians(az), np.radians(el)
    d = np.cos(e) * (np.cos(a) * e1 + np.sin(a) * e2) + np.sin(e) * up
    p = center + radius * d
    z = center - p
    z = z / (np.linalg.norm(z) + 1e-12)
    x = np.cross(z, up)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    M = np.eye(4, dtype=np.float64)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, p
    return M


def calibrate_pose_builder(c2w, center, up, e1, e2):
    """自检：把每个观测相机按 (az,el,r) 重建一遍，看和 GT 差多少。返回 (rot_deg_p50, pos_rms)。"""
    errs, poss = [], []
    for M in c2w:
        az, el, r = to_azel(center, up, e1, e2, M[:3, 3])
        Mb = make_c2w(center, up, e1, e2, az, el, r)
        cosang = np.clip((np.trace(Mb[:3, :3].T @ M[:3, :3]) - 1) / 2, -1, 1)
        errs.append(np.degrees(np.arccos(cosang)))
        poss.append(np.linalg.norm(Mb[:3, 3] - M[:3, 3]))
    return float(np.median(errs)), float(np.max(errs)), float(np.sqrt(np.mean(np.square(poss))))


# ---------------------------------------------------------------- 指标
def view_metrics(rgb, alpha):
    """GT-free 四项。rgb [H,W,3] / alpha [H,W] 或 [H,W,1]，均为 [0,1]。"""
    a = np.asarray(alpha, np.float32).squeeze()
    rgb = np.asarray(rgb, np.float32).squeeze()
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    m = a > 0.3
    filled = ndimage.binary_fill_holes(m) if m.any() else m
    hole = float((filled & ~m).sum()) / max(float(filled.sum()), 1.0)
    semi = float(((a > 0.15) & (a < 0.85)).mean())
    gy, gx = np.gradient(lum)
    g = np.sqrt(gx ** 2 + gy ** 2)
    sharp = float(g[m].mean()) if m.any() else 0.0
    return dict(cov=float(a.mean()), hole=hole, semi=semi, sharp=sharp,
                area=float(m.mean()))


def label_strip(imgs, labels, pad=4):
    """竖向堆一张带文字标签的拼图（每行一张图，左侧留白写字）。"""
    w, h = imgs[0].size
    canvas = Image.new("RGB", (w + 96, (h + pad) * len(imgs)), (28, 28, 32))
    dr = ImageDraw.Draw(canvas)
    for i, (im, lb) in enumerate(zip(imgs, labels)):
        canvas.paste(im, (96, i * (h + pad)))
        dr.text((4, i * (h + pad) + h // 2 - 4), lb, fill=(230, 230, 230))
    return canvas


def to_u8(x):
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


def alpha_u8(a):
    """alpha [H,W,1] → 三通道灰图（拼图里和 RGB 同尺寸同模式）。"""
    g = (np.clip(np.asarray(a, np.float32).squeeze(), 0, 1) * 255).round().astype(np.uint8)
    return np.repeat(g[:, :, None], 3, axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="含 scene.ply / cameras.npz / result.json")
    ap.add_argument("--result-b", help="第二份结果（通常是 <结果>/ft），用同一批位姿对照")
    ap.add_argument("--images-dir", help="原图目录：给了就算观测视角的 PSNR，用于锚定指标量级")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="extrap")
    ap.add_argument("--n-base", type=int, default=6, help="观测视角基线取样数")
    ap.add_argument("--el-deltas", default="15,30", help="俯仰越界角度（上下各测一遍）")
    ap.add_argument("--r-factors", default="1.3,1.6,0.7", help="半径倍率（>1 拉远 / <1 进入）")
    ap.add_argument("--az-past", default="20,40,60",
                    help="从观测弧两端往未观测区走的角度（扫退化曲线）")
    ap.add_argument("--az-gap-min", type=float, default=40.0,
                    help="方位空洞 ≥ 该角度才测方位外推（环绕 360° 时自动跳过）")
    ap.add_argument("--eps2d", type=float, default=0.01)
    ap.add_argument("--background", type=float, default=1.0)
    ap.add_argument("--stats-only", action="store_true", help="只打印轨道几何，不渲染")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    res = Path(args.result)
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    cams = np.load(res / "cameras.npz", allow_pickle=False)
    c2w, K = np.array(cams["c2w"], np.float64), np.array(cams["intrinsics"], np.float32)
    meta = json.loads((res / "result.json").read_text())
    center, up, rad = orbit_fit(c2w)
    e1, e2 = orbit_basis(center, up, c2w)

    azel = np.array([to_azel(center, up, e1, e2, M[:3, 3]) for M in c2w])
    az_sorted = np.sort(azel[:, 0])
    gaps = np.diff(np.concatenate([az_sorted, [az_sorted[0] + 360.0]]))
    biggest_gap = float(gaps.max())
    r_p50, r_max, pos_rms = calibrate_pose_builder(c2w, center, up, e1, e2)

    print(f"🛰️  [16] 轨道几何 | {len(c2w)} 视角")
    print(f"   球心 {np.round(center, 4).tolist()}  半径 p50 {rad:.4f} (RDNorm)")
    print(f"   俯仰 {azel[:, 1].min():+.1f}° ~ {azel[:, 1].max():+.1f}°"
          f"   半径 {azel[:, 2].min():.3f} ~ {azel[:, 2].max():.3f}")
    print(f"   方位跨度 {360.0 - biggest_gap:.1f}°（最大空洞 {biggest_gap:.1f}°）")
    print(f"   位姿构造器自检: 旋转误差 p50 {r_p50:.3f}° / max {r_max:.3f}°  位置 RMS {pos_rms:.5f}")
    # 阈值 25°：残差不是"构造错了"（那种错一定是 90°/180° 量级），而是手持相机
    # 相对"严格对准球心 + 严格水平"的真实姿态偏差。超过 25° 才说明约定搞错了。
    if r_p50 > 25.0 or pos_rms > 1e-3 * rad:
        print("   ❌ 自检不通过 —— 位姿约定搞错了，外推结论不可信")
    else:
        print("   ✅ 自检通过（残差 = 手持姿态相对理想 look-at 的真实偏差，不是约定错误）")
    if args.stats_only:
        return

    # ---- 基线：均匀取观测视角（用它们的**真实** c2w，避免构造误差污染基线）----
    order = np.argsort(azel[:, 0])
    base_idx = order[np.linspace(0, len(order) - 1, args.n_base).round().astype(int)]
    base_idx = sorted(set(int(i) for i in base_idx))
    wh = tuple(meta["inputs"][0].get("processed_wh", [518, 518]))
    W, H = int(wh[0]), int(wh[1])
    Kmed = np.median(K, axis=0)

    mid_az = float(azel[base_idx[len(base_idx) // 2], 0])
    mid_el = float(np.median(azel[:, 1]))
    els = [float(x) for x in args.el_deltas.split(",") if x.strip()]
    rs = [float(x) for x in args.r_factors.split(",") if x.strip()]

    poses = []          # dict(name, c2w, kind, gt=观测序号|None, group, off=越界量)
    for i in base_idx:
        poses.append(dict(name=f"obs_az{azel[i,0]:.0f}", c2w=c2w[i], kind="obs",
                          gt=int(i), group="obs", off=0.0))
    for d in els:
        poses.append(dict(name=f"el+{d:.0f}", group="el", off=d, kind="extrap", gt=None,
                          c2w=make_c2w(center, up, e1, e2, mid_az,
                                       azel[:, 1].max() + d, rad)))
        poses.append(dict(name=f"el-{d:.0f}", group="el", off=d, kind="extrap", gt=None,
                          c2w=make_c2w(center, up, e1, e2, mid_az,
                                       azel[:, 1].min() - d, rad)))
    for f in rs:
        poses.append(dict(name=f"r{f:.2f}", group="r", off=abs(f - 1.0), kind="extrap", gt=None,
                          c2w=make_c2w(center, up, e1, e2, mid_az, mid_el, rad * f)))

    # 方位越界：从观测弧的**两个端点**往外走，扫一条"走出轨道多远开始坏"的曲线。
    # 比只测空洞中点更有用 —— 中点（158°）是永远没拍过的那一侧，必然无信息，与精调好坏无关。
    gap_lo = float(az_sorted[int(np.argmax(gaps))])
    edges = [gap_lo, (gap_lo + biggest_gap) % 360.0]      # 空洞两端 = 观测弧两端
    past = [float(x) for x in args.az_past.split(",") if x.strip()]
    for d in past:
        for k, e0 in enumerate(edges):
            # 从 edge 走进空洞的方向：低端 +d，高端 -d
            az = (e0 + d) if k == 0 else (e0 - d)
            poses.append(dict(name=f"az{k}+{d:.0f}", group="az", off=d, kind="extrap", gt=None,
                              c2w=make_c2w(center, up, e1, e2, az % 360.0, mid_el, rad)))
    if biggest_gap >= args.az_gap_min:
        gap_mid = (gap_lo + biggest_gap / 2) % 360.0
        poses.append(dict(name=f"azgap{gap_mid:.0f}", group="azgap", off=biggest_gap / 2,
                          kind="extrap", gt=None,
                          c2w=make_c2w(center, up, e1, e2, gap_mid, mid_el, rad)))

    # ---- GT（只在观测视角有）----
    gts = {}
    if args.images_dir:
        from pathlib import Path as _P
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = sorted(p for p in _P(args.images_dir).iterdir()
                       if p.is_file() and p.suffix.lower() in exts)
        for i in base_idx:
            if i < len(files):
                arr, _ = model_preprocess(files[i], W, H)
                gts[int(i)] = arr.astype(np.float32) / 255.0

    models = [("orig", res)]
    if args.result_b:
        models.append(("ft", Path(args.result_b)))

    reports = {}
    for name, d in models:
        g = load_ply(d / "scene.ply")
        print(f"\n🎨 [{name}] {d}/scene.ply | 高斯 {len(g['means']):,}")
        print(f"   {'pose':11s} {'group':6s} {'cov':>6s} {'hole%':>6s} {'semi%':>6s} {'sharp':>7s}  PSNR")
        rows, tiles, labels = [], [], []
        for p in poses:
            rgb, al = render(g, p["c2w"], Kmed, W, H, eps2d=args.eps2d,
                             background=args.background, device=args.device)
            m = view_metrics(rgb[0], al[0])
            if p["gt"] is not None and p["gt"] in gts:
                m["psnr"] = psnr(gts[p["gt"]], rgb[0])
            m.update(name=p["name"], kind=p["kind"], group=p["group"], off=p["off"])
            rows.append(m)
            rgb_u8 = to_u8(rgb[0])
            Image.fromarray(rgb_u8).save(out / f"{name}_{p['name']}.png")
            tiles.append(Image.fromarray(rgb_u8))
            labels.append(p["name"])
            tiles.append(Image.fromarray(alpha_u8(al[0])))
            labels.append(" alpha")
            print(f"   {p['name']:11s} {p['group']:6s} {m['cov']:6.3f} {m['hole']*100:6.1f}"
                  f" {m['semi']*100:6.1f} {m['sharp']*1000:7.2f}"
                  + (f"  {m['psnr']:5.2f}" if "psnr" in m else ""))

        # 按 (group, off) 聚合成曲线点：同一 group 同 off 的多条（方位两边缘）取均值
        curve = {}
        for r in rows:
            if r["group"] == "obs":
                continue
            cell = curve.setdefault(r["group"], {}).setdefault(r["off"], [])
            cell.append(r)
        curve = {gk: {o: {k: float(np.mean([x[k] for x in v]))
                          for k in ("cov", "hole", "semi", "sharp")}
                      for o, v in ov.items()} for gk, ov in curve.items()}
        obs_mean = {k: float(np.mean([r[k] for r in rows if r["group"] == "obs"]))
                    for k in ("cov", "hole", "semi", "sharp", "psnr")}
        reports[name] = dict(result=str(d), rows=rows, obs=obs_mean, curve=curve)
        label_strip(tiles, labels).save(out / f"sheet_{name}.png")

    # ---- 汇总：观测基线 + 各方向越界曲线 ----
    print("\n📊 观测基线（同一批指标，用于判断「外推掉了多少」）")
    print(f"   {'模型':6s} {'覆盖':>7s} {'空洞%':>7s} {'半透明%':>8s} {'锐度×1e3':>9s} {'PSNR':>7s}")
    for name, r in reports.items():
        o = r["obs"]
        print(f"   {name:6s} {o['cov']:7.3f} {o['hole']*100:7.2f} {o['semi']*100:8.2f}"
              f" {o['sharp']*1000:9.2f} {o['psnr']:7.2f}")

    for gk, glabel in (("el", "俯仰越界(度)"), ("az", "方位越界(度)"),
                       ("r", "半径偏离(倍)"), ("azgap", "方位空洞中点")):
        have = [n for n in reports if gk in reports[n]["curve"]]
        if not have:
            continue
        offs = sorted({o for n in have for o in reports[n]["curve"][gk]})
        print(f"\n📈 {glabel}")
        print(f"   {'模型':6s} " + "".join(f"{o:>19.4g}{'':>6s}" for o in offs))
        for name in have:
            cells = []
            for o in offs:
                c = reports[name]["curve"][gk].get(o)
                cells.append(f"{c['cov']:.2f}/{c['semi']*100:4.1f}%" if c else " " * 12)
            print(f"   {name:6s} " + "  ".join(f"{x:>19s}" for x in cells))
        print("   （单元 = 覆盖 / 半透明率；方位两边缘已取均值）")

    if reports.get("orig") and reports.get("ft"):
        print("\n🔍 判据：精调是否在制造视角依赖 —— 比「外推 − 观测」的相对退化，不比绝对值")
        for gk in ("el", "az", "r", "azgap"):
            if gk not in reports["orig"]["curve"]:
                continue
            for o in sorted(reports["orig"]["curve"][gk]):
                a = reports["orig"]["curve"][gk][o]
                b = reports["ft"]["curve"][gk].get(o)
                if not b:
                    continue
                print(f"   {gk}{o:>5.4g}: 覆盖 {b['cov']-a['cov']:+.3f}  "
                      f"空洞 {(b['hole']-a['hole'])*100:+.2f}pp  "
                      f"半透明 {(b['semi']-a['semi'])*100:+.2f}pp  "
                      f"锐度 {(b['sharp']-a['sharp'])*1000:+.2f}（精调−原始）")

    (out / "metrics.json").write_text(json.dumps(dict(
        tag=args.tag, n_views=len(c2w),
        orbit=dict(center=center.tolist(), radius=rad,
                   el=[float(azel[:, 1].min()), float(azel[:, 1].max())],
                   az_span=360.0 - biggest_gap, biggest_az_gap=biggest_gap,
                   pose_calib=dict(rot_p50=r_p50, rot_max=r_max, pos_rms=pos_rms)),
        models=reports), indent=2))
    print(f"\n✅ 输出: {out}/metrics.json, sheet_*.png")


if __name__ == "__main__":
    main()
