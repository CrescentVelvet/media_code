#!/usr/bin/env python3
"""AUDIT-8 「仅本视角」为什么糊/重影：几何残差 + 多壳深度离散

两条判据（跑之前定死）：
  G1 自投影残差：view v 的高斯用 camera v 投回图像，与其来源像素的残差
     中位 |Δ| 应 < 0.5 px（构造上应该精确为 0，除浮点与半像素约定）。
     若呈系统性偏移（bias 明显非 0），说明像素中心约定或半像素错位 → 直接造成重影。
  G2 多壳深度离散：同一像素上来自不同视角的高斯，其相机系深度 z 的
     p90-p10 离散；> 1 px 的切向位移即足以产生可辨重影。

只读，不改产物。
"""
import json
import os
import re
import sys
import numpy as np

PLY = os.environ["PLY"]
NPZ = os.environ["NPZ"]
OUT = os.environ.get("OUT", "/tmp")
VIEWS = [int(x) for x in os.environ.get("VIEWS", "0,8,16").split(",")]


def log(*a):
    print(*a, flush=True)


d = np.load(NPZ)
W2C = d["extrinsics"].astype(np.float64)      # (V,4,4) or (V,3,4)
K = d["intrinsics"].astype(np.float64)        # (V,3,3)
DEP = d["depth"].astype(np.float64)           # (V,H,W)
V, H, W = DEP.shape
if W2C.shape[-2] == 4:
    W2C = W2C[:, :3, :]

# ── 复刻 save_gaussian_ply 的屏蔽，拿到每个顶点来自哪个视角、哪个像素 ──
th, tw = int(8 / 256 * H), int(8 / 256 * W)
mask = np.zeros((V, H, W), bool)
mask[:, th:H - th, tw:W - tw] = True
mask &= DEP <= np.quantile(DEP.reshape(V, -1), 0.90, axis=1).reshape(V, 1, 1)

with open(PLY, "rb") as f:
    blob = f.read()
cut = blob.index(b"end_header\n") + len(b"end_header\n")
head = blob[:cut].decode("ascii", "replace")
N = int(re.search(r"element vertex (\d+)", head).group(1))
cnt = mask.reshape(V, -1).sum(1)
if cnt.sum() != N:
    raise SystemExit(f"❌ 顶点数 {N:,} ≠ 复刻屏蔽 {cnt.sum():,}，来源无法复原")
vid = np.repeat(np.arange(V), cnt)
log(f"📦 {N:,} 高斯 / {V} 视角，每视角 {cnt.min()}~{cnt.max()} 个"
    f"（覆盖 {cnt.mean()/(H*W):.1%} 的像素）")

# ── 读 PLY ──
props = re.search(r"element vertex \d+\n((?:property float \S+\n)+)", head)
names = re.findall(r"property float (\S+)", props.group(1))
dt = np.dtype([(n, "f4") for n in names])
raw = np.frombuffer(blob, dtype=dt, count=N, offset=cut)
xyz = np.stack([raw["x"], raw["y"], raw["z"]], 1).astype(np.float64)
op = 1.0 / (1.0 + np.exp(-raw["opacity"].astype(np.float64)))

# 每个顶点对应的源像素（行主序）
src_pix = np.zeros((N, 2), np.int64)
off = 0
for v in range(V):
    hh, ww = np.nonzero(mask[v])
    n = len(hh)
    src_pix[off:off + n, 0] = ww
    src_pix[off:off + n, 1] = hh
    off += n

fx = np.array([K[v, 0, 0] for v in range(V)])
fy = np.array([K[v, 1, 1] for v in range(V)])
cx = np.array([K[v, 0, 2] for v in range(V)])
cy = np.array([K[v, 1, 2] for v in range(V)])


def project(v, pts):
    R, t = W2C[v, :, :3], W2C[v, :, 3]
    p = pts @ R.T + t
    z = p[:, 2]
    zs = np.where(np.abs(z) < 1e-9, np.nan, z)
    u = fx[v] * p[:, 0] / zs + cx[v]
    w = fy[v] * p[:, 1] / zs + cy[v]
    return u, w, z


rep = {"ply": PLY, "npz": NPZ, "n": int(N), "wh": [W, H], "views": {}}

log("\n[G0] 判别：means 是否等价于「深度抬升点」（决定 G1 的 1 px 是头给的还是映射错）")
log(f"  {'view':>5} {'‖Δ‖中位mm':>11} {'‖Δ‖p90mm':>10} {'折合px中位':>10} {'lift残差|Δ|中位':>15}")
for v in VIEWS:
    idx = np.flatnonzero(vid == v)
    hh, ww = np.nonzero(mask[v])
    z = DEP[v][hh, ww]
    # 相机系 → 世界系：Xc = [ (u+.5-cx)/fx*z, (v+.5-cy)/fy*z, z ]，Xw = R^T (Xc - t)
    xc = np.stack([(ww + 0.5 - cx[v]) / fx[v] * z,
                   (hh + 0.5 - cy[v]) / fy[v] * z, z], 1)
    R, t = W2C[v, :, :3], W2C[v, :, 3]
    xw = (xc - t) @ R
    dm = np.linalg.norm(xyz[idx] - xw, axis=1)
    px = np.median(dm) / np.median(z) * fx[v]
    # 同时用同一套抬升点自投回去，检查映射是否自洽（应当≈0）
    u2, w2, _ = project(v, xw)
    r2 = np.hypot(u2 - (ww + 0.5), w2 - (hh + 0.5))
    log(f"  {v:>5} {np.median(dm)*1000:>11.1f} {np.percentile(dm,90)*1000:>10.1f} "
        f"{px:>10.3f} {np.median(r2):>15.4f}")
    rep["views"].setdefault(str(v), {}).update(
        dlift_mm=float(np.median(dm) * 1000),
        dlift_px=float(px),
        lift_self_resid=float(np.median(r2)))

log("\n[G1] 自投影残差（本视角高斯 → 本视角相机 → 来源像素）")
log(f"  {'view':>5} {'n':>8} {'|Δu|中位':>10} {'|Δv|中位':>10} {'p95|Δ|':>9} "
    f"{'bias u':>8} {'bias v':>8} {'去偏后中位':>11}")
for v in VIEWS:
    idx = np.flatnonzero(vid == v)
    u, w, z = project(v, xyz[idx])
    ok = np.isfinite(u) & (z > 0)
    du = u[ok] - (src_pix[idx][ok, 0] + 0.5)
    dv = w[ok] - (src_pix[idx][ok, 1] + 0.5)
    rad = np.hypot(du, dv)
    # 去掉逐视角的整体平移后，看残余是否为纯平移（纯平移 = 等价于极小角度位姿误差）
    duc, dvc = du - np.median(du), dv - np.median(dv)
    radc = np.hypot(duc, dvc)
    # 若约定是像素索引（不加 0.5），残差会整体平移 (0.5,0.5)
    log(f"  {v:>5} {ok.sum():>8} {np.median(np.abs(du)):>10.3f} "
        f"{np.median(np.abs(dv)):>10.3f} {np.percentile(rad, 95):>9.3f} "
        f"{du.mean():>8.3f} {dv.mean():>8.3f} {np.median(radc):>11.3f}")
    rep["views"].setdefault(str(v), {}).update(
        n=int(ok.sum()),
        abs_du_med=float(np.median(np.abs(du))), abs_dv_med=float(np.median(np.abs(dv))),
        rad_med=float(np.median(rad)), rad_p95=float(np.percentile(rad, 95)),
        rad_demeaned_med=float(np.median(radc)),
        bias_u=float(du.mean()), bias_v=float(dv.mean()))

log("\n[G2] 同一像素上的多壳深度离散（全部视角的高斯投到 view v）")
log(f"  {'view':>5} {'覆盖px':>8} {'层数中位':>9} {'层数p90':>8} "
    f"{'Δz中位mm':>10} {'Δz p90mm':>10} {'切向px中位':>11}")
for v in VIEWS:
    u, w, z = project(v, xyz)
    ok = np.isfinite(u) & (z > 0) & (u >= 0) & (u < W) & (w >= 0) & (w < H)
    ui = np.floor(u[ok]).astype(np.int64)
    wi = np.floor(w[ok]).astype(np.int64)
    zz = z[ok]
    pix = wi * W + ui
    order = np.argsort(pix, kind="stable")
    pix, zz = pix[order], zz[order]
    uniq, start = np.unique(pix, return_index=True)
    if len(uniq) == 0:
        log(f"  {v:>5} 无高斯落入画面（跳过）")
        continue
    bounds = np.append(start, len(pix))
    layers = np.diff(bounds)
    # 逐像素深度离散：向量化用 reduceat
    zsum = np.add.reduceat(zz, start)
    zmean = zsum / layers
    zvar = np.add.reduceat((zz - np.repeat(zmean, layers)) ** 2, start) / layers
    zstd = np.sqrt(np.maximum(zvar, 0))
    # p90-p10 用 std*2.56 近似（正态下 p90-p10≈2.56σ）
    spread = zstd * 2.56
    tang = spread / zmean * fx[v]          # 折算为像素切向位移
    log(f"  {v:>5} {len(uniq):>8} {np.median(layers):>9.0f} "
        f"{np.percentile(layers, 90):>8.0f} "
        f"{np.median(spread)*1000:>10.1f} {np.percentile(spread, 90)*1000:>10.1f} "
        f"{np.median(tang):>11.2f}")
    rep["views"][str(v)].update(
        px_covered=int(len(uniq)), layers_med=float(np.median(layers)),
        layers_p90=float(np.percentile(layers, 90)),
        dz_med_mm=float(np.median(spread) * 1000),
        dz_p90_mm=float(np.percentile(spread, 90) * 1000),
        tang_px_med=float(np.median(tang)))

log("\n[G3] 单视角自身的像素覆盖：一个像素平均分到几个高斯")
per_view = N / V / (H * W)
log(f"  每视角 {N/V:,.0f} 高斯 / {H*W:,} 像素 = {per_view:.3f} 个/像素"
    f"  → {'稀疏（有空洞）' if per_view < 1.2 else '基本填满'}")
rep["per_view_per_px"] = float(per_view)
log(f"  opacity 中位 {np.median(op):.4f}  均值 {op.mean():.4f}  >0.5 占比 {(op>0.5).mean():.4%}")
rep["op_median"] = float(np.median(op))
rep["op_gt_half"] = float((op > 0.5).mean())

dst = os.path.join(OUT, "audit_selfview_geom.json")
with open(dst, "w", encoding="utf-8", newline="\n") as f:
    json.dump(rep, f, indent=1, ensure_ascii=False)
log(f"\n💾 {dst}")
