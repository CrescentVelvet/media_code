#!/usr/bin/env python3
"""AUDIT-12 DA3 的盘是「朝相机的 billboard」还是「朝真实表面的 surfel」？

这是「把 3DGS 头换成 2DGS 头能带来什么」的前置事实判定。
2DGS 的全部收益（法向一致性正则、深度畸变正则、ray-splat 精确求交）
都建立在「盘的法向有意义、且应当对齐真实表面」这一前提上。
若 DA3 的盘法向只是「朝来源相机」，那么换成 2DGS 就是给它补上原本缺失的法向约束；
若它已经近似对齐表面，2DGS 的法向正则就是白给，收益只能来自求交方式。

判据（跑之前定死，见 P1_EXPERIMENTS.md 或 IMPROVEMENTS.md）：
  O1 每个高斯的最小 σ 轴，记为 n_g（盘法向）。
  O2 视线方向 view_dir = (P − C)/|P − C|（相机中心指向该点）。
  O3 深度图法向 n_L：在来源像素处用世界坐标的 u/v 前向差分叉积。
  O4 掠射分层：slant = arccos(|n_L · view_dir|)，0°=正对相机，90°=掠射。
     只有 slant 大的区域才能区分 billboard 与 surfel。
  O5 判读：若 ang_view（n_g vs 视线）在所有 slant 档都小、且 ang_surf
     （n_g vs 表面法向）随 slant 单调上升 → billboard。
     反之 → surfel。

只读，不改产物。
"""
import json
import os
import re
import numpy as np

PLY = os.environ["PLY"]
NPZ = os.environ["NPZ"]
OUT = os.environ.get("OUT", "/tmp")
VIEWS = [int(x) for x in os.environ.get("VIEWS", "0,8,16,24").split(",")]
STRIDE = int(os.environ.get("STRIDE", "4"))   # 带宽像素的采样步长


def log(*a):
    print(*a, flush=True)


d = np.load(NPZ)
W2C = d["extrinsics"].astype(np.float64)      # (V,4,4)
K = d["intrinsics"].astype(np.float64)
DEP = d["depth"].astype(np.float64)           # (V,H,W)
V, H, W = DEP.shape

# ── 复刻 save_gaussian_ply 的屏蔽，复原每个顶点的来源视角与像素 ──
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

props = re.search(r"element vertex \d+\n((?:property float \S+\n)+)", head)
names = re.findall(r"property float (\S+)", props.group(1))
dt = np.dtype([(n, "f4") for n in names])
raw = np.frombuffer(blob, dtype=dt, count=N, offset=cut)
xyz = np.stack([raw["x"], raw["y"], raw["z"]], 1).astype(np.float64)
sc = np.exp(np.stack([raw[f"scale_{i}"] for i in range(3)], 1).astype(np.float64))
quat = np.stack([raw[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)  # wxyz
log(f"📦 {N:,} 高斯 / {V} 视角")

src_pix = np.zeros((N, 2), np.int64)
off = 0
for v in range(V):
    hh, ww = np.nonzero(mask[v])
    n = len(hh)
    src_pix[off:off + n, 0] = ww
    src_pix[off:off + n, 1] = hh
    off += n


def quat_to_rot(q):
    """wxyz → (N,3,3)，与 3DGS build_rotation 同式。"""
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


ROT = quat_to_rot(quat)            # (N,3,3)
AXES = [ROT[:, :, i] for i in range(3)]   # 三列 = 三个主轴方向
kmin = np.argmin(sc, axis=1)              # 每个高斯的最小 σ 轴索引


def view_geom(v):
    """返回 (Pc, Pw, C, 相机系→世界的 R)"""
    fx, fy = K[v, 0, 0], K[v, 1, 1]
    cx, cy = K[v, 0, 2], K[v, 1, 2]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    z = DEP[v]
    Pc = np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], -1)
    R, t = W2C[v, :, :3], W2C[v, :, 3]
    Pw = (Pc - t) @ R                      # Xw = R^T (Xc − t)
    C = -R.T @ t                           # 相机中心（世界）
    return Pc, Pw, C, R


def normal_map(Pw, C):
    """世界坐标法向（朝相机），u/v 前向差分叉积。"""
    du = np.zeros_like(Pw)
    dv = np.zeros_like(Pw)
    du[:, :-1] = Pw[:, 1:] - Pw[:, :-1]
    dv[:-1, :] = Pw[1:, :] - Pw[:-1, :]
    n = np.cross(du, dv)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    ok = nn[..., 0] > 1e-12
    n = np.divide(n, np.maximum(nn, 1e-12))
    view = Pw - C[None, None, :]
    view /= np.maximum(np.linalg.norm(view, axis=-1, keepdims=True), 1e-12)
    flip = np.sum(n * view, axis=-1) > 0
    n[flip] *= -1
    return n, ok


def ang(u, v):
    c = np.abs(np.sum(u * v, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


rep = {"ply": PLY, "npz": NPZ, "n": int(N), "views": {}}
log(f"\n{'view':>5} {'n':>8} {'σ中位(px)':>10} {'argmin轴0/1/2':>14} "
    f"{'ang_view中位':>12} {'ang_surf中位':>12}")

OFF = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)

for v in VIEWS:
    # 像素索引（取几何）与全局高斯索引（取参数）必须分开 —— 二者长度相同但语义不同
    pix = np.nonzero(mask[v].reshape(-1))[0]
    gid = OFF[v] + np.arange(cnt[v])
    if len(pix) == 0:
        continue
    _, Pw_map, C, R = view_geom(v)
    nL_map, okL = normal_map(Pw_map, C)

    Pw = Pw_map.reshape(-1, 3)[pix]
    nL = nL_map.reshape(-1, 3)[pix]
    okLv = okL.reshape(-1)[pix]

    # 视线方向（相机 → 点）
    view = Pw - C[None, :]
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)

    k = kmin[gid]
    # n_g：最小 σ 轴的方向
    n_g = np.stack([AXES[0][gid], AXES[1][gid], AXES[2][gid]], 1)[
        np.arange(len(k)), :, k]
    # 三轴 vs 表面法向 / 视线，取最小夹角（判断是否有某一轴已对齐）
    ang_surf_axes = np.stack([ang(AXES[j][gid], nL) for j in range(3)], 1)
    ang_view_axes = np.stack([ang(AXES[j][gid], view) for j in range(3)], 1)

    a_view = ang(n_g, view)
    a_surf = ang(n_g, nL)

    # σ 折合到该视角的像素半径
    z = np.linalg.norm(Pw - C[None, :], axis=1)
    s_fx = sc[gid] / np.maximum(z, 1e-9)[:, None] * K[v, 0, 0]

    slant = ang(nL, view)
    m = okLv
    log(f"{v:>5} {len(pix):>8,} {np.median(s_fx.max(1)[m]):>10.2f} "
        f"{np.bincount(k[m], minlength=3)}  "
        f"{np.median(a_view[m]):>11.1f}° {np.median(a_surf[m]):>11.1f}°")

    tiers = []
    for lo, hi in ((0, 30), (30, 60), (60, 90)):
        t = m & (slant >= lo) & (slant < hi)
        if t.sum() < 200:
            continue
        tiers.append(dict(
            slant=f"{lo}-{hi}", n=int(t.sum()),
            frac=float(t.sum() / max(m.sum(), 1)),
            ang_view_med=float(np.median(a_view[t])),
            ang_surf_med=float(np.median(a_surf[t])),
            ang_surf_axes_med=float(np.median(ang_surf_axes[t].min(1))),
            ang_view_axes_med=float(np.median(ang_view_axes[t].min(1))),
        ))

    rep["views"][str(v)] = dict(
        n=int(len(pix)), kmin_hist=np.bincount(k[m], minlength=3).tolist(),
        sigma_px_med=np.median(s_fx[m], 0).tolist(),
        ang_view_med=float(np.median(a_view[m])),
        ang_surf_med=float(np.median(a_surf[m])),
        ang_view_p25_75=[float(np.percentile(a_view[m], 25)),
                         float(np.percentile(a_view[m], 75))],
        ang_surf_p25_75=[float(np.percentile(a_surf[m], 25)),
                         float(np.percentile(a_surf[m], 75))],
        tiers=tiers,
    )

# ── 汇总：掠射档的判别力最强 ──
allt = [t for r in rep["views"].values() for t in r["tiers"]]
log("\n按表面倾斜度分层（slant = 表面法向与视线夹角；只有大 slant 才区分得开）：")
log(f"{'slant':>10} {'占比':>8} {'ang_view中位':>12} {'ang_surf中位':>12} "
    f"{'最近轴vs表面':>12} {'最近轴vs视线':>12}")
for tag in ("0-30", "30-60", "60-90"):
    ts = [t for t in allt if t["slant"] == tag]
    if not ts:
        continue
    log(f"{tag:>10} {np.mean([t['frac'] for t in ts]):>7.1%} "
        f"{np.mean([t['ang_view_med'] for t in ts]):>11.1f}° "
        f"{np.mean([t['ang_surf_med'] for t in ts]):>11.1f}° "
        f"{np.mean([t['ang_surf_axes_med'] for t in ts]):>11.1f}° "
        f"{np.mean([t['ang_view_axes_med'] for t in ts]):>11.1f}°")

rep["tier_summary"] = {
    tag: dict(
        frac=float(np.mean([t["frac"] for t in [x for x in allt if x["slant"] == tag]]))
        if any(x["slant"] == tag for x in allt) else None,
        ang_view_med=float(np.mean([t["ang_view_med"] for t in allt if t["slant"] == tag]))
        if any(x["slant"] == tag for x in allt) else None,
        ang_surf_med=float(np.mean([t["ang_surf_med"] for t in allt if t["slant"] == tag]))
        if any(x["slant"] == tag for x in allt) else None,
        ang_surf_axes_med=float(np.mean([t["ang_surf_axes_med"] for t in allt if t["slant"] == tag]))
        if any(x["slant"] == tag for x in allt) else None,
    )
    for tag in ("0-30", "30-60", "60-90")
}

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(rep, f, ensure_ascii=False, indent=1, default=float)
log(f"\n💾 {OUT}")
