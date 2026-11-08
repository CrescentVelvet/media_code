#!/usr/bin/env python3
"""AUDIT-15 · I-03：只换 2DGS 光栅器（头不动、不重训），量「表示/渲染器错配」的净代价。

单变量：同一个 PLY（i12_fullpts_sh，官方等价 22.11 dB）、同一组相机、同 sh_degree=2、
同 eps2d=0.3、同一 slant mask、同一组视角对。唯一变量是光栅器。

三档：
  3dgs          EWA 3DGS（I-00 基线，22.11 dB / 掠射一致性 = 参照）
  2dgs_naive    直接喂 2DGS、不重排轴序 → 预期崩成零宽刀片（现场验证 E7）
  2dgs_reorder  轴重排后喂 2DGS（I-03 本体）

轴重排（E7）：
  gsplat 1.5.3 的 `rasterization_2dgs` **没有 normals 参数**（audit_15a Q1 实测），
  盘法向由 quats 推出的旋转矩阵**第三列**决定；且光栅器只用 scales[0]、scales[1]
  （`Projection2DGSFused.cu:152` 第三维硬编码 1.0）。而 DA3 的最小轴 99.9% 落在
  index 0（audit_12）。所以：
      s' = [s1, s2, s0]          第三轴 = 原最小轴 = 盘法向
      R' = R @ P,  P = [[0,0,1],[1,0,0],[0,1,0]]
      q' = q ⊗ q_P,  q_P = (0.5, 0.5, 0.5, 0.5)   ← P 是绕 (1,1,1)/√3 转 120°
  验证：R(0.5,0.5,0.5,0.5) 逐元素等于 P（脚本内有断言）。

判据见 IMPROVEMENTS.md §3.5.0（跑之前写死）。

一个易踩的坑（audit_15a Q3）：gsplat 1.5.3 的 `rasterization_2dgs` 在
`sh_degree=None` 时不可用 —— "Compute the per-view colors" 那段被整块注释掉，
colors 停在 (N,3)，下游断言 `colors.shape[:-2] == (B,C,N)` 直接失败。
本脚本一律传 SH 系数 + sh_degree。
"""
import json
import os
import re

import numpy as np
import torch
from PIL import Image
from gsplat import rasterization, rasterization_2dgs
from skimage.metrics import structural_similarity as ssim

SC = os.environ["SC"]
NPZ = os.environ["NPZ"]                       # 参照物：cameras + GT 深度 + processed_images
PLY = os.environ.get("PLY", os.path.join(SC, "official_export/i12_fullpts_sh/0000.ply"))
OUT = os.environ.get("OUT", os.path.join(SC, "i03_2dgs"))
VIEWS = os.environ.get("VIEWS", "all")
NAIVE_VIEWS = os.environ.get("NAIVE_VIEWS", "0,8,16")
dev = "cuda"


def log(*a):
    print(*a, flush=True)


# ── PLY ──────────────────────────────────────────────────────────────────────
def read_ply(path):
    """按 header 真实偏移读，不猜 4096（E9）。f_rest 是 channel-major（E8）。"""
    with open(path, "rb") as f:
        head = b""
        while b"end_header" not in head:
            head += f.read(1)
        tail = f.read(1)
        while tail not in (b"\n", b""):
            head += tail
            tail = f.read(1)
        head += tail
        offset = len(head)
        text = head.decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", text).group(1))
        names = re.findall(r"^property float (\w+)$", text, re.M)
        dt = np.dtype([(nm, "<f4") for nm in names])
        f.seek(offset)
        raw = np.fromfile(f, dtype=dt, count=n, offset=0)
    if raw.shape[0] != n:
        raise SystemExit(f"❌ {path}: 读到 {raw.shape[0]} 顶点，头里写 {n}")
    means = np.stack([raw["x"], raw["y"], raw["z"]], 1).astype(np.float32)
    scales = np.exp(np.stack([raw[f"scale_{i}"] for i in range(3)], 1)).astype(np.float32)
    opac = (1.0 / (1.0 + np.exp(-raw["opacity"].astype(np.float64)))).astype(np.float32)
    quats = np.stack([raw[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    f_dc = np.stack([raw[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32)
    rest_names = [nm for nm in names if nm.startswith("f_rest_")]
    k = len(rest_names) // 3
    if k:
        f_rest = np.stack([raw[f"f_rest_{i}"] for i in range(len(rest_names))], 1)
        f_rest = f_rest.reshape(len(means), 3, k).transpose(0, 2, 1)
        sh = np.concatenate([f_dc[:, None, :], f_rest], 1)
    else:
        sh = f_dc[:, None, :]
    return dict(means=means, scales=scales, opac=opac, quats=quats, sh=sh,
                n=int(n), n_rest=len(rest_names),
                deg=int(round((1 + k) ** 0.5)) - 1)


# ── 轴重排（E7）：q' = q ⊗ q_P，s' = [s1, s2, s0] ─────────────────────────────
def quat_to_mat(q):
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), np.float64)
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


def quat_mul(a, b):
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], 1)


P_PERM = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], np.float64)
Q_P = np.array([[0.5, 0.5, 0.5, 0.5]])


def reorder_for_2dgs(scales, quats):
    """把最小轴换到 index 2，切平面内两轴放到 index 0/1。"""
    s_new = np.stack([scales[:, 1], scales[:, 2], scales[:, 0]], 1).astype(np.float32)
    q_new = quat_mul(quats.astype(np.float64), Q_P).astype(np.float32)
    q_new /= np.linalg.norm(q_new, axis=1, keepdims=True) + 1e-12
    return s_new, q_new


# ── 相机 / GT ────────────────────────────────────────────────────────────────
d = np.load(NPZ)
W2C = d["extrinsics"].astype(np.float64)
if W2C.shape[-2] == 4:
    W2C = W2C[:, :3, :]
K = d["intrinsics"].astype(np.float64)
GTDEP = d["depth"].astype(np.float64)
GT = d["processed_images"]
V, H, Wd = GTDEP.shape
fx, fy = K[:, 0, 0], K[:, 1, 1]
cx, cy = K[:, 0, 2], K[:, 1, 2]
log(f"📷 {V} 视角 {Wd}x{H}  fx={fx.mean():.2f}  scale_factor={float(d['scale_factor']):.4f}")

FACE = (0.35, 0.105, 0.565, 0.335)
fbox = (int(FACE[0] * Wd), int(FACE[1] * H), int(FACE[2] * Wd), int(FACE[3] * H))
fmask = np.zeros((H, Wd), bool)
fmask[fbox[1]:fbox[3], fbox[0]:fbox[2]] = True

uu, vv = np.meshgrid(np.arange(Wd) + 0.5, np.arange(H) + 0.5)


def unproject(v, z):
    R, t = W2C[v, :, :3], W2C[v, :, 3]
    xc = np.stack([(uu - cx[v]) / fx[v] * z, (vv - cy[v]) / fy[v] * z, z], -1)
    return (xc - t) @ R


def project(v, Pw):
    R, t = W2C[v, :, :3], W2C[v, :, 3]
    p = Pw @ R.T + t
    z = p[..., 2]
    zs = np.where(np.abs(z) < 1e-9, np.nan, z)
    return fx[v] * p[..., 0] / zs + cx[v], fy[v] * p[..., 1] / zs + cy[v], z


def slant_map(v):
    """GT 深度图的表面倾斜度（法向 vs 视线），与渲染器无关 → 两档共用同一 mask。"""
    z = GTDEP[v]
    Pw = unproject(v, z)
    C = -W2C[v, :, :3].T @ W2C[v, :, 3]
    du = np.zeros_like(Pw)
    dv = np.zeros_like(Pw)
    du[:, :-1] = Pw[:, 1:] - Pw[:, :-1]
    dv[:-1, :] = Pw[1:, :] - Pw[:-1, :]
    n = np.cross(du, dv)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.divide(n, np.maximum(nn, 1e-12))
    view = Pw - C[None, None, :]
    view /= np.maximum(np.linalg.norm(view, axis=-1, keepdims=True), 1e-12)
    s = np.degrees(np.arccos(np.clip(np.abs(np.sum(n * view, -1)), -1, 1)))
    return s, z > 0, (nn[..., 0] > 1e-12)


SLANT, GTVALID, OKN = zip(*[slant_map(v) for v in range(V)])
log(f"🎯 掠射区（slant>60°）占 GT 前景 "
    f"{np.mean([((SLANT[v] > 60) & GTVALID[v]).sum() / max(GTVALID[v].sum(), 1) for v in range(V)]):.1%}")


# ── 渲染 ─────────────────────────────────────────────────────────────────────
def vm_of(v):
    """(4,4) w2c，与 audit_13 同构（W2C 只存 3×4，补最后一行）。"""
    return torch.from_numpy(
        np.vstack([W2C[v], np.array([0, 0, 0, 1.0])]).astype(np.float32)).to(dev)


def render_3dgs(g, v, deg):
    with torch.no_grad():
        img, alpha, _ = rasterization(
            g["means"], g["quats"], g["scales"], g["opac"], g["sh"],
            vm_of(v)[None].float(), torch.from_numpy(K[v]).float().to(dev)[None],
            Wd, H, sh_degree=deg, packed=False, render_mode="RGB+ED")
    return (img[0, ..., :3].clamp(0, 1).mul(255).byte().cpu().numpy(),
            alpha[0].squeeze(-1).cpu().numpy(),
            img[0, ..., 3].cpu().numpy())


def render_2dgs(g, v, deg, depth_mode="expected"):
    with torch.no_grad():
        out = rasterization_2dgs(
            g["means"], g["quats"], g["scales"], g["opac"], g["sh"],
            vm_of(v)[None].float(), torch.from_numpy(K[v]).float().to(dev)[None],
            Wd, H, sh_degree=deg, packed=False, render_mode="RGB+ED",
            depth_mode=depth_mode)
    img = out[0]
    return (img[0, ..., :3].clamp(0, 1).mul(255).byte().cpu().numpy(),
            out[1][0].squeeze(-1).cpu().numpy(),
            img[0, ..., 3].cpu().numpy(),
            out[5][0].squeeze(-1).cpu().numpy(),
            out[4][0].squeeze(-1).cpu().numpy())


def psnr(a, b, m=None):
    if m is not None:
        a, b = a[m], b[m]
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def sharp(rgb):
    import cv2
    g = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# 视角对闸门（首尾不相邻，audit_15b 实测）
cent = np.stack([-W2C[v, :, :3].T @ W2C[v, :, 3] for v in range(V)])
step = np.linalg.norm(np.roll(cent, -1, 0) - cent, axis=1)
PAIRS = [(v, (v + 1) % V) for v in range(V) if step[v] <= 3 * np.median(step)]
log(f"🔗 视角对闸门：保留 {len(PAIRS)}/{V} 对（剔除基线 >3× 中位的）")


def depth_consistency(DEP, tag):
    """把 v 的渲染深度反投影、投到 v+1，比较相对深度误差；按 GT slant 分档。

    只用「两视角都已渲染」的相邻对 —— 快验（VIEWS=0,1,8,9,16,17）时也成立。
    """
    tiers = {"0-30": (0, 30), "30-60": (30, 60), "60-90": (60, 90), "all": (0, 90)}
    acc = {k: [] for k in tiers}
    covs = []
    per_pair = []
    for v, u2 in [(a, b) for a, b in PAIRS if a in DEP and b in DEP]:
        z = DEP[v]
        cov = float((z > 0).sum() / max(GTVALID[v].sum(), 1))
        covs.append(cov)
        ok = GTVALID[v] & OKN[v] & (z > 0) & (DEP[u2] > 0)
        if ok.sum() < 500:
            per_pair.append(dict(v=v, u=u2, cov=cov, med=None))
            continue
        Pw = unproject(v, z)                  # 用渲染深度反投影
        uu2, vv2, z2 = project(u2, Pw)
        ui = np.round(uu2).astype(np.int64)
        vi = np.round(vv2).astype(np.int64)
        ok &= np.isfinite(z2) & (z2 > 0) & (ui >= 0) & (ui < Wd) & (vi >= 0) & (vi < H)
        zb = np.zeros_like(z2)
        zb[ok] = DEP[u2][vi[ok], ui[ok]]
        ok &= zb > 0
        if ok.sum() < 500:
            per_pair.append(dict(v=v, u=u2, cov=cov, med=None))
            continue
        rel = np.abs(z2 - zb) / np.maximum(z2, 1e-9) * 100.0
        s = SLANT[v]
        cells = {}
        for k, (lo, hi) in tiers.items():
            m = ok & (s >= lo) & (s < hi)
            if m.sum() > 300:
                acc[k].append(rel[m])
                cells[k] = float(np.median(rel[m]))
        per_pair.append(dict(v=v, u=u2, cov=cov, med=float(np.median(rel[ok])), tiers=cells))
    out = {}
    for k in tiers:
        if acc[k]:
            a = np.concatenate(acc[k])
            out[k] = dict(med=float(np.median(a)), p90=float(np.percentile(a, 90)),
                          n=int(a.size))
        else:
            out[k] = dict(med=None, p90=None, n=0)
    out["depth_cov"] = float(np.mean(covs))
    out["per_pair"] = per_pair
    log(f"  [{tag}] 深度覆盖 {out['depth_cov']:.1%} ｜一致性中位 "
        + "  ".join(f"{k} {out[k]['med']:.2f}%" for k in tiers if out[k]["med"] is not None))
    return out


# ── 主流程 ───────────────────────────────────────────────────────────────────
os.makedirs(OUT, exist_ok=True)
g0 = read_ply(PLY)
log(f"\n📦 {os.path.basename(os.path.dirname(PLY))}/{os.path.basename(PLY)}: "
    f"{g0['n']:,} 点  SH deg {g0['deg']}  α中位 {np.median(g0['opac']):.4f}")

kmin = np.argmin(g0["scales"], 1)
log(f"  最小轴落在 index 0/1/2 = "
    f"{np.bincount(kmin, minlength=3).tolist()}  "
    f"（index0 占 {(kmin == 0).mean():.2%} → E7 必做）")

# 自证：q_P 就是那个置换矩阵
RP = quat_to_mat(Q_P.copy())[0]
assert np.allclose(RP, P_PERM, atol=1e-9), f"q_P 不等于置换矩阵\n{RP}"
log("  ✅ q_P 自证：R(0.5,0.5,0.5,0.5) 逐元素等于 [[0,0,1],[1,0,0],[0,1,0]]")

# 重排自证：新旋转第 3 列 == 原第 1 列（最小轴），新 scale 顺序 == [s1,s2,s0]
s_re, q_re = reorder_for_2dgs(g0["scales"], g0["quats"])
R0 = quat_to_mat(g0["quats"].astype(np.float64))
R1 = quat_to_mat(q_re.astype(np.float64))
assert np.allclose(R1[:, :, 2], R0[:, :, 0], atol=1e-5), "重排后第 3 列 ≠ 原最小轴"
assert np.allclose(s_re, g0["scales"][:, [1, 2, 0]], atol=1e-9)
log("  ✅ 重排自证：新第 3 列 == 原第 0 列（最小轴），s' == [s1,s2,s0]")

def to_dev(g):
    return {k: torch.from_numpy(v).to(dev) for k, v in g.items()
            if k in ("means", "quats", "scales", "opac", "sh")}


G3 = to_dev(g0)
G2R = to_dev(dict(g0, scales=s_re, quats=q_re))
G2N = to_dev(g0)                                    # naive：不重排
deg = g0["deg"]
view_list = list(range(V)) if VIEWS.strip() == "all" else [int(x) for x in VIEWS.split(",")]
naive_views = [int(x) for x in NAIVE_VIEWS.split(",") if x.strip()]

rep = dict(ply=PLY, npz=NPZ, wh=[Wd, H], views=view_list,
           kmin_hist=np.bincount(kmin, minlength=3).tolist(), cases={})

# ① 3DGS 基线
log("\n" + "=" * 78)
log("① 3DGS EWA（基线）")
log("=" * 78)
DEP3, ps, ss, sp, fs, fsp, al = {}, [], [], [], [], [], []
for v in view_list:
    rgb, a, dep = render_3dgs(G3, v, deg)
    DEP3[v] = dep
    ps.append(psnr(rgb, GT[v]))
    ss.append(float(ssim(rgb, GT[v], channel_axis=-1, data_range=255)))
    sp.append(sharp(rgb))
    al.append(float(np.median(a)))
    if v in (0, 8, 16):
        fs.append(psnr(rgb, GT[v], fmask))
        fsp.append(sharp(rgb[fbox[1]:fbox[3], fbox[0]:fbox[2]]))
        Image.fromarray(rgb).save(os.path.join(OUT, f"3dgs_view{v:03d}.png"))
        np.save(os.path.join(OUT, f"depth_3dgs_view{v:03d}.npy"), dep)
log(f"  均值 PSNR {np.mean(ps):.2f} dB  SSIM {np.mean(ss):.3f}  锐度 {np.mean(sp):.0f}  "
    f"α中位 {np.mean(al):.4f}")
if fs:
    log(f"  人脸（view 0/8/16）PSNR {np.mean(fs):.2f} dB  锐度 {np.mean(fsp):.0f}")
rep["cases"]["3dgs"] = dict(psnr=float(np.mean(ps)), ssim=float(np.mean(ss)),
                            sharp=float(np.mean(sp)), alpha_med=float(np.mean(al)),
                            face_psnr=float(np.mean(fs)) if fs else None,
                            face_sharp=float(np.mean(fsp)) if fsp else None)
rep["cases"]["3dgs"].update(depth_consistency(DEP3, "3dgs"))

# ② 2DGS naive（不重排）—— E7 现场验证。同视角跑一次重排版做严格对照。
# 注：α 像素数不可用（32 层叠加后 α 处处 ≈1，恒等于全图 378×504），改用 PSNR 差。
log("\n" + "=" * 78)
log("② 2DGS naive（不重排轴序）—— 预期崩成零宽刀片")
log("=" * 78)
nv, e7_naive_ps, e7_re_ps = {}, [], []
for v in naive_views:
    rgb, a, dep, med, dist = render_2dgs(G2N, v, deg)
    rgb_re, a_re, _, _, _ = render_2dgs(G2R, v, deg)
    p_naive, p_re = psnr(rgb, GT[v]), psnr(rgb_re, GT[v])
    e7_naive_ps.append(p_naive)
    e7_re_ps.append(p_re)
    nv[v] = dict(psnr=p_naive, psnr_reorder=p_re, delta=p_naive - p_re,
                 alpha_px=int((a > 0.01).sum()))
    log(f"  view {v:>2}: 不重排 PSNR {p_naive:>6.2f} ｜ 重排 {p_re:>6.2f} ｜ "
        f"Δ {p_naive - p_re:>+6.2f} dB")
rep["cases"]["2dgs_naive"] = nv
E7_DELTA = float(np.mean(e7_naive_ps) - np.mean(e7_re_ps))

# ③ 2DGS reorder（I-03 本体）
log("\n" + "=" * 78)
log("③ 2DGS reorder（I-03 本体）")
log("=" * 78)
DEP2, DEP2M, ps2, ss2, sp2, fs2, fsp2, al2, dcov = {}, {}, [], [], [], [], [], [], []
for v in view_list:
    rgb, a, dep, med, dist = render_2dgs(G2R, v, deg, "expected")
    DEP2[v], DEP2M[v] = dep, med
    ps2.append(psnr(rgb, GT[v]))
    ss2.append(float(ssim(rgb, GT[v], channel_axis=-1, data_range=255)))
    sp2.append(sharp(rgb))
    al2.append(float(np.median(a)))
    dcov.append(float(np.median(dist)))
    if v in (0, 8, 16):
        fs2.append(psnr(rgb, GT[v], fmask))
        fsp2.append(sharp(rgb[fbox[1]:fbox[3], fbox[0]:fbox[2]]))
        Image.fromarray(rgb).save(os.path.join(OUT, f"2dgs_view{v:03d}.png"))
        np.save(os.path.join(OUT, f"depth_2dgs_view{v:03d}.npy"), dep)
log(f"  均值 PSNR {np.mean(ps2):.2f} dB  SSIM {np.mean(ss2):.3f}  锐度 {np.mean(sp2):.0f}  "
    f"α中位 {np.mean(al2):.4f}  distort 中位 {np.mean(dcov):.4f}")
if fs2:
    log(f"  人脸（view 0/8/16）PSNR {np.mean(fs2):.2f} dB  锐度 {np.mean(fsp2):.0f}")
rep["cases"]["2dgs_reorder"] = dict(psnr=float(np.mean(ps2)), ssim=float(np.mean(ss2)),
                                    sharp=float(np.mean(sp2)), alpha_med=float(np.mean(al2)),
                                    face_psnr=float(np.mean(fs2)) if fs2 else None,
                                    face_sharp=float(np.mean(fsp2)) if fsp2 else None,
                                    distort_med=float(np.mean(dcov)))
rep["cases"]["2dgs_reorder"].update(depth_consistency(DEP2, "2dgs_exp"))

# 次判据：2DGS median depth vs expected
log("\n" + "=" * 78)
log("④ 次判据：2DGS median_depth（ray 交点中位数）")
log("=" * 78)
rep["cases"]["2dgs_median"] = depth_consistency(DEP2M, "2dgs_med")

# ── 判据自检 ─────────────────────────────────────────────────────────────────
log("\n" + "=" * 78)
log("判据（IMPROVEMENTS.md §3.5.0 + §3.5.1 订正，跑前写死）")
log("=" * 78)
c3, c2, c2m = rep["cases"]["3dgs"], rep["cases"]["2dgs_reorder"], rep["cases"]["2dgs_median"]
r = c2["60-90"]["med"] / c3["60-90"]["med"]
r_med = c2m["60-90"]["med"] / c3["60-90"]["med"]
gate = dict(
    P_D_ratio=round(float(r), 3),
    P_D_pass=bool(r <= 0.90),
    P_D_abs_3dgs=round(float(c3["60-90"]["med"]), 3),
    P_D_abs_2dgs=round(float(c2["60-90"]["med"]), 3),
    P_D2_abs=round(float(c2["60-90"]["med"]), 3),
    P_D2_pass=bool(c2["60-90"]["med"] < 2.40),
    P_D3_ratio=round(float(r_med), 3),
    P_D3_abs=round(float(c2m["60-90"]["med"]), 3),
    P_D3_pass=bool(r_med <= 0.70),
    P_COV_ratio=round(float(c2["depth_cov"] / max(c3["depth_cov"], 1e-9)), 3),
    P_COV_pass=bool(c2["depth_cov"] / max(c3["depth_cov"], 1e-9) >= 0.95),
    P_E7_delta=round(E7_DELTA, 3) if nv else None,
    P_E7_pass=bool(E7_DELTA <= -1.0) if nv else None,
    d_psnr_all=round(float(c2["psnr"] - c3["psnr"]), 3),
    d_psnr_face=round(float((c2["face_psnr"] or 0) - (c3["face_psnr"] or 0)), 3),
    d_sharp=round(float(c2["sharp"] - c3["sharp"]), 1),
    d_face_sharp=round(float((c2["face_sharp"] or 0) - (c3["face_sharp"] or 0)), 1),
)
gate["counter"] = bool(gate["d_psnr_all"] < -1.0 and not gate["P_D_pass"])
gate["verdict"] = ("A 档：错配是主因" if gate["P_D_pass"]
                   else "B 档：能力存在但未转化为观感（A2 成立 / A1 未成立）"
                   if gate["P_D3_pass"] else "C 档：换渲染器不解决问题")
rep["gate"] = gate

log(f"  P-D   掠射区 expected：3DGS {gate['P_D_abs_3dgs']:.2f}% → 2DGS "
    f"{gate['P_D_abs_2dgs']:.2f}%（比值 {gate['P_D_ratio']:.3f}，阈值 ≤0.90）"
    f" {'✅' if gate['P_D_pass'] else '❌'}")
log(f"  P-D2  2DGS 掠射绝对值 {gate['P_D2_abs']:.2f}%（阈值 <2.40%）"
    f" {'✅' if gate['P_D2_pass'] else '❌'}")
log(f"  P-D3  掠射区 median：3DGS {gate['P_D_abs_3dgs']:.2f}% → 2DGS "
    f"{gate['P_D3_abs']:.2f}%（比值 {gate['P_D3_ratio']:.3f}，阈值 ≤0.70）"
    f" {'✅' if gate['P_D3_pass'] else '❌'}")
log(f"  P-COV 深度覆盖比 {gate['P_COV_ratio']:.3f}（阈值 ≥0.95）"
    f" {'✅' if gate['P_COV_pass'] else '❌'}")
if gate["P_E7_delta"] is not None:
    log(f"  P-E7  naive ΔPSNR {gate['P_E7_delta']:+.2f} dB（阈值 ≤−1.0）"
        f" {'✅' if gate['P_E7_pass'] else '❌'}")
log(f"  只报不判：全图 PSNR {gate['d_psnr_all']:+.2f} dB ｜ 人脸 {gate['d_psnr_face']:+.2f} dB"
    f" ｜ 锐度 {gate['d_sharp']:+.0f} ｜ 人脸锐度 {gate['d_face_sharp']:+.0f}")
if gate["counter"]:
    log("  ⚠️ 触发反判据：全图 PSNR 降 >1 dB 且掠射区无改善")
log(f"\n  ⇒ 判定：{gate['verdict']}")

with open(os.path.join(OUT, "i03_2dgs.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(rep, f, indent=1, ensure_ascii=False, default=float)
log(f"\n💾 {os.path.join(OUT, 'i03_2dgs.json')}")
