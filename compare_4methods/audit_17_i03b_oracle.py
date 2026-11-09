#!/usr/bin/env python3
"""AUDIT-17 · I-03b：把盘法向换成 oracle（GT 深度图表面法向），验 H2。

背景（I-03，audit_15）：只换 2DGS 光栅器 → 全图 PSNR −0.29、锐度 −66、人脸锐度 −69。
按 GT slant 分档看 RGB 误差增量，**正对档 0–30° 反而最差（+7.54%）**，掠射档 +5.14%。
推翻了"掠射透视病态"假设，指向 H2：DA3 的盘法向 vs 真实表面 **56.9~66.4°**
（随机基线 57.3°）→ 盘朝向没有几何意义，2DGS 把每个高斯投到随机朝向的平面上。

本脚本是 H2 的直接检验：**法向给正确答案**，其余（点数/位置/尺度/颜色/透明度/
轴序约定/相机/判据）一律不动。
  O1  per-view oracle：渲染 view v 时用 v 自己的 GT 法向（上界，但跨视角不一致）
  O2  world-consistent oracle：每点取各视角可见法向的平均（唯一法向，可被监督项逼出来）
        —— 这才是"如果训练时加正确的法向正则，能不能到"的现实上限

法向采样必须过一致性闸门：高斯投影到 v 后，若该像素的 GT 深度与投影深度相对差
> GATE（默认 3%），说明该像素的 GT 表面不是这个高斯本身（遮挡/背景）→ 不采信，回落
到 I-03 的重排法向。覆盖率与回落比例逐档报出。

法向符号：盘平面对 ±n 对称（法向轴上是厚度 s0），符号不影响渲染；但 O2 求平均前
必须统一朝向（翻到朝相机），否则两个反号的单位向量相加会互相抵消。

grep 关键词：I-03b / oracle normal / H2 / 2DGS
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
NPZ = os.environ["NPZ"]
PLY = os.environ.get("PLY", os.path.join(SC, "official_export/i12_fullpts_sh/0000.ply"))
OUT = os.environ.get("OUT", os.path.join(SC, "i03b_oracle"))
VIEWS = os.environ.get("VIEWS", "all")
GATE = float(os.environ.get("GATE", "0.03"))
dev = "cuda"


def log(*a):
    print(*a, flush=True)


# ── PLY（同 audit_15：真实 header 偏移 / f_rest channel-major）─────────────────
def read_ply(path):
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


# ── 四元数 ↔ 矩阵 ────────────────────────────────────────────────────────────
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


def mat_to_quat(R):
    """R (N,3,3) → wxyz (N,4)，按 trace 最大分支取，再归一化。"""
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    s0 = np.sqrt(np.maximum(tr + 1.0, 1e-12)) * 2
    q0 = np.stack([0.25 * s0, (m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0], 1)
    s1 = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 1e-12)) * 2
    q1 = np.stack([(m21 - m12) / s1, 0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1], 1)
    s2 = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 1e-12)) * 2
    q2 = np.stack([(m02 - m20) / s2, (m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2], 1)
    s3 = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 1e-12)) * 2
    q3 = np.stack([(m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3], 1)
    k = np.argmax(np.stack([tr, m00, m11, m22], 1), 1)
    q = np.where((k == 0)[:, None], q0,
                 np.where((k == 1)[:, None], q1, np.where((k == 2)[:, None], q2, q3)))
    return q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)


def quat_mul(a, b):
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], 1)


Q_P = np.array([[0.5, 0.5, 0.5, 0.5]])
P_PERM = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], np.float64)


def reorder_for_2dgs(scales, quats):
    s_new = np.stack([scales[:, 1], scales[:, 2], scales[:, 0]], 1).astype(np.float32)
    q_new = quat_mul(quats.astype(np.float64), Q_P)
    q_new /= np.linalg.norm(q_new, axis=1, keepdims=True) + 1e-12
    return s_new, q_new.astype(np.float32)


def set_normal(R_in, n):
    """把 R_in 的第三列换成 n，面内主轴仍取**第 0 列**（对应 scales[0] = s1）。

    重排后 R 的列序是 [R1(s1), R2(s2), R0(s0)]，第 0/1 列才是面内两轴、第 2 列是法向。
    主轴必须跟着 s1 走：s_mid/s_max 中位差 4 倍（2.7 vs 11.6 mm），
    拿错列会把椭圆长轴旋转 90°，等于偷偷换掉了各向异性方向（单变量污染）。
    """
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    c0, c1 = R_in[:, :, 0], R_in[:, :, 1]
    t1 = c0 - np.sum(c0 * n, 1, keepdims=True) * n
    bad = np.linalg.norm(t1, axis=1) < 0.2
    if bad.any():
        t1[bad] = np.cross(n[bad], c1[bad])       # c0∥n 时 c1 ⟂ n，恒可用
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 = np.cross(n, t1)
    Rn = np.empty_like(R_in)
    Rn[:, :, 0] = t1
    Rn[:, :, 1] = t2
    Rn[:, :, 2] = n
    return Rn, int(bad.sum())


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

# 人脸窗必须逐视角给：人物在各视角的水平位置差很多（audit_21 实测）
#   旧写法 FACE=(0.35,0.105,0.565,0.335) 只对 view 0 勉强成立，view 8 直接落在书架上
FACE_V = {0: (0.415, 0.140, 0.585, 0.300),
          8: (0.515, 0.145, 0.665, 0.310),
          16: (0.430, 0.155, 0.600, 0.310)}


def face_crop(v):
    b = FACE_V[v]
    return (slice(int(b[1] * H), int(b[3] * H)), slice(int(b[0] * Wd), int(b[2] * Wd)))

uu, vv = np.meshgrid(np.arange(Wd) + 0.5, np.arange(H) + 0.5)
CAMS = np.stack([-W2C[v, :, :3].T @ W2C[v, :, 3] for v in range(V)])


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


def normal_map(v):
    """GT 深度图 → 世界系单位法向（同 audit_12/15 的前向差分叉积）。"""
    z = GTDEP[v]
    Pw = unproject(v, z)
    du = np.zeros_like(Pw)
    dv = np.zeros_like(Pw)
    du[:, :-1] = Pw[:, 1:] - Pw[:, :-1]
    dv[:-1, :] = Pw[1:, :] - Pw[:-1, :]
    n = np.cross(du, dv)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(nn, 1e-12), (z > 0) & (nn[..., 0] > 1e-12)


NMAP, NOK = zip(*[normal_map(v) for v in range(V)])
log(f"🧭 法向图就绪：{V} 张 @ {Wd}x{H}，有效像素均值 "
    f"{float(np.mean([NOK[v].mean() for v in range(V)])):.1%}")

SLANT = []
for v in range(V):
    z = GTDEP[v]
    Pw = unproject(v, z)
    view = Pw - CAMS[v][None, None, :]
    view /= np.maximum(np.linalg.norm(view, axis=-1, keepdims=True), 1e-12)
    SLANT.append(np.degrees(np.arccos(np.clip(np.abs(np.sum(NMAP[v] * view, -1)), -1, 1))))
GRZ = float(np.mean([((SLANT[v] > 60) & (GTDEP[v] > 0)).sum()
                     / max((GTDEP[v] > 0).sum(), 1) for v in range(V)]))
log(f"🎯 掠射区（slant>60°）占 GT 前景 {GRZ:.1%}")


# ── 渲染 ─────────────────────────────────────────────────────────────────────
def vm_of(v):
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


step = np.linalg.norm(np.roll(CAMS, -1, 0) - CAMS, axis=1)
PAIRS = [(v, (v + 1) % V) for v in range(V) if step[v] <= 3 * np.median(step)]
log(f"🔗 视角对闸门：保留 {len(PAIRS)}/{V} 对")


def depth_consistency(DEP, tag):
    tiers = {"0-30": (0, 30), "30-60": (30, 60), "60-90": (60, 90), "all": (0, 90)}
    acc = {k: [] for k in tiers}
    covs = []
    for v, u2 in [(a, b) for a, b in PAIRS if a in DEP and b in DEP]:
        z = DEP[v]
        covs.append(float((z > 0).sum() / max((GTDEP[v] > 0).sum(), 1)))
        ok = (GTDEP[v] > 0) & NOK[v] & (z > 0) & (DEP[u2] > 0)
        if ok.sum() < 500:
            continue
        Pw = unproject(v, z)
        uu2, vv2, z2 = project(u2, Pw)
        ui = np.round(uu2).astype(np.int64)
        vi = np.round(vv2).astype(np.int64)
        ok &= np.isfinite(z2) & (z2 > 0) & (ui >= 0) & (ui < Wd) & (vi >= 0) & (vi < H)
        zb = np.zeros_like(z2)
        zb[ok] = DEP[u2][vi[ok], ui[ok]]
        ok &= zb > 0
        if ok.sum() < 500:
            continue
        rel = np.abs(z2 - zb) / np.maximum(z2, 1e-9) * 100.0
        for k, (lo, hi) in tiers.items():
            m = ok & (SLANT[v] >= lo) & (SLANT[v] < hi)
            if m.sum() > 300:
                acc[k].append(rel[m])
    out = {}
    for k in tiers:
        if acc[k]:
            a = np.concatenate(acc[k])
            out[k] = dict(med=float(np.median(a)), p90=float(np.percentile(a, 90)),
                          n=int(a.size))
        else:
            out[k] = dict(med=None, p90=None, n=0)
    out["depth_cov"] = float(np.mean(covs)) if covs else float("nan")
    log(f"  [{tag}] 深度覆盖 {out['depth_cov']:.1%} ｜一致性中位 "
        + "  ".join(f"{k} {out[k]['med']:.2f}%" for k in tiers if out[k]["med"] is not None))
    return out


# ── 采样：把 GT 法向赋给每个高斯（带一致性闸门）──────────────────────────────
def sample_normals(v, means):
    """返回 (n (N,3), hit (N,) bool)。hit=False 的用重排法向兜底。"""
    u, vv_, z = project(v, means)
    ui = np.round(u).astype(np.int64)
    vi = np.round(vv_).astype(np.int64)
    ok = np.isfinite(z) & (z > 0) & (ui >= 0) & (ui < Wd) & (vi >= 0) & (vi < H)
    n = np.zeros_like(means)
    ok_idx = np.nonzero(ok)[0]
    zg = np.zeros(len(means))
    zg[ok_idx] = GTDEP[v][vi[ok_idx], ui[ok_idx]]
    ok[ok_idx] &= zg[ok_idx] > 0
    ok_idx = np.nonzero(ok)[0]
    rel = np.abs(z[ok_idx] - zg[ok_idx]) / np.maximum(z[ok_idx], 1e-9)
    keep = ok_idx[rel < GATE]
    n[keep] = NMAP[v][vi[keep], ui[keep]]
    hit = np.zeros(len(means), bool)
    hit[keep] = True
    # 翻到朝相机，供 O2 求平均
    dirv = CAMS[v][None, :] - means
    s = np.sum(n * dirv, 1)
    n[s < 0] *= -1.0
    return n, hit


# ── 主流程 ───────────────────────────────────────────────────────────────────
os.makedirs(OUT, exist_ok=True)
g0 = read_ply(PLY)
log(f"\n📦 {os.path.basename(os.path.dirname(PLY))}/{os.path.basename(PLY)}: "
    f"{g0['n']:,} 点  SH deg {g0['deg']}  α中位 {np.median(g0['opac']):.4f}")

s_re, q_re = reorder_for_2dgs(g0["scales"], g0["quats"])
R0 = quat_to_mat(g0["quats"].astype(np.float64))
Rre = quat_to_mat(q_re.astype(np.float64))
assert np.allclose(Rre[:, :, 2], R0[:, :, 0], atol=1e-5)
assert np.allclose(quat_to_mat(mat_to_quat(Rre)), Rre, atol=1e-5)
RP = quat_to_mat(Q_P.copy())[0]
assert np.allclose(RP, P_PERM, atol=1e-9)
log("  ✅ 自证：q_P 是置换矩阵；reorder 第 3 列 == 原最小轴；mat↔quat 往返 < 1e-5")

# 关键自证：法向给"原值"时必须还原成原矩阵（否则改的可不止是法向）
R_same, _ = set_normal(Rre, Rre[:, :, 2])
assert np.allclose(R_same, Rre, atol=1e-6), "set_normal 在 n=原法向时没还原，改动了面内轴"
log("  ✅ 自证：set_normal(n=原法向) 逐元素还原 Rre → 面内各向异性方向未被换")

M = g0["means"].astype(np.float64)
N0 = M.shape[0]
view_list = list(range(V)) if VIEWS.strip() == "all" else [int(x) for x in VIEWS.split(",")]
deg = g0["deg"]
rep = dict(ply=PLY, npz=NPZ, wh=[Wd, H], views=view_list, gate=GATE, cases={})


def to_dev(g):
    return {k: torch.from_numpy(v).to(dev) for k, v in g.items()
            if k in ("means", "quats", "scales", "opac", "sh")}


G3 = to_dev(g0)
G2R = to_dev(dict(g0, scales=s_re, quats=q_re))

# ① 3DGS 基线（同视角集）
log("\n" + "=" * 78)
log("① 3DGS EWA（基线，应与 I-03 逐位相同）")
log("=" * 78)
DEP3, ps, ss, sp, fs, fsp = {}, [], [], [], [], []
for v in view_list:
    rgb, a, dep = render_3dgs(G3, v, deg)
    DEP3[v] = dep
    ps.append(psnr(rgb, GT[v]))
    ss.append(float(ssim(rgb, GT[v], channel_axis=-1, data_range=255)))
    sp.append(sharp(rgb))
    if v in (0, 8, 16):
        fs.append(psnr(rgb[face_crop(v)], GT[v][face_crop(v)]))
        fsp.append(sharp(rgb[face_crop(v)]))
        Image.fromarray(rgb).save(os.path.join(OUT, f"3dgs_view{v:03d}.png"))
log(f"  均值 PSNR {np.mean(ps):.2f} dB  SSIM {np.mean(ss):.3f}  锐度 {np.mean(sp):.0f}")
if fs:
    log(f"  人脸（view 0/8/16）PSNR {np.mean(fs):.2f} dB  锐度 {np.mean(fsp):.0f}")
rep["cases"]["3dgs"] = dict(psnr=float(np.mean(ps)), ssim=float(np.mean(ss)),
                            sharp=float(np.mean(sp)),
                            face_psnr=float(np.mean(fs)) if fs else None,
                            face_sharp=float(np.mean(fsp)) if fsp else None)
rep["cases"]["3dgs"].update(depth_consistency(DEP3, "3dgs"))

# ①b 2DGS reorder（I-03 本体）—— 法向实验的严格对照：与 O2/O1 只差"第三列"
log("\n" + "=" * 78)
log("①b 2DGS reorder（I-03 本体，对照：与 oracle 只差旋转第三列）")
log("=" * 78)
DEP2R, ps_r, ss_r, sp_r, fs_r, fsp_r = {}, [], [], [], [], []
for v in view_list:
    rgb, a, dep, med, dist = render_2dgs(G2R, v, deg, "expected")
    DEP2R[v] = dep
    ps_r.append(psnr(rgb, GT[v]))
    ss_r.append(float(ssim(rgb, GT[v], channel_axis=-1, data_range=255)))
    sp_r.append(sharp(rgb))
    if v in (0, 8, 16):
        fs_r.append(psnr(rgb[face_crop(v)], GT[v][face_crop(v)]))
        fsp_r.append(sharp(rgb[face_crop(v)]))
        Image.fromarray(rgb).save(os.path.join(OUT, f"reorder_view{v:03d}.png"))
log(f"  均值 PSNR {np.mean(ps_r):.2f} dB  SSIM {np.mean(ss_r):.3f}  锐度 {np.mean(sp_r):.0f}")
if fs_r:
    log(f"  人脸（view 0/8/16）PSNR {np.mean(fs_r):.2f} dB  锐度 {np.mean(fsp_r):.0f}")
rep["cases"]["2dgs_reorder"] = dict(psnr=float(np.mean(ps_r)), ssim=float(np.mean(ss_r)),
                                    sharp=float(np.mean(sp_r)),
                                    face_psnr=float(np.mean(fs_r)) if fs_r else None,
                                    face_sharp=float(np.mean(fsp_r)) if fsp_r else None)
rep["cases"]["2dgs_reorder"].update(depth_consistency(DEP2R, "2dgs_reorder"))

# ② O2：world-consistent oracle（唯一法向）
log("\n" + "=" * 78)
log("② O2 world-consistent oracle：每点取各视角可见法向的平均")
log("=" * 78)
acc = np.zeros_like(M)
cnt = np.zeros(N0)
hit_cov = []
for v in range(V):
    n, hit = sample_normals(v, M)
    acc[hit] += n[hit]
    cnt[hit] += 1
    hit_cov.append(float(hit.mean()))
cov = cnt > 0
log(f"  单视角采样命中率 均值 {np.mean(hit_cov):.1%}（闸门 |Δz|/z < {GATE:.0%}）"
    f" ｜ 至少一处命中 {cov.mean():.1%} ｜ 平均命中视角数 {cnt[cov].mean():.2f}")

nO2 = np.copy(Rre[:, :, 2])                       # 兜底：重排法向
norm_acc = np.linalg.norm(acc, axis=1)            # 全长 N，后面按 good 取子集
good = cov & (norm_acc > 1e-9)
nO2[good] = acc[good] / norm_acc[good][:, None]
R2o, n_bad_gs = set_normal(Rre, nO2)
q_o2 = mat_to_quat(R2o).astype(np.float32)
log(f"  法向来源：oracle {good.mean():.1%} ｜ 兜底重排法向 {1 - good.mean():.1%}"
    f" ｜ 面内轴退化用 cross 兜底 {n_bad_gs:,}")
# 自证：构造出的第三列就是 nO2
assert np.allclose(quat_to_mat(q_o2.astype(np.float64))[:, :, 2],
                   nO2 / np.maximum(np.linalg.norm(nO2, axis=1, keepdims=True), 1e-12),
                   atol=1e-4), "set_normal 没把第三列设成 oracle 法向"
# ⚠️ 夹角只能算在真正拿到 oracle 的点上：回落点的 nO2 就是重排法向，夹角恒为 0，
# 混进去会把中位数整体拉低（第一版就是这样算出虚低的 30.3°，p10=0.0 是它的指纹）。
u_o = nO2[good] / np.linalg.norm(nO2[good], axis=1, keepdims=True)
u_d = Rre[good][:, :, 2]
alt_g = np.degrees(np.arccos(np.clip(np.abs(np.sum(u_o * u_d, 1)), -1, 1)))
log(f"  oracle 法向 vs DA3 盘法向 夹角（仅 oracle 点，n={good.sum():,}）："
    f"中位 {np.median(alt_g):.1f}°  p10 {np.percentile(alt_g, 10):.1f}°  "
    f"p90 {np.percentile(alt_g, 90):.1f}°  ｜ <10° 占 {(alt_g < 10).mean():.1%}")
log("    （I-03 用 audit_12 口径读到 56.9~66.4°；这两个数口径不同，见 ERRATA E12）")
rep["oracle"] = dict(hit_cov=float(np.mean(hit_cov)), any_hit=float(cov.mean()),
                     used_oracle=float(good.mean()),
                     angle_med=float(np.median(alt_g)),
                     angle_p10=float(np.percentile(alt_g, 10)),
                     angle_p90=float(np.percentile(alt_g, 90)),
                     angle_lt10=float((alt_g < 10).mean()),
                     cnt_mean=float(cnt[cov].mean()) if cov.any() else 0.0)

G2O = to_dev(dict(g0, scales=s_re, quats=q_o2))
DEP2O, ps_o, ss_o, sp_o, fs_o, fsp_o = {}, [], [], [], [], []
for v in view_list:
    rgb, a, dep, med, dist = render_2dgs(G2O, v, deg, "expected")
    DEP2O[v] = dep
    ps_o.append(psnr(rgb, GT[v]))
    ss_o.append(float(ssim(rgb, GT[v], channel_axis=-1, data_range=255)))
    sp_o.append(sharp(rgb))
    if v in (0, 8, 16):
        fs_o.append(psnr(rgb[face_crop(v)], GT[v][face_crop(v)]))
        fsp_o.append(sharp(rgb[face_crop(v)]))
        Image.fromarray(rgb).save(os.path.join(OUT, f"oracle2_view{v:03d}.png"))
log(f"  均值 PSNR {np.mean(ps_o):.2f} dB  SSIM {np.mean(ss_o):.3f}  锐度 {np.mean(sp_o):.0f}")
if fs_o:
    log(f"  人脸（view 0/8/16）PSNR {np.mean(fs_o):.2f} dB  锐度 {np.mean(fsp_o):.0f}")
rep["cases"]["2dgs_oracle"] = dict(psnr=float(np.mean(ps_o)), ssim=float(np.mean(ss_o)),
                                   sharp=float(np.mean(sp_o)),
                                   face_psnr=float(np.mean(fs_o)) if fs_o else None,
                                   face_sharp=float(np.mean(fsp_o)) if fsp_o else None)
rep["cases"]["2dgs_oracle"].update(depth_consistency(DEP2O, "o2_oracle"))

# ③ O1：per-view oracle（上界；每视角用该视角自己的 GT 法向）
log("\n" + "=" * 78)
log("③ O1 per-view oracle：渲染各视角时用该视角自己的 GT 法向（上界）")
log("=" * 78)
ps_1, sp_1, fs_1, fsp_1, hits = [], [], [], [], []
for v in view_list:
    n, hit = sample_normals(v, M)
    hits.append(float(hit.mean()))
    Rv, _ = set_normal(Rre, n)
    qv = mat_to_quat(Rv).astype(np.float32)
    Gv = to_dev(dict(g0, scales=s_re, quats=qv))
    rgb, a, dep, med, dist = render_2dgs(Gv, v, deg, "expected")
    ps_1.append(psnr(rgb, GT[v]))
    sp_1.append(sharp(rgb))
    if v in (0, 8, 16):
        fs_1.append(psnr(rgb[face_crop(v)], GT[v][face_crop(v)]))
        fsp_1.append(sharp(rgb[face_crop(v)]))
        Image.fromarray(rgb).save(os.path.join(OUT, f"oracle1_view{v:03d}.png"))
    del Gv
log(f"  均值 PSNR {np.mean(ps_1):.2f} dB（逐视角命中率均值 {np.mean(hits):.1%}）"
    f"  锐度 {np.mean(sp_1):.0f}")
if fs_1:
    log(f"  人脸（view 0/8/16）PSNR {np.mean(fs_1):.2f} dB  锐度 {np.mean(fsp_1):.0f}")
rep["cases"]["2dgs_oracle_perview"] = dict(
    psnr=float(np.mean(ps_1)), sharp=float(np.mean(sp_1)), hit=float(np.mean(hits)),
    face_psnr=float(np.mean(fs_1)) if fs_1 else None,
    face_sharp=float(np.mean(fsp_1)) if fsp_1 else None)

# ── 判据 ─────────────────────────────────────────────────────────────────────
log("\n" + "=" * 78)
log("判据（IMPROVEMENTS.md §3.5.2，跑前写死）")
log("=" * 78)
c3 = rep["cases"]["3dgs"]
c2r = rep["cases"]["2dgs_oracle"]
c2re = rep["cases"]["2dgs_reorder"]


def rel(a, b):
    return (a - b) / abs(b) * 100.0


J = dict(
    J_psnr_gap=round(float(c3["psnr"] - c2r["psnr"]), 3),
    J_psnr_pass=bool(c2r["psnr"] >= c3["psnr"] - 0.15),
    J_face_gap=round(float((c3["face_psnr"] or 0) - (c2r["face_psnr"] or 0)), 3),
    J_face_pass=bool((c2r["face_psnr"] or 0) >= (c3["face_psnr"] or 0) - 0.15),
    J_sharp=round(float(c2r["sharp"] - c3["sharp"]), 1),
    J_face_sharp=round(float((c2r["face_sharp"] or 0) - (c3["face_sharp"] or 0)), 1),
    J_oracle_vs_reorder=round(float(c2r["psnr"] - c2re["psnr"]), 3),
    J_face_oracle_vs_reorder=round(float((c2r["face_psnr"] or 0) - (c2re["face_psnr"] or 0)), 3),
    J_sharp_oracle_vs_reorder=round(float(c2r["sharp"] - c2re["sharp"]), 1),
    J_reorder_gap=round(float(c3["psnr"] - c2re["psnr"]), 3),
)
J["verdict"] = ("H2 成立：法向是瓶颈，I-04 值得做"
                if J["J_psnr_pass"] and J["J_face_pass"]
                else "H2 部分成立：PSNR 追平但清晰度仍差"
                if J["J_psnr_pass"]
                else "H2 证伪：给了正确法向仍不如 3DGS → 2DGS 路线到此为止")
rep["judge"] = J

log(f"  对照：2DGS reorder（I-03 本体）  全图 {c2re['psnr']:.2f} dB ｜ "
    f"人脸 {(c2re['face_psnr'] or 0):.2f} ｜ 锐度 {c2re['sharp']:.0f}")
log(f"  J-PSNR 全图：3DGS {c3['psnr']:.2f} → oracle {c2r['psnr']:.2f} dB"
    f"（差 {J['J_psnr_gap']:+.2f}，阈值 ≥−0.15）{'✅' if J['J_psnr_pass'] else '❌'}")
log(f"  J-FACE 人脸：3DGS {(c3['face_psnr'] or 0):.2f} → oracle {(c2r['face_psnr'] or 0):.2f} dB"
    f"（差 {J['J_face_gap']:+.2f}，阈值 ≥−0.15）{'✅' if J['J_face_pass'] else '❌'}")
log(f"  法向的净贡献（oracle − reorder）：全图 {J['J_oracle_vs_reorder']:+.2f} dB ｜ "
    f"人脸 {J['J_face_oracle_vs_reorder']:+.2f} dB ｜ 锐度 {J['J_sharp_oracle_vs_reorder']:+.0f}")
log(f"  只报不判：锐度 {J['J_sharp']:+.0f} ｜ 人脸锐度 {J['J_face_sharp']:+.0f}")
log(f"\n  ⇒ 判定：{J['verdict']}")

with open(os.path.join(OUT, "i03b_oracle.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(rep, f, indent=1, ensure_ascii=False, default=float)
np.save(os.path.join(OUT, "o2_normals.npy"), nO2.astype(np.float32))
log(f"\n💾 {os.path.join(OUT, 'i03b_oracle.json')}")
