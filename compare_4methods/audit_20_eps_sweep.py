#!/usr/bin/env python3
"""AUDIT-20 · 关掉 I-03 留的未结项：2DGS 更糊是不是 eps2d 语义不同（最小足迹下限更大）？

台账 §3.5.1 明确留着这一条：「快验读到 2DGS 锐度 449→371，与 A1 预期相反，
正式跑完要单独查 eps2d 语义」。

做法：固定 2DGS 重排版（I-03 本体的几何），只扫 eps2d ∈ {0.3, 0.1, 0.01}；
3DGS 侧同样扫，确认「eps2d 对本档影响多大」是可比的口径。

判据（写死）：若 2DGS 在 eps2d=0.01 时 PSNR 追平 3DGS 的 22.11 − 0.15 以内，
则「更糊」主要来自最小足迹下限，而不是 ray-splat 求交本身。
"""
import json
import os
import re

import numpy as np
import torch
from gsplat import rasterization, rasterization_2dgs

SC = os.environ["SC"]
NPZ = os.environ["NPZ"]
PLY = os.environ.get("PLY", os.path.join(SC, "official_export/i12_fullpts_sh/0000.ply"))
OUT = os.environ.get("OUT", os.path.join(SC, "i03b_oracle"))
VIEWS = [int(x) for x in os.environ.get("VIEWS", "0,8,16,24").split(",")]
EPS = [float(x) for x in os.environ.get("EPS", "0.3,0.1,0.01").split(",")]
dev = "cuda"


def log(*a):
    print(*a, flush=True)


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
        text = head.decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", text).group(1))
        names = re.findall(r"^property float (\w+)$", text, re.M)
        dt = np.dtype([(nm, "<f4") for nm in names])
        f.seek(len(head))
        raw = np.fromfile(f, dtype=dt, count=n, offset=0)
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
                deg=int(round((1 + k) ** 0.5)) - 1)


def quat_mul(a, b):
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], 1)


d = np.load(NPZ)
W2C = d["extrinsics"].astype(np.float64)
if W2C.shape[-2] == 4:
    W2C = W2C[:, :3, :]
K = d["intrinsics"].astype(np.float64)
GT = d["processed_images"]
V, H, Wd = d["depth"].shape

g0 = read_ply(PLY)
Q_P = np.array([[0.5, 0.5, 0.5, 0.5]])
q_re = quat_mul(g0["quats"].astype(np.float64), Q_P)
q_re = (q_re / np.linalg.norm(q_re, axis=1, keepdims=True)).astype(np.float32)
s_re = np.stack([g0["scales"][:, 1], g0["scales"][:, 2], g0["scales"][:, 0]], 1).astype(np.float32)

FACE = (0.35, 0.105, 0.565, 0.335)
fbox = (int(FACE[0] * Wd), int(FACE[1] * H), int(FACE[2] * Wd), int(FACE[3] * H))
fmask = np.zeros((H, Wd), bool)
fmask[fbox[1]:fbox[3], fbox[0]:fbox[2]] = True


def td(g):
    return {k: torch.from_numpy(v).to(dev) for k, v in g.items()
            if k in ("means", "quats", "scales", "opac", "sh")}


G3 = td(g0)
GR = td(dict(g0, scales=s_re, quats=q_re))
deg = g0["deg"]


def vm_of(v):
    return torch.from_numpy(
        np.vstack([W2C[v], np.array([0, 0, 0, 1.0])]).astype(np.float32)).to(dev)


def psnr(a, b, m=None):
    if m is not None:
        a, b = a[m], b[m]
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def sharp(rgb):
    import cv2
    g = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def render(g, v, two_d, eps):
    with torch.no_grad():
        if two_d:
            out = rasterization_2dgs(
                g["means"], g["quats"], g["scales"], g["opac"], g["sh"],
                vm_of(v)[None].float(), torch.from_numpy(K[v]).float().to(dev)[None],
                Wd, H, sh_degree=deg, packed=False, render_mode="RGB+ED",
                eps2d=eps, depth_mode="expected")
            img = out[0]
        else:
            img, _, _ = rasterization(
                g["means"], g["quats"], g["scales"], g["opac"], g["sh"],
                vm_of(v)[None].float(), torch.from_numpy(K[v]).float().to(dev)[None],
                Wd, H, sh_degree=deg, packed=False, render_mode="RGB+ED", eps2d=eps)
    return img[0, ..., :3].clamp(0, 1).mul(255).byte().cpu().numpy()


log(f"{'档':>6} {'eps2d':>6} {'PSNR':>7} {'人脸PSNR':>9} {'锐度':>7} {'人脸锐度':>9}")
res = []
for tag, g, two in (("3dgs", G3, False), ("2dgs", GR, True)):
    for eps in EPS:
        ps, fs, sp, fsp = [], [], [], []
        for v in VIEWS:
            rgb = render(g, v, two, eps)
            ps.append(psnr(rgb, GT[v]))
            fs.append(psnr(rgb, GT[v], fmask))
            sp.append(sharp(rgb))
            fsp.append(sharp(rgb[fbox[1]:fbox[3], fbox[0]:fbox[2]]))
        row = dict(tag=tag, eps=eps, psnr=float(np.mean(ps)), face=float(np.mean(fs)),
                   sharp=float(np.mean(sp)), face_sharp=float(np.mean(fsp)))
        res.append(row)
        log(f"{tag:>6} {eps:>6.2f} {row['psnr']:>7.2f} {row['face']:>9.2f} "
            f"{row['sharp']:>7.0f} {row['face_sharp']:>9.0f}")

base = [r for r in res if r["tag"] == "3dgs" and abs(r["eps"] - 0.3) < 1e-9][0]
best2d = max([r for r in res if r["tag"] == "2dgs"], key=lambda r: r["psnr"])
gap = best2d["psnr"] - base["psnr"]
log(f"\n判据（写死）：2DGS 最好档（eps2d={best2d['eps']:.2f}）{best2d['psnr']:.2f} dB vs "
    f"3DGS 基线 {base['psnr']:.2f} dB → 差 {gap:+.2f}")
log(f"  ⇒ {'eps2d 是主因（追平到 0.15 以内）' if gap >= -0.15 else 'eps2d 不是主因'}")

with open(os.path.join(OUT, "i03b_eps_sweep.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(dict(rows=res, best_2dgs=best2d, base_3dgs=base, gap=gap), f,
              indent=1, ensure_ascii=False)
log(f"💾 {os.path.join(OUT, 'i03b_eps_sweep.json')}")
