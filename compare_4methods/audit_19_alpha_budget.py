#!/usr/bin/env python3
"""AUDIT-19 · 为什么 2DGS 更糊：每像素的叠加预算（α 累积）够不够？

I-03b 的读数：把法向换成正确答案后，PSNR 再降 0.43 dB、锐度再降 12。
法向不是瓶颈，那就要问：换成精确 ray-splat 求交后，**每像素拿到的覆盖变少了没有**。

已知约束（audit_9 系列）：每视角只有 0.79 个高斯/像素、单层 α 中位 0.036，
成品的 α=1.000 靠 32 视角叠加。也就是说这张图是"薄覆盖"合成的，
任何减少单层覆盖的做法都会直接表现为发虚。

本脚本量（同 PLY / 同相机 / 同视角）：
  A1  每视角的累计 α：前景中位 / 均值 / α>0.5 的像素占比
  A2  α<0.05 的"稀薄像素"占比（薄覆盖的直接体现）
  A3  三档对照：3DGS EWA ｜ 2DGS 重排 ｜ 2DGS oracle

判据（写死）：若 2DGS 的 A1 中位 < 3DGS 的 0.8 倍，则"覆盖变薄"成立，
2DGS 更糊是覆盖预算问题，不是法向/几何问题。

只读，不写产物（除 json）。
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


def mat_to_quat(R):
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


Q_P = np.array([[0.5, 0.5, 0.5, 0.5]])

d = np.load(NPZ)
W2C = d["extrinsics"].astype(np.float64)
if W2C.shape[-2] == 4:
    W2C = W2C[:, :3, :]
K = d["intrinsics"].astype(np.float64)
GTDEP = d["depth"].astype(np.float64)
V, H, Wd = GTDEP.shape

g0 = read_ply(PLY)
q_re = quat_mul(g0["quats"].astype(np.float64), Q_P)
q_re = (q_re / np.linalg.norm(q_re, axis=1, keepdims=True)).astype(np.float32)
s_re = np.stack([g0["scales"][:, 1], g0["scales"][:, 2], g0["scales"][:, 0]], 1).astype(np.float32)

nO2 = np.load(os.path.join(OUT, "o2_normals.npy")).astype(np.float64)
nO2 /= np.linalg.norm(nO2, axis=1, keepdims=True) + 1e-12
Rre = quat_to_mat(q_re.astype(np.float64))
c0, c1 = Rre[:, :, 0], Rre[:, :, 1]
t1 = c0 - np.sum(c0 * nO2, 1, keepdims=True) * nO2
bad = np.linalg.norm(t1, axis=1) < 0.2
t1[bad] = np.cross(nO2[bad], c1[bad])
t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
Rn = np.empty_like(Rre)
Rn[:, :, 0] = t1
Rn[:, :, 1] = np.cross(nO2, t1)
Rn[:, :, 2] = nO2
q_o2 = mat_to_quat(Rn).astype(np.float32)


def td(g):
    return {k: torch.from_numpy(v).to(dev) for k, v in g.items()
            if k in ("means", "quats", "scales", "opac", "sh")}


G3 = td(g0)
GR = td(dict(g0, scales=s_re, quats=q_re))
GO = td(dict(g0, scales=s_re, quats=q_o2))
deg = g0["deg"]


def vm_of(v):
    return torch.from_numpy(
        np.vstack([W2C[v], np.array([0, 0, 0, 1.0])]).astype(np.float32)).to(dev)


def alpha_of(g, v, two_d):
    with torch.no_grad():
        if two_d:
            out = rasterization_2dgs(
                g["means"], g["quats"], g["scales"], g["opac"], g["sh"],
                vm_of(v)[None].float(), torch.from_numpy(K[v]).float().to(dev)[None],
                Wd, H, sh_degree=deg, packed=False, render_mode="RGB+ED",
                depth_mode="expected")
            return out[1][0].squeeze(-1).cpu().numpy()
        _, a, _ = rasterization(
            g["means"], g["quats"], g["scales"], g["opac"], g["sh"],
            vm_of(v)[None].float(), torch.from_numpy(K[v]).float().to(dev)[None],
            Wd, H, sh_degree=deg, packed=False, render_mode="RGB+ED")
        return a[0].squeeze(-1).cpu().numpy()


rows = []
log(f"{'view':>5} {'档':>12} {'α中位':>8} {'α均值':>8} {'α>0.5':>7} {'α<0.05':>8}")
for v in VIEWS:
    fg = GTDEP[v] > 0
    for tag, g, two in (("3dgs", G3, False), ("2dgs_reorder", GR, True),
                        ("2dgs_oracle", GO, True)):
        a = alpha_of(g, v, two)[fg]
        row = dict(v=v, tag=tag, med=float(np.median(a)), mean=float(a.mean()),
                   gt05=float((a > 0.5).mean()), lt005=float((a < 0.05).mean()))
        rows.append(row)
        log(f"{v:>5} {tag:>12} {row['med']:>8.4f} {row['mean']:>8.4f} "
            f"{row['gt05']:>6.1%} {row['lt005']:>7.1%}")

log("\n汇总（各视角均值）")
agg = {}
for tag in ("3dgs", "2dgs_reorder", "2dgs_oracle"):
    sub = [r for r in rows if r["tag"] == tag]
    agg[tag] = dict(med=float(np.mean([r["med"] for r in sub])),
                    mean=float(np.mean([r["mean"] for r in sub])),
                    gt05=float(np.mean([r["gt05"] for r in sub])),
                    lt005=float(np.mean([r["lt005"] for r in sub])))
    log(f"  {tag:>12}: α中位 {agg[tag]['med']:.4f}  α均值 {agg[tag]['mean']:.4f}  "
        f"α>0.5 {agg[tag]['gt05']:.1%}  α<0.05 {agg[tag]['lt005']:.1%}")

ratio = agg["2dgs_reorder"]["mean"] / max(agg["3dgs"]["mean"], 1e-9)
ratio_o = agg["2dgs_oracle"]["mean"] / max(agg["3dgs"]["mean"], 1e-9)
log(f"\n判据（写死）：2DGS 的 α 均值 / 3DGS < 0.8 → 覆盖变薄成立")
log(f"  重排 {ratio:.3f} ｜ oracle {ratio_o:.3f} → "
    f"{'覆盖变薄成立' if ratio < 0.8 else '覆盖没有明显变薄（另有原因）'}")

res = dict(rows=rows, agg=agg, ratio_reorder=ratio, ratio_oracle=ratio_o,
           verdict=("覆盖变薄" if ratio < 0.8 else "覆盖未变薄"))
p = os.path.join(OUT, "i03b_alpha.json")
with open(p, "w", encoding="utf-8", newline="\n") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
log(f"💾 {p}")
