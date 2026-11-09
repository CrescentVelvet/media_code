#!/usr/bin/env python3
"""DA3-GS 是否「藏」着同样的重影：只改 opacity 归一化做交叉验证。

观察到的差异（audit_26）
------------------------
  DA3-GS      opacity 中位 0.033，88.2% < 0.1
  InfiniSplat opacity 中位 0.180，仅 30.2% < 0.1，另有 6.1% ≥ 0.95
两者点数几乎一样（610 万 / 613 万），都是「每像素一个高斯 × 32 视角」。
0.033 ≈ 0.95/32：**DA3 官方等价导出把每份 opacity 压到 1/32，靠 32 层叠加还原**。
若几何真的一致，压 1/32 只是变淡；若几何有分层，压 1/32 会把分层「雾化」成糊。

交叉实验（同一插值视角、同一内参）
----------------------------------
  A  DA3 原样              → 已知：糊、单脸
  B  DA3 opacity ×32       → 若出现双脸 ⇒ 它本来就分层，只是被 1/32 平均掉了
  C  Infini 原样           → 已知：锐、双脸
  D  Infini opacity ÷32    → 若变糊、双脸消失 ⇒ 与 DA3 的差别就在这一步归一化
"""
import importlib.util
import os

import numpy as np
import torch
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "m24", "/mnt/c/code/media_code/compare_4methods/audit_24_five_way_video.py")
m24 = importlib.util.module_from_spec(SPEC)
os.environ["NFRAMES"] = "1"
SPEC.loader.exec_module(m24)
from gsplat import rasterization            # noqa: E402

CELLW, CELLH = 372, 496
OUT = os.environ.get("OUT", m24.A + "/orbit_video/_ghost")


def log(*a):
    print(*a, flush=True)


def render(g, w2c, K, scale=1.0, dev="cuda"):
    G = {k: torch.from_numpy(g[k]).to(dev) for k in ("means", "quats", "scales")}
    sh = torch.from_numpy(g["sh"]).to(dev)
    op = torch.from_numpy(np.clip(g["opac"] * scale, 0, 1).astype(np.float32)).to(dev)
    with torch.no_grad():
        img, al, _ = rasterization(G["means"], G["quats"], G["scales"], op, sh,
                                   torch.from_numpy(w2c).float().to(dev)[None],
                                   torch.from_numpy(K).float().to(dev)[None],
                                   CELLW, CELLH, sh_degree=g["deg"], packed=False,
                                   backgrounds=torch.ones(3, device=dev)[None],
                                   render_mode="RGB+ED")
    return (img[0, :, :, :3].clamp(0, 1).cpu().numpy(),
            float(al[0, :, :, 0].median()))


def main():
    os.makedirs(OUT, exist_ok=True)
    C, d, up = m24.read_cams(*m24.CASES[0]["cams"])
    orb = m24.fit_orbit(C, d, up)
    K0 = np.load(m24.A + "/cameras.npz")["intrinsics"].astype(np.float64)[0]
    K = K0.copy()
    K[0, 0] *= CELLW / 378.0
    K[1, 1] *= CELLH / 504.0
    K[0, 2], K[1, 2] = (CELLW - 1) / 2.0, (CELLH - 1) / 2.0

    ths = orb["th"]
    dif = np.diff(np.concatenate([ths, ths[:1] + 2 * np.pi]))
    k = int(np.argmax(dif))
    pick = (k + 1 + len(ths) // 3) % len(ths)
    a0 = float(ths[pick])
    a1 = float(ths[(pick + 1) % len(ths)]) + (2 * np.pi if pick + 1 >= len(ths) else 0.0)
    mid = m24.ring_interp(orb, np.array([0.5 * (a0 + a1)]))[0]
    w2c = m24.look_at(mid, orb["target"], orb["axis"])
    log("🎥 插值视角 %.1f°" % np.degrees(0.5 * (a0 + a1)))

    # 每条 PLY 的段长 = 单视角像素数，用来数「同一位置叠了几个视角」
    plan = [("DA3-GS", 378 * 504, [("原样", 1.0), ("opacity×32", 32.0)]),
            ("InfiniSplat v2", 392 * 518, [("原样", 1.0), ("opacity÷32", 1 / 32.0)])]

    tiles = []
    for tag, seg_px, modes in plan:
        c = [x for x in m24.CASES if x["name"] == tag][0]
        g = m24.read_ply(c["ply"])
        Cx, _, _ = m24.read_cams(*c["cams"])
        s, R, t = (1.0, np.eye(3), np.zeros(3)) if tag == "DA3-GS" else m24.umeyama(Cx, C)
        g = m24.warp(g, s, R, t)
        log("\n▶ %s  %d 点（%d 段 × %d 像素/段）" % (tag, g["n"], g["n"] // seg_px, seg_px))

        # 同一 1 cm 体素里有几个不同段
        key = np.floor(g["means"] / 0.01).astype(np.int64)
        key -= key.min(0)
        lin = (key[:, 0] * (key[:, 1].max() + 1) + key[:, 1]) * (key[:, 2].max() + 1) + key[:, 2]
        seg = np.arange(g["n"]) // seg_px
        _, inv = np.unique(lin, return_inverse=True)
        nseg = np.zeros(len(g["means"]), np.int32)
        np.maximum.at(nseg, inv, seg + 1)                    # 记录该体素出现过的最大段号
        # 用「段号种类数」太贵，改统计每体素的点数与不同段的存在性
        uniq_seg = np.zeros_like(nseg, dtype=np.int32)
        order = np.argsort(inv * 100 + seg)
        same = np.concatenate([[False], (inv[order][1:] == inv[order][:-1]) &
                               (seg[order][1:] != seg[order][:-1])])
        uniq_seg[order] = same
        cnt_v = np.bincount(inv)
        multi = np.zeros(len(cnt_v), bool)
        multi[inv[order][same]] = True
        log("   1 cm 体素 %d 个 ｜ 含 ≥2 个不同视角的体素 %.1f%% ｜ 每体素点数 中位 %.1f"
            % (len(cnt_v), 100 * multi.mean(), np.median(cnt_v)))

        for name, sc in modes:
            img, amed = render(g, w2c, K, sc)
            log("   %-12s α 中位 %.4f ｜ 亮度均值 %.3f" % (name, amed, img.mean()))
            tiles.append((tag + " " + name, img))
        del g
        torch.cuda.empty_cache()

    # 拼图：2×2，头部区域裁出来再放大
    BX, BY = (110, 300), (40, 200)
    S = 3
    heads = []
    for name, img in tiles:
        c = img[BY[0]:BY[1], BX[0]:BX[1]]
        im = Image.fromarray((c * 255).astype(np.uint8)).resize(
            ((BX[1] - BX[0]) * S, (BY[1] - BY[0]) * S), Image.LANCZOS)
        heads.append(im)
    w, h = heads[0].size
    sheet = Image.new("RGB", (w * 2 + 8, h * 2 + 8), "white")
    for i, im in enumerate(heads):
        sheet.paste(im, ((i % 2) * (w + 8), (i // 2) * (h + 8)))
    fn = os.path.join(OUT, "opacity_cross.png")
    sheet.save(fn)
    log("\n🖼  %s" % fn)
    log("   左上 DA3 原样 ｜ 右上 DA3 opacity×32 ｜ 左下 Infini 原样 ｜ 右下 Infini ÷32")


if __name__ == "__main__":
    main()
