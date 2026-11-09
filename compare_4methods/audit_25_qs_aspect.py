#!/usr/bin/env python3
"""QuerySplat 的 3:4 口径：是不是「非等比 resize」把几何压扁了？

背景
----
fig35 里 QuerySplat 那一格是 512×512 正方形，与其它列的 3:4 不一致。
它的输入帧（input_frames/*_preprocessed.png）实测是 512×512，而原帧是 720×960，
且 K 给的是 fx≈fy≈494（cx=cy=256）——「方形像素 + 方形视场」。
若预处理是非等比 resize，几何就会横向被压，正方形渲染看着「矮胖」。

判据（三选一，谁 PSNR 高谁就是对的，三视角各投一次）
  A 原样   512² 画布，fx=fy=494.4        ↔ GT resize 512²
  B 压窄   384×512 画布，fx=0.75·494.4   ↔ GT resize 384×512
  C 裁切   384×512 画布，fx=fy=494.4     ↔ GT 中心裁 3:4 再 resize 384×512

产出：<QS>/rendered_34/render_view{n}.png（540×720，用胜出方案）
      <QS>/rendered_34/psnr_check.json、<QS>/rendered_34/diag.png
"""
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from gsplat import rasterization

Q = "/home/velvet/output/querysplat_results/佳佳抱枕_paper"
GT = "/home/velvet/output/vggt_omega_results/佳佳抱枕/04_model_3dgs/train/ours_30000/gt"
OUT = Q + "/rendered_34"
VIEWS = [0, 8, 16]
FONT_R = "/mnt/c/Windows/Fonts/msyh.ttc"

SCHEMES = [
    dict(tag="A 原样 512²", W=512, H=512, kx=1.0, ky=1.0,
         gt=lambda im: im.resize((512, 512), Image.LANCZOS)),
    dict(tag="B 只压横 384×512", W=384, H=512, kx=0.75, ky=1.0,
         gt=lambda im: im.resize((384, 512), Image.LANCZOS)),
    dict(tag="C 裁切 384×512", W=384, H=512, kx=1.0, ky=1.0,
         gt=lambda im: _center_crop(im, 3 / 4).resize((384, 512), Image.LANCZOS)),
    # D：焦距等比缩 0.75 放进 3:4 画布 —— 像素仍方形，垂直视场自然变宽 33%
    dict(tag="D 等比 384×512", W=384, H=512, kx=0.75, ky=0.75,
         gt=lambda im: im.resize((384, 512), Image.LANCZOS)),
]


def _center_crop(im, ratio):
    w, h = im.size
    if w / h > ratio:
        nw = int(round(h * ratio))
        return im.crop(((w - nw) // 2, 0, (w + nw) // 2, h))
    nh = int(round(w / ratio))
    return im.crop((0, (h - nh) // 2, w, (h + nh) // 2))


def read_ply(path):
    import re
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
        raw = np.fromfile(f, dtype=dt, count=n)
    means = np.stack([raw["x"], raw["y"], raw["z"]], 1).astype(np.float32)
    scales = np.exp(np.stack([raw["scale_%d" % i] for i in range(3)], 1)).astype(np.float32)
    opac = (1.0 / (1.0 + np.exp(-raw["opacity"].astype(np.float64)))).astype(np.float32)
    quats = np.stack([raw["rot_%d" % i] for i in range(4)], 1).astype(np.float32)
    f_dc = np.stack([raw["f_dc_%d" % i] for i in range(3)], 1).astype(np.float32)
    rest = ["f_rest_%d" % i for i in range(len(names)) if "f_rest_%d" % i in names]
    k = len(rest) // 3
    if k:
        fr = np.stack([raw[nm] for nm in rest], 1).reshape(n, 3, k).transpose(0, 2, 1)
        sh = np.concatenate([f_dc[:, None, :], fr], 1)
    else:
        sh = f_dc[:, None, :]
    return dict(means=means, scales=scales, opac=opac, quats=quats, sh=sh, n=int(n),
                deg=int(round((1 + k) ** 0.5)) - 1)


def psnr(a, b):
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 99.0 if mse <= 1e-12 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"
    g = read_ply(Q + "/gaussians.ply")
    d = np.load(Q + "/predicted_input_cameras.npz")
    w2c = d["w2c"].astype(np.float64)
    K0 = d["K"].astype(np.float64)[0]
    print("QS %d 点 SH%d ｜ K0 fx %.3f fy %.3f cx %.3f cy %.3f"
          % (g["n"], g["deg"], K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]))
    G = {k: torch.from_numpy(g[k]).to(dev) for k in ("means", "quats", "scales", "opac", "sh")}
    bg = torch.ones(3, device=dev)

    res = {}
    for sc in SCHEMES:
        ps = []
        for v in VIEWS:
            K = K0.copy()
            K[0, 0] *= sc["kx"]
            K[1, 1] *= sc["ky"]
            K[0, 2] = (sc["W"] - 1) / 2.0
            K[1, 2] = (sc["H"] - 1) / 2.0
            with torch.no_grad():
                img, _, _ = rasterization(
                    G["means"], G["quats"], G["scales"], G["opac"], G["sh"],
                    torch.from_numpy(w2c[v]).float().to(dev)[None],
                    torch.from_numpy(K).float().to(dev)[None],
                    sc["W"], sc["H"], sh_degree=g["deg"], packed=False,
                    backgrounds=bg[None], render_mode="RGB")
            out = img[0].clamp(0, 1).mul(255).byte().cpu().numpy()
            gt = np.array(sc["gt"](Image.open(os.path.join(GT, "%05d.png" % v)).convert("RGB")))
            ps.append(psnr(out, gt))
        res[sc["tag"]] = ps
        print("  %-14s PSNR  " % sc["tag"] + "  ".join("v%d %.2f" % (v, p) for v, p in zip(VIEWS, ps))
              + "  ｜ 均值 %.2f" % np.mean(ps))

    best = max(res, key=lambda k: float(np.mean(res[k])))
    print("\n🏆 胜出：%s（均值 %.2f dB）" % (best, np.mean(res[best])))
    sc = [s for s in SCHEMES if s["tag"] == best][0]

    # 用胜出方案出 540×720（3:4）的图给 fig35 —— 关键是**像素保持方形**：
    # 归一化焦距 f/宽 不变（QS 的 494.4/512 = 0.966，与 GT 的 1394.7/1439 = 0.969 几乎一致），
    # 所以 fx = fy = f_norm × 540，垂直视场自然从 54.8° 张到 69.3°（与 GT 的 68.6° 对齐）。
    f_norm = float(K0[0, 0]) / 512.0
    W, H = 540, 720
    K = np.array([[f_norm * W, 0, (W - 1) / 2.0],
                  [0, f_norm * W, (H - 1) / 2.0],
                  [0, 0, 1.0]])
    print("   出图 %d×%d ｜ fx=fy=%.1f（水平 fov %.1f°，垂直 %.1f°）"
          % (W, H, K[0, 0],
             2 * np.degrees(np.arctan(W / 2 / K[0, 0])),
             2 * np.degrees(np.arctan(H / 2 / K[1, 1]))))
    ims = {}
    for v in range(32):
        with torch.no_grad():
            img, _, _ = rasterization(
                G["means"], G["quats"], G["scales"], G["opac"], G["sh"],
                torch.from_numpy(w2c[v]).float().to(dev)[None],
                torch.from_numpy(K).float().to(dev)[None],
                W, H, sh_degree=g["deg"], packed=False, backgrounds=bg[None], render_mode="RGB")
        a = img[0].clamp(0, 1).mul(255).byte().cpu().numpy()
        Image.fromarray(a).save(os.path.join(OUT, "render_view%d.png" % v))
        if v in VIEWS:
            ims[v] = a
    print("   ✅ 写出 32 张 →", OUT)

    # 诊断拼图：上排 三方案渲染，下排 对应 GT
    cols = len(SCHEMES)
    hh = 300
    tiles = []
    for sc2 in SCHEMES:
        w = int(round(hh * sc2["W"] / sc2["H"]))
        K = K0.copy()
        K[0, 0] *= sc2["kx"]
        K[1, 1] *= sc2["ky"]
        K[0, 2] = (sc2["W"] - 1) / 2.0
        K[1, 2] = (sc2["H"] - 1) / 2.0
        with torch.no_grad():
            img, _, _ = rasterization(
                G["means"], G["quats"], G["scales"], G["opac"], G["sh"],
                torch.from_numpy(w2c[0]).float().to(dev)[None],
                torch.from_numpy(K).float().to(dev)[None],
                sc2["W"], sc2["H"], sh_degree=g["deg"], packed=False,
                backgrounds=bg[None], render_mode="RGB")
        out = Image.fromarray(img[0].clamp(0, 1).mul(255).byte().cpu().numpy())
        gt = sc2["gt"](Image.open(os.path.join(GT, "00000.png")).convert("RGB"))
        tiles.append((sc2["tag"], out.resize((w, hh), Image.LANCZOS),
                      gt.resize((w, hh), Image.LANCZOS), w))
    Wc = sum(t[3] for t in tiles) + 12 * (cols + 1)
    canvas = Image.new("RGB", (Wc, hh * 2 + 90), "white")
    dr = ImageDraw.Draw(canvas)
    f = ImageFont.truetype(FONT_R, 18)
    x = 12
    for tag, out, gt, w in tiles:
        dr.text((x, 8), tag, font=f, fill="#333")
        canvas.paste(out, (x, 40))
        canvas.paste(gt, (x, 40 + hh + 8))
        x += w + 12
    canvas.save(os.path.join(OUT, "diag.png"))
    json.dump({k2: [float(x) for x in v] for k2, v in res.items()},
              open(os.path.join(OUT, "psnr_check.json"), "w"), indent=2)
    print("   ✅ 诊断图 →", os.path.join(OUT, "diag.png"))


if __name__ == "__main__":
    main()
