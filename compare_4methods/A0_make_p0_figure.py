#!/usr/bin/env python3
"""P0 三项实验的汇总图（fig8）与数据重算。

四组数据全部来自 **REF_STRATEGY=first（参考视角 = 0）** 的同族运行，保证可比：
  A. 504 / 32v / DC-only       佳佳抱枕_ref0_r504
  B. 504 / 32v / full-SH       佳佳抱枕_ref0_sh（same forward as A）
  C. 1008 / 32v / DC-only      佳佳抱枕_ref0_r1008
  D. 504 / 50v / DC-only       佳佳抱枕_v50

左图：逐视角 PSNR 曲线（A）——说明 DA3-GS 的质量由"离参考视角多远"决定，不是随机视角不稳。
右图：三项干预的 ΔPSNR（相对各自 baseline）——都只有 ±0.3 dB 量级，
      唯一显著的正向是"参考视角 + full-SH +0.7 dB"，而它在别的视角是负的。

输出：compare_4methods/ppt_assets/fig8_p0_evidence.png + p0_summary.json（写到 WSL output）
"""
import json
from pathlib import Path

import cv2
import numpy as np
from matplotlib import font_manager
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

B = Path.home() / "output/da3_gs_results"
CAP32 = Path.home() / "output/da3_results/_captures/佳佳抱枕/images"
CAP50 = B / "_captures/佳佳抱枕_v50/images"
OUT = Path("/mnt/c/code/media_code/compare_4methods/ppt_assets")
OUT.mkdir(exist_ok=True)

FONT = "/mnt/c/Windows/Fonts/msyh.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

C_REF = "#2E8B7F"
C_SH = "#5B7FB5"
C_RES = "#C77C2E"
C_V = "#8A62AC"


def rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def scan(d):
    fs = sorted([p for p in d.iterdir() if p.suffix.lower() == ".png"])
    return fs, [p.stem.split("_")[1] for p in fs]


def sharp(r):
    return float(cv2.Laplacian(cv2.cvtColor(r, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def psnr(r, ref):
    mse = ((r.astype(np.float64) - ref.astype(np.float64)) ** 2).mean()
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def evaluate(render_dir, refs, ids):
    ps, ss, sh = {}, {}, {}
    for k, f in enumerate(refs):
        r = cv2.resize(rgb(Path(render_dir) / f"view_{k:03d}.png"), (378, 504), interpolation=cv2.INTER_AREA)
        ref = cv2.resize(rgb(f), (378, 504), interpolation=cv2.INTER_AREA)
        ps[ids[k]] = psnr(r, ref)
        ss[ids[k]] = float(ssim(ref, r, channel_axis=2, data_range=255))
        sh[ids[k]] = sharp(r)
    return {"psnr": ps, "ssim": ss, "sharp": sh}


f32, i32 = scan(CAP32)
f50, i50 = scan(CAP50)
common = [x for x in i32 if x in set(i50)]

runs = {
    "A_504_dc": evaluate(B / "佳佳抱枕_ref0_r504/render_dc_all", f32, i32),
    "B_504_fullsh": evaluate(B / "佳佳抱枕_ref0_sh/render_full_all", f32, i32),
    "C_1008_dc": evaluate(B / "佳佳抱枕_ref0_r1008/render_dc_all", f32, i32),
    "D_50v_dc": evaluate(B / "佳佳抱枕_v50/render_dc_all", f50, i50),
}


def mean_of(tag, keys=None):
    d = runs[tag]["psnr"]
    ks = keys if keys is not None else list(d)
    return float(np.mean([d[k] for k in ks]))


base = mean_of("A_504_dc")
others = [k for k in i32 if k != i32[0]]
deltas = {
    "SH rest\n@参考视角": mean_of("B_504_fullsh", [i32[0]]) - mean_of("A_504_dc", [i32[0]]),
    "SH rest\n@其余 31 视角": mean_of("B_504_fullsh", others) - mean_of("A_504_dc", others),
    "输入分辨率 ×2\n(504→1008)": mean_of("C_1008_dc") - base,
    "视角数 ×1.56\n(32→50)": mean_of("D_50v_dc", common) - mean_of("A_504_dc", common),
}

summary = {
    "note": "全部为 REF_STRATEGY=first（参考视角=0）；PSNR/SSIM 为与输入帧的自一致性，非 NVS holdout",
    "base_504_32v": {"psnr_mean": base,
                     "ssim_mean": float(np.mean(list(runs["A_504_dc"]["ssim"].values()))),
                     "sharp_mean": float(np.mean(list(runs["A_504_dc"]["sharp"].values()))),
                     "collapsed_views": int(sum(1 for v in runs["A_504_dc"]["sharp"].values() if v < 20)),
                     "psnr_view0": runs["A_504_dc"]["psnr"][i32[0]]},
    "deltas": {k.replace("\n", " "): round(v, 3) for k, v in deltas.items()},
    "per_view": {t: {k: round(v, 3) for k, v in runs[t]["psnr"].items()} for t in runs},
    "common_frames_32v_vs_50v": len(common),
}
(B / "p0_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary["deltas"], indent=2, ensure_ascii=False))
print("base 504/32v:", round(base, 2), "| common frames:", len(common))

# ---------------- 画图 ----------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 4.6), dpi=150,
                               gridspec_kw={"width_ratios": [1.25, 1]})

# 左：逐视角 PSNR，横轴用「相机中心到参考视角的三维距离」而不是序号
# 理由：这条轨迹是绕人一圈的弧，序号 21 在空间上离参考视角只有 ~0.16 m，
# 用序号画会出现"远处反而好"的假象；用真实距离才看得出单调关系。
cam = np.load(B / "佳佳抱枕_ref0_r504/cameras.npz")
e = cam["extrinsics"]
C = np.einsum("nij,nj->ni", -e[:, :3, :3].transpose(0, 2, 1), e[:, :3, 3])
dist = np.linalg.norm(C - C[0], axis=1)
order = np.argsort(dist)
py = np.array([runs["A_504_dc"]["psnr"][k] for k in i32])[order]
px = dist[order]
shs = np.array([runs["A_504_dc"]["sharp"][k] for k in i32])[order]

axL.plot(px, py, "-", lw=1.5, color=C_REF, zorder=2)
ok = shs >= 20
axL.scatter(px[ok], py[ok], s=34, color=C_REF, zorder=3, label="可读视角")
axL.scatter(px[~ok], py[~ok], s=42, color="#C0504D", marker="X", zorder=4,
            label=f"整片糊掉（锐度<20，{int((~ok).sum())}/32）")
axL.axhline(base, ls="--", lw=1, color="#999999")
axL.annotate(f"参考视角：{py[0]:.1f} dB", xy=(px[0], py[0]), xytext=(0.35, 19.6),
             fontsize=10.5, color=C_REF,
             arrowprops=dict(arrowstyle="->", color=C_REF, lw=1))
axL.annotate(f"均值 {base:.1f} dB", xy=(px[-1], base), xytext=(px[-1] - 0.9, 12.6),
             fontsize=9.5, color="#777777")
axL.legend(loc="upper right", fontsize=9, frameon=False)
axL.set_xlabel("相机中心到参考视角的三维距离（m）", fontsize=10.5)
axL.set_ylabel("PSNR vs 输入帧（dB）", fontsize=10.5)
axL.set_title("① 质量随「离参考视角的距离」单调衰减", fontsize=12.5, loc="left")
axL.set_ylim(6, 22)
axL.spines[["top", "right"]].set_visible(False)

# 右：三项干预的 ΔPSNR
labels = list(deltas.keys())
vals = [deltas[k] for k in labels]
cols = [C_SH, C_SH, C_RES, C_V]
yy = np.arange(len(labels))
bars = axR.barh(yy, vals, color=cols, height=0.55)
axR.axvline(0, color="#555555", lw=1)
for b, v in zip(bars, vals):
    off = 0.12 if v >= 0 else -0.12
    axR.text(v + off, b.get_y() + b.get_height() / 2, f"{v:+.2f} dB",
             va="center", ha="left" if v >= 0 else "right", fontsize=10.5)
axR.set_yticks(yy, labels, fontsize=10.5)
axR.invert_yaxis()
axR.set_xlim(-1.6, 1.4)
axR.set_xlabel("相对各自 baseline 的 ΔPSNR（dB）", fontsize=10.5)
axR.set_title("② 三项低成本干预，量级都在 ±0.3 dB", fontsize=12.5, loc="left")
axR.spines[["top", "right"]].set_visible(False)

fig.suptitle("P0 三项验证：外观容量 / 输入分辨率 / 视角密度，都不是当前瓶颈（佳佳抱枕 · 参考视角固定为 0）",
             fontsize=13)
fig.tight_layout()
out = OUT / "fig8_p0_evidence.png"
fig.savefig(out, bbox_inches="tight")
print("✅", out)
