"""mask 时序稳定性 + 相邻帧视角覆盖分析（为「删帧」提供代价估计）。"""
import os
import json
import glob
import numpy as np
import cv2

ROOT = "/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou"


def load_cams(path):
    with open(path, encoding="utf-8") as f:
        cams = json.load(f)
    cams = sorted(cams, key=lambda c: c["img_name"])
    return cams


def rot_angle_deg(R1, R2):
    R = np.array(R1) @ np.array(R2).T
    c = (np.trace(R) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def main():
    for pid in ("p02", "p07"):
        mdir = os.path.join(ROOT, "colmap_refined", "persons", pid, "masks")
        mp = sorted(glob.glob(os.path.join(mdir, "*.png")))
        cj = os.path.join(ROOT, f"model_{pid}", "cameras.json")
        print(f"\n===== {pid}: masks={len(mp)} =====")
        if not mp:
            continue
        names = [os.path.basename(p) for p in mp]
        cams = load_cams(cj) if os.path.exists(cj) else []
        cmap = {os.path.basename(c["img_name"]): c for c in cams}

        areas, cx, cy, ious = [], [], [], []
        prev = None
        for p, nm in zip(mp, names):
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            mm = m > 127
            areas.append(mm.sum() / mm.size)
            ys, xs = np.where(mm)
            cx.append(xs.mean() if xs.size else np.nan)
            cy.append(ys.mean() if ys.size else np.nan)
            if prev is not None:
                inter = (mm & prev).sum()
                uni = (mm | prev).sum()
                ious.append(inter / uni if uni else np.nan)
            prev = mm

        areas = np.array(areas)
        ious = np.array(ious)
        cx = np.array(cx)
        cy = np.array(cy)
        print(f"  area: med={np.median(areas):.4f} min={areas.min():.4f} max={areas.max():.4f} "
              f"相对离散={areas.std()/np.median(areas)*100:.1f}%")
        print(f"  时序 IoU: med={np.nanmedian(ious):.3f} p10={np.nanpercentile(ious,10):.3f} "
              f"min={np.nanmin(ious):.3f}")
        bad = np.where(np.nan_to_num(ious, nan=1.0) < 0.85)[0]
        print(f"  IoU<0.85 的帧对: {len(bad)}/{len(ious)}  (位置: {bad[:12].tolist()})")

        # 面积离群
        med = np.median(areas)
        out = np.where(np.abs(areas / med - 1.0) > 0.35)[0]
        print(f"  面积偏离中位 >35% 的帧: {len(out)}  (位置: {out[:12].tolist()})")

        # 相邻帧视角差
        if cams:
            angs = []
            for i in range(1, len(cams)):
                angs.append(rot_angle_deg(cams[i - 1]["rotation"], cams[i]["rotation"]))
            angs = np.array(angs)
            print(f"  相邻帧旋转角差: med={np.median(angs):.2f} p90={np.percentile(angs,90):.2f} "
                  f"max={angs.max():.2f} deg; 累计={angs.sum():.1f} deg")

            # 若只保留「最清晰 60%」会损失多少角度跨度
            sharp = []
            for p in mp:
                g = cv2.imread(p.replace("/masks/", "/masks/"), cv2.IMREAD_GRAYSCALE)
                sharp.append(0)
            # 用面积法不行，锐度在另一脚本已算；这里只报覆盖
            print(f"  （视角覆盖累计 {angs.sum():.1f} deg，帧均 {angs.mean():.2f} deg → "
                  f"每删 1 帧损失约 {angs.mean():.2f} deg 角覆盖）")
        else:
            print("  cameras.json 缺失")


if __name__ == "__main__":
    main()
