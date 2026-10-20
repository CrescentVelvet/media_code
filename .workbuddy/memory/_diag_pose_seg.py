"""量化「抬手段 vs 垂手段」的分布，以及相机方位覆盖，判断删帧代价。"""
import os
import json
import glob
import numpy as np
import cv2

ROOT = "/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou"


def load_cams(path):
    with open(path, encoding="utf-8") as f:
        cams = json.load(f)
    return sorted(cams, key=lambda c: c["img_name"])


def main():
    for pid in ("p02", "p07"):
        mdir = os.path.join(ROOT, "colmap_refined", "persons", pid, "masks")
        mp = sorted(glob.glob(os.path.join(mdir, "*.png")))
        cj = os.path.join(ROOT, f"model_{pid}", "cameras.json")
        cams = load_cams(cj) if os.path.exists(cj) else []
        print(f"\n===== {pid}  n={len(mp)}  cams={len(cams)} =====")
        if not mp:
            continue

        rows = []
        for i, p in enumerate(mp):
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            mm = m > 127
            h, w = mm.shape
            area = mm.mean()
            ys, xs = np.where(mm)
            if ys.size < 100:
                rows.append(dict(i=i, area=area, up=0.0, width=0.0, top=1.0, cx=0.5))
                continue
            top = ys.min() / h
            width = (xs.max() - xs.min()) / w
            # 上半身带（头肩以上区域）的横向宽度 —— 抬手时明显变宽
            band = mm[int(h * 0.15):int(h * 0.42), :]
            upw = (band.any(0).sum() / w) if band.size else 0.0
            rows.append(dict(i=i, area=float(area), up=float(upw),
                             width=float(width), top=float(top),
                             cx=float(xs.mean() / w)))

        if not rows:
            continue
        area = np.array([r["area"] for r in rows])
        up = np.array([r["up"] for r in rows])
        top = np.array([r["top"] for r in rows])

        # 分段：按上半带宽度（抬手 = 宽）做 1D 二聚类（大津法近似）
        v = up.copy()
        best_t, best_s = None, -1
        for t in np.linspace(v.min(), v.max(), 100):
            a = v[v <= t]
            b = v[v > t]
            if len(a) < 5 or len(b) < 5:
                continue
            s = len(a) * len(b) * (a.mean() - b.mean()) ** 2
            if s > best_s:
                best_s, best_t = s, t
        hi = v > best_t
        print(f"  上半带宽度: min={v.min():.3f} med={np.median(v):.3f} max={v.max():.3f}")
        print(f"  姿态二分段阈值={best_t:.3f} → 抬手类 {int(hi.sum())} 帧 / 垂手类 {int((~hi).sum())} 帧")
        idx_hi = np.where(hi)[0]
        idx_lo = np.where(~hi)[0]
        print(f"  抬手类帧号范围: {idx_hi.min() if idx_hi.size else '-'}~{idx_hi.max() if idx_hi.size else '-'}"
              f"  中位 {int(np.median(idx_hi)) if idx_hi.size else '-'}")
        print(f"  垂手类帧号范围: {idx_lo.min() if idx_lo.size else '-'}~{idx_lo.max() if idx_lo.size else '-'}"
              f"  中位 {int(np.median(idx_lo)) if idx_lo.size else '-'}")
        # 连续性
        def runs(mask_arr):
            out, s = [], None
            for i, f in enumerate(mask_arr):
                if f and s is None:
                    s = i
                elif not f and s is not None:
                    out.append((s, i - 1)); s = None
            if s is not None:
                out.append((s, len(mask_arr) - 1))
            return out
        print(f"  抬手类连续段: {runs(hi)[:8]}")
        print(f"  垂手类连续段: {runs(~hi)[:8]}")

        # 相机方位覆盖
        if cams and len(cams) == len(rows):
            pos = np.array([c["position"] for c in cams], dtype=float)
            # 以人（mask 质心反投影不好做）近似用相机中心绕原点方位角
            az = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))
            az = np.unwrap(np.radians(az))
            az = np.degrees(az)
            span_all = az.max() - az.min()
            span_hi = (az[hi].max() - az[hi].min()) if hi.sum() > 1 else 0
            span_lo = (az[~hi].max() - az[~hi].min()) if (~hi).sum() > 1 else 0
            print(f"  相机方位角跨度: 全帧 {span_all:.1f}deg | 抬手段 {span_hi:.1f}deg "
                  f"| 垂手段 {span_lo:.1f}deg")

        out = "/tmp/pose_seg_%s.tsv" % pid
        with open(out, "w", encoding="utf-8") as f:
            f.write("idx\tarea\tupband\ttop\twidth\n")
            for r in rows:
                f.write(f"{r['i']}\t{r['area']:.5f}\t{r['up']:.4f}\t{r['top']:.4f}\t{r['width']:.4f}\n")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
