"""渲染 vs GT 锐度对比：量化重建在手部额外丢失了多少细节。"""
import os
import glob
import numpy as np
import cv2

ROOT = "/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou"


def sharp_of(gray, m):
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    v = lap[m]
    return float(np.log10(np.var(v) + 1e-6)), float(np.percentile(np.abs(v), 99.0))


def block_report(gray_gt, gray_rd, m, bs=32, topk=8):
    """逐块 GT/渲染 锐度比，返回最差的块。"""
    h, w = gray_gt.shape
    lap_g = cv2.Laplacian(gray_gt, cv2.CV_32F, ksize=3)
    lap_r = cv2.Laplacian(gray_rd, cv2.CV_32F, ksize=3)
    rows = []
    for y in range(0, h - bs + 1, bs):
        for x in range(0, w - bs + 1, bs):
            mm = m[y:y + bs, x:x + bs]
            if mm.mean() < 0.6:
                continue
            vg = np.var(lap_g[y:y + bs, x:x + bs][mm]) + 1e-6
            vr = np.var(lap_r[y:y + bs, x:x + bs][mm]) + 1e-6
            rows.append((vr / vg, x, y, vg, vr))
    rows.sort()
    return rows[:topk], rows


def run_model(tag, renders, gts, masks):
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(renders, "*.png")))
    print(f"\n##### {tag}  n={len(names)} #####")
    agg = []
    for nm in names:
        rp = os.path.join(renders, nm)
        gp = os.path.join(gts, nm)
        mp = os.path.join(masks, nm)
        if not (os.path.exists(gp) and os.path.exists(mp)):
            continue
        r = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        g = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if r is None or g is None or m is None:
            continue
        if m.shape != g.shape:
            m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = m > 127
        if m.sum() < 500:
            continue
        sg, pg = sharp_of(g, m)
        sr, pr = sharp_of(r, m)
        h = g.shape[0]
        upper = m.copy()
        upper[int(h * 0.45):, :] = False
        lower = m.copy()
        lower[:int(h * 0.45), :] = False
        su_g, su_r = (sharp_of(g, upper)[0], sharp_of(r, upper)[0]) if upper.sum() > 500 else (np.nan, np.nan)
        sl_g, sl_r = (sharp_of(g, lower)[0], sharp_of(r, lower)[0]) if lower.sum() > 500 else (np.nan, np.nan)
        agg.append(dict(name=nm, sg=sg, sr=sr, ds=sr - sg, pg=pg, pr=pr,
                        su_g=su_g, su_r=su_r, sl_g=sl_g, sl_r=sl_r,
                        area=float(m.mean())))
        if nm == names[0]:
            worst, allrows = block_report(g, r, m)
            print(f"  [ex {nm}] 最差块 (渲染/GT 方差比, x, y):")
            for ratio, x, y, vg, vr in worst:
                print(f"    ratio={ratio:.3f} at ({x},{y}) gt_var={vg:.1f} rd_var={vr:.1f}")
            allr = np.array([r0[0] for r0 in allrows])
            print(f"  [ex {nm}] 块比值: med={np.median(allr):.3f} "
                  f"p10={np.percentile(allr,10):.3f} p90={np.percentile(allr,90):.3f}")

    if not agg:
        print("  no pairs")
        return
    ds = np.array([a["ds"] for a in agg])
    print(f"  全 mask log10sharp: GT med={np.median([a['sg'] for a in agg]):.4f} "
          f"RD med={np.median([a['sr'] for a in agg]):.4f}")
    print(f"  Δ(渲染-GT): med={np.median(ds):+.4f} "
          f"p10={np.percentile(ds,10):+.4f} p90={np.percentile(ds,90):+.4f} "
          f"(负=渲染更糊; -0.1 ≈ 细节能量剩 79%, -0.3 ≈ 50%)")
    print(f"  线性比值 渲染/GT 方差: med={np.median([10**(2*a['ds']) for a in agg]):.3f}")
    sgd = np.array([a["su_r"] - a["su_g"] for a in agg if np.isfinite(a["su_g"])])
    sld = np.array([a["sl_r"] - a["sl_g"] for a in agg if np.isfinite(a["sl_g"])])
    if sgd.size:
        print(f"  上半身 Δ med={np.median(sgd):+.4f} | 下半(含手) Δ med={np.median(sld):+.4f}")


def main():
    pairs = [
        ("p02", os.path.join(ROOT, "model_p02/test/ours_10000/renders"),
         os.path.join(ROOT, "model_p02/test/ours_10000/gt"),
         os.path.join(ROOT, "colmap_refined/persons/p02/masks")),
        ("p07", os.path.join(ROOT, "model_p07/test/ours_10000/renders"),
         os.path.join(ROOT, "model_p07/test/ours_10000/gt"),
         os.path.join(ROOT, "colmap_refined/persons/p07/masks")),
        ("colmap_full", os.path.join(ROOT, "model_colmap/test/ours_10000/renders"),
         os.path.join(ROOT, "model_colmap/test/ours_10000/gt"),
         os.path.join(ROOT, "colmap_refined/persons/p07/masks")),
    ]
    for tag, rd, gt, mk in pairs:
        if os.path.isdir(rd):
            run_model(tag, rd, gt, mk)
        else:
            print(f"\n##### {tag} missing: {rd}")


if __name__ == "__main__":
    main()
