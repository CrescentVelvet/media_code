"""一次性诊断：手部运动模糊是不是输入里就有的？重建有没有额外损失？删帧能不能救？

指标（都在 person mask 内算）：
  1. area       mask 面积占比
  2. flow_p75   光流位移 75 分位（px/帧）——手速代理量
  3. sharp      拉普拉斯方差的对数（清晰度）
  4. diff       与上一帧的平均绝对光度差（观测一致性代理量）

同时读 COLMAP 位姿算相邻帧旋转角差，判断「删帧是否损失视角覆盖」。
"""
import os
import sys
import glob
import json
import numpy as np
import cv2

ROOT = "/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou"
IMAGES = os.path.join(ROOT, "colmap_scene", "images")
PERSONS = os.path.join(ROOT, "colmap_refined", "persons")


def list_images():
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths += glob.glob(os.path.join(IMAGES, ext))
    return sorted(paths), sorted(os.path.basename(p) for p in paths)


def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def mask_stats(gray, prev_gray, mask, prev_mask):
    m = mask > 127
    area = m.mean()
    if m.sum() < 50:
        return dict(area=area, flow_p75=np.nan, sharp=np.nan, diff=np.nan)

    # 只在 mask 包围盒里算，省 10 倍时间
    ys, xs = np.where(m)
    y0, y1 = max(0, ys.min() - 8), min(gray.shape[0], ys.max() + 9)
    x0, x1 = max(0, xs.min() - 8), min(gray.shape[1], xs.max() + 9)
    g = gray[y0:y1, x0:x1]
    mm = m[y0:y1, x0:x1]

    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    sharp = float(np.log10(np.var(lap[mm]) + 1e-6))

    flow_p75 = np.nan
    diff = np.nan
    if prev_gray is not None and prev_mask is not None:
        pg = prev_gray[y0:y1, x0:x1]
        pm = (prev_mask > 127)[y0:y1, x0:x1]
        both = mm & pm
        if both.sum() > 200:
            f = cv2.calcOpticalFlowFarneback(
                pg, g, None, 0.5, 4, 15, 3, 5, 1.2, 0)
            mag = np.linalg.norm(f, axis=2)
            flow_p75 = float(np.percentile(mag[both], 75))
            d = np.abs(g.astype(np.float32) - pg.astype(np.float32))
            diff = float(d[both].mean())
    return dict(area=area, flow_p75=flow_p75, sharp=sharp, diff=diff)


def read_cameras_bin(path):
    """极简 COLMAP cameras/images.bin 读取，只要每帧的 qvec。"""
    quats = []
    with open(path, "rb") as f:
        n = np.fromfile(f, np.uint64, 1)[0]
        for _ in range(int(n)):
            cam_id = np.fromfile(f, np.int32, 1)[0]
            model = np.fromfile(f, np.int32, 1)[0]
            w = np.fromfile(f, np.uint64, 1)[0]
            h = np.fromfile(f, np.uint64, 1)[0]
            np.fromfile(f, np.float64, 4)  # params
            _ = None
    return quats


def read_images_bin(path):
    """返回 [(name, qvec, tvec), ...]，COLMAP images.bin。

    每条记录：image_id(i32) qvec(f64x4) tvec(f64x3) cam_id(i32) name(\\0结尾)
              npoints(u64) 每点 32 字节（point3D_id i64 + xyz f64x3）
    """
    out = []
    with open(path, "rb") as f:
        n = int(np.fromfile(f, np.uint64, 1)[0])
        if n <= 0 or n > 1_000_000:
            raise ValueError(f"bad image count {n}")
        for _ in range(n):
            _img_id = np.fromfile(f, np.int32, 1)[0]
            q = np.fromfile(f, np.float64, 4)
            t = np.fromfile(f, np.float64, 3)
            _cam = np.fromfile(f, np.int32, 1)[0]
            chars = []
            while True:
                c = f.read(1)
                if c in (b"\x00", b""):
                    break
                chars.append(c.decode("utf-8", "ignore"))
            name = "".join(chars)
            npts = int(np.fromfile(f, np.uint64, 1)[0])
            if npts < 0 or npts > 50_000_000:
                raise ValueError(f"bad npoints {npts} for {name}")
            if npts:
                f.seek(32 * npts, 1)
            out.append((name, q, t))
    return out


def quat_angle(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    d = abs(float(np.dot(q1, q2)))
    d = min(1.0, max(-1.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


def main():
    paths, names = list_images()
    print(f"[io] images={len(paths)} dir={IMAGES}")
    if not paths:
        return

    # 位姿角差
    sparse = os.path.join(ROOT, "colmap_refined", "sparse_bin", "0")
    im_bin = os.path.join(sparse, "images.bin")
    ang = {}
    if os.path.exists(im_bin):
        try:
            recs = read_images_bin(im_bin)
            recs.sort(key=lambda r: r[0])
            recs = [r for r in recs if os.path.basename(r[0]) in set(names)]
            for i in range(1, len(recs)):
                ang[os.path.basename(recs[i][0])] = quat_angle(recs[i - 1][1], recs[i][1])
            deltas = list(ang.values())
            if deltas:
                print(f"[pose] 相邻帧旋转角差: median={np.median(deltas):.2f}deg "
                      f"p90={np.percentile(deltas, 90):.2f}deg max={max(deltas):.2f}deg")
                print(f"[pose] 相邻帧 t 差(仅报告): "
                      f"{np.median([np.linalg.norm(recs[i][2]-recs[i-1][2]) for i in range(1,len(recs))]):.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"[pose] 读取失败 {e}")

    pids = []
    if os.path.isdir(PERSONS):
        pids = sorted(d for d in os.listdir(PERSONS) if d.startswith("p"))
    print(f"[pid] {pids}")

    for pid in pids:
        mdir = os.path.join(PERSONS, pid, "masks")
        mpaths = sorted(glob.glob(os.path.join(mdir, "*.png")))
        print(f"\n===== {pid}: masks={len(mpaths)} =====")
        prev_gray = None
        prev_mask = None
        rows = []
        gray_cache = {}
        for p in paths:
            base = os.path.basename(p)
            stem = os.path.splitext(base)[0]
            mp = os.path.join(mdir, stem + ".png")
            if not os.path.exists(mp):
                continue
            img, gray = read_gray(p)
            if img is None:
                continue
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if mask.shape != gray.shape:
                mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            st = mask_stats(gray, prev_gray, mask, prev_mask)
            st["name"] = base
            st["ang"] = ang.get(base, np.nan)
            rows.append(st)
            prev_gray, prev_mask = gray, mask
            gray_cache[base] = gray.shape

        if not rows:
            print("  no paired mask/image")
            continue

        flow = np.array([r["flow_p75"] for r in rows], dtype=float)
        sharp = np.array([r["sharp"] for r in rows], dtype=float)
        area = np.array([r["area"] for r in rows], dtype=float)
        diff = np.array([r["diff"] for r in rows], dtype=float)
        angs = np.array([r["ang"] for r in rows], dtype=float)

        def stat(a, label):
            v = a[np.isfinite(a)]
            if v.size == 0:
                print(f"  {label}: n/a")
                return
            print(f"  {label}: min={v.min():.4f} p10={np.percentile(v,10):.4f} "
                  f"med={np.median(v):.4f} p90={np.percentile(v,90):.4f} max={v.max():.4f}")

        stat(flow, "flow_p75(px)")
        stat(sharp, "log10 sharp")
        stat(area, "area")
        stat(diff, "diff")
        stat(angs, "dAng(deg)")

        ok = np.isfinite(flow) & np.isfinite(sharp)
        if ok.sum() > 10:
            c = float(np.corrcoef(flow[ok], sharp[ok])[0, 1])
            print(f"  corr(flow, sharp) = {c:+.3f}  (负 = 动得越快越糊)")

        # 分档：按 flow 分三档，看 sharp
        if ok.sum() > 30:
            qs = np.percentile(flow[ok], [33, 66])
            bands = [(0, qs[0]), (qs[0], qs[1]), (qs[1], np.inf)]
            labels = ["静(低流速)", "中", "快(高流速)"]
            for (lo, hi), lb in zip(bands, labels):
                sel = ok & (flow >= lo) & (flow < hi)
                if sel.sum() > 3:
                    print(f"  {lb}: n={int(sel.sum())} flow_med={np.median(flow[sel]):.2f} "
                          f"sharp_med={np.median(sharp[sel]):.4f} "
                          f"dAng_med={np.nanmedian(angs[sel]):.2f}")

        # 锐度分位
        v = sharp[np.isfinite(sharp)]
        thr = np.percentile(v, 40)
        crisp = np.isfinite(sharp) & (sharp >= thr)
        print(f"  最锐 60% 帧: n={int(crisp.sum())} 占角覆盖 "
              f"dAng_sum={np.nansum(angs[crisp]):.1f}deg vs 全帧 {np.nansum(angs):.1f}deg")

        out = os.path.join("/tmp", f"diag_blur_{pid}.tsv")
        with open(out, "w", encoding="utf-8") as f:
            f.write("name\tarea\tflow_p75\tsharp\tdiff\tdang\n")
            for r in rows:
                f.write(f"{r['name']}\t{r['area']:.5f}\t{r['flow_p75']:.3f}\t"
                        f"{r['sharp']:.4f}\t{r['diff']:.3f}\t{r['ang']:.3f}\n")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
