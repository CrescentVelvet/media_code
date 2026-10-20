"""运动区 vs 静止区 的细节损失对照。

要回答的问题：渲染糊，是「手在动」造成的，还是「3DGS 容量/过平滑」造成的？
做法：
  1. 把 test 帧的 GT 与原图序列匹配，拿到真实帧号（test 命名与序列号无关！）
  2. 用相邻原图差分定位运动区（手），mask 内其余为静止区
  3. 分别在两区算 log10(Laplacian 方差)：GT / 渲染 / Δ
若 运动区 Δ ≈ 静止区 Δ → 糊与运动无关，纯重建保真度问题。
"""
import os
import glob
import numpy as np
import cv2

ROOT = "/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou"
IMAGES = os.path.join(ROOT, "colmap_scene", "images")


def list_images():
    ps = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        ps += glob.glob(os.path.join(IMAGES, ext))
    return sorted(ps)


def small(gray, w=160):
    h = int(gray.shape[0] * w / gray.shape[1])
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)


def match_index(gt_small, seq_small):
    best, bi = 1e18, -1
    for i, s in enumerate(seq_small):
        if s.shape != gt_small.shape:
            continue
        d = float(np.mean((s - gt_small) ** 2))
        if d < best:
            best, bi = d, i
    return bi, best


def sharp_var(gray, m):
    if m.sum() < 300:
        return np.nan
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(np.log10(np.var(lap[m]) + 1e-6))


def run(tag, model_dir, mask_dir):
    renders = os.path.join(model_dir, "test/ours_10000/renders")
    gts = os.path.join(model_dir, "test/ours_10000/gt")
    if not os.path.isdir(renders):
        print(f"\n### {tag}: missing {renders}")
        return
    imgs = list_images()
    seq_small = [small(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in imgs]
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(renders, "*.png")))

    rows = []
    for nm in names:
        g = cv2.imread(os.path.join(gts, nm), cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(os.path.join(renders, nm), cv2.IMREAD_GRAYSCALE)
        if g is None or r is None:
            continue
        gi, err = match_index(small(g), seq_small)
        if gi < 0:
            continue
        mp = os.path.join(mask_dir, os.path.basename(imgs[gi]).rsplit(".", 1)[0] + ".png")
        if not os.path.exists(mp):
            continue
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != g.shape:
            m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = m > 127
        if m.sum() < 2000:
            continue

        # 运动区：前后帧差分（在 mask 内取高差分）
        prev = cv2.imread(imgs[max(0, gi - 1)], cv2.IMREAD_GRAYSCALE)
        nxt = cv2.imread(imgs[min(len(imgs) - 1, gi + 1)], cv2.IMREAD_GRAYSCALE)
        if prev is None or nxt is None:
            continue
        if prev.shape != g.shape:
            prev = cv2.resize(prev, (g.shape[1], g.shape[0]))
            nxt = cv2.resize(nxt, (g.shape[1], g.shape[0]))
        d = (np.abs(g.astype(np.float32) - prev.astype(np.float32)) +
             np.abs(nxt.astype(np.float32) - g.astype(np.float32))) * 0.5
        dv = d[m]
        thr = np.percentile(dv, 85)
        mov = m & (d > thr)
        # 静止区：mask 内低差分
        thr2 = np.percentile(dv, 25)
        sta = m & (d <= thr2)
        if mov.sum() < 300 or sta.sum() < 300:
            continue

        rows.append(dict(
            name=nm, frame=gi, err=err,
            gt_mov=sharp_var(g, mov), rd_mov=sharp_var(r, mov),
            gt_sta=sharp_var(g, sta), rd_sta=sharp_var(r, sta),
            mov_ratio=float(mov.sum() / m.sum()),
            mean_diff=float(dv.mean()),
        ))

    if not rows:
        print(f"\n### {tag}: no rows")
        return

    print(f"\n### {tag}  n={len(rows)}  (mask 内运动区占 {np.median([x['mov_ratio'] for x in rows])*100:.0f}% 面积)")
    for zone in ("mov", "sta"):
        g = np.array([x[f"gt_{zone}"] for x in rows])
        r = np.array([x[f"rd_{zone}"] for x in rows])
        dd = r - g
        print(f"  {zone}: GT={np.median(g):.4f} RD={np.median(r):.4f} "
              f"Δ={np.median(dd):+.4f} 剩余方差比={np.median(10**(2*dd)):.3f}")
    d_mov = np.array([x["rd_mov"] - x["gt_mov"] for x in rows])
    d_sta = np.array([x["rd_sta"] - x["gt_sta"] for x in rows])
    print(f"  Δ(运动区) - Δ(静止区) = {np.median(d_mov - d_sta):+.4f}  "
          f"(≈0 说明糊与运动无关; 明显为负说明运动区额外丢细节)")
    print(f"  GT 自身锐度: 运动区 {np.median([x['gt_mov'] for x in rows]):.4f} "
          f"vs 静止区 {np.median([x['gt_sta'] for x in rows]):.4f} "
          f"(差得越多 = 输入里运动区越糊)")
    print("  前 8 帧明细 frame/gt_mov/rd_mov/gt_sta/rd_sta:")
    for x in rows[:8]:
        print(f"    {x['frame']:3d}  {x['gt_mov']:.3f} {x['rd_mov']:.3f} "
              f"{x['gt_sta']:.3f} {x['rd_sta']:.3f}  diff={x['mean_diff']:.1f}")


def main():
    for tag, md in (("p02", "model_p02"), ("p07", "model_p07"), ("colmap_full", "model_colmap")):
        pid = "p07" if tag == "colmap_full" else tag
        run(tag, os.path.join(ROOT, md),
            os.path.join(ROOT, "colmap_refined", "persons", pid, "masks"))


if __name__ == "__main__":
    main()
