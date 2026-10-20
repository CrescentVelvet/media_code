"""生成 GT / 渲染 / 差异 的放大对照图，直观展示细节丢失位置。"""
import os
import glob
import numpy as np
import cv2

ROOT = "/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou"
OUT = "/mnt/d/output/som_human_hand/diag"
os.makedirs(OUT, exist_ok=True)


def make_panel(nm, model, mask_dir, tag):
    rp = os.path.join(ROOT, model, "test/ours_10000/renders", nm)
    gp = os.path.join(ROOT, model, "test/ours_10000/gt", nm)
    r = cv2.imread(rp)
    g = cv2.imread(gp)
    if r is None or g is None:
        return
    # 用差异能量最大处定位（在 mask 内）
    d = np.abs(r.astype(np.float32) - g.astype(np.float32)).sum(2)
    h, w = d.shape
    bs = 64
    best, bx, by = -1, 0, 0
    for y in range(0, h - bs, 32):
        for x in range(0, w - bs, 32):
            v = d[y:y + bs, x:x + bs].mean()
            if v > best:
                best, bx, by = v, x, y
    # 以最差点为中心裁 320x320
    cx, cy = bx + bs // 2, by + bs // 2
    x0 = max(0, min(w - 320, cx - 160))
    y0 = max(0, min(h - 320, cy - 160))
    crop_g = g[y0:y0 + 320, x0:x0 + 320]
    crop_r = r[y0:y0 + 320, x0:x0 + 320]
    heat = cv2.applyColorMap(
        cv2.normalize(d[y0:y0 + 320, x0:x0 + 320], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET)
    row = np.hstack([crop_g, crop_r, heat])
    label = np.zeros((34, row.shape[1], 3), np.uint8)
    label[:] = (32, 32, 32)
    cv2.putText(label, "GT(left)  render(mid)  |diff|(right)   " + tag,
                (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    out = os.path.join(OUT, f"zoom_{tag}_{nm}")
    cv2.imwrite(out, np.vstack([label, row]))
    print("wrote", out)


def main():
    nm = "00000.png"
    make_panel(nm, "model_p07", "p07", "p07_10k")
    make_panel(nm, "model_colmap", "p07", "full_10k")
    # 拼一张三联总览
    files = sorted(glob.glob(os.path.join(OUT, "zoom_*00000.png")))
    if files:
        ims = [cv2.imread(f) for f in files]
        ims = [i for i in ims if i is not None and i.shape[0] == ims[0].shape[0]]
        if ims:
            allimg = np.vstack(ims)
            cv2.imwrite(os.path.join(OUT, "zoom_overview.png"), allimg)
            print("wrote", os.path.join(OUT, "zoom_overview.png"))


if __name__ == "__main__":
    main()
