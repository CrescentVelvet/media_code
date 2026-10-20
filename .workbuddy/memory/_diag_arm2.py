"""整图并排 + mask 内手臂区域放大。"""
import os
import numpy as np
import cv2

ROOT = "/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou"
OUT = "/mnt/d/output/som_human_hand/diag"
MASKS = os.path.join(ROOT, "colmap_refined/persons/p07/masks")


def load(model, idx):
    base = os.path.join(ROOT, model, "test/ours_10000")
    r = cv2.imread(os.path.join(base, "renders", f"{idx:05d}.png"))
    g = cv2.imread(os.path.join(base, "gt", f"{idx:05d}.png"))
    return r, g


def mask_for(fid):
    p = os.path.join(MASKS, f"{fid:05d}.png")
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return m


def arm_box(g, m):
    """在 mask 内找「垂下的手臂」：mask 的左右两侧边缘列带。"""
    mm = m > 127
    h, w = mm.shape
    ys, xs = np.where(mm)
    if ys.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    # 手臂 = mask 下半部分的左右两侧
    ymid = int(y0 + (y1 - y0) * 0.55)
    left = mm[ymid:y1, x0:int(x0 + (x1 - x0) * 0.3)]
    right = mm[ymid:y1, int(x1 - (x1 - x0) * 0.3):x1]
    side = "left" if left.sum() > right.sum() else "right"
    if side == "left":
        bx = (x0, int(x0 + (x1 - x0) * 0.35))
    else:
        bx = (int(x1 - (x1 - x0) * 0.35), x1)
    by = (int(y0 + (y1 - y0) * 0.45), int(y1 * 0.98))
    return bx, by


def main():
    panels = []
    for idx, fid in ((2, 16), (6, 48), (12, 96)):
        r_f, g = load("model_p07", idx)
        r_t, _ = load("model_p07_trim", idx)
        m = mask_for(fid)
        if g is None or r_f is None or r_t is None or m is None:
            print("skip", idx)
            continue
        m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        box = arm_box(g, m)
        if box is None:
            continue
        (x0, x1), (y0, y1) = box
        pw = x1 - x0
        ph = y1 - y0
        s = max(pw, ph)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        s = min(s + 40, g.shape[0] - 4)
        sx = max(0, min(g.shape[1] - s, cx - s // 2))
        sy = max(0, min(g.shape[0] - s, cy - s // 2))
        sl = np.s_[sy:sy + s, sx:sx + s]
        row = np.hstack([g[sl], r_f[sl], r_t[sl]])
        label = np.zeros((28, row.shape[1], 3), np.uint8)
        label[:] = (28, 28, 28)
        cv2.putText(label, f"fid={fid}  GT | full160 | trim105  (手臂区)",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        panels.append(np.vstack([label, row]))
    if panels:
        wmax = max(p.shape[1] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, 0, 0, wmax - p.shape[1],
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0)) for p in panels]
        out = np.vstack(panels)
        p = os.path.join(OUT, "arm_compare2.jpg")
        cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("wrote", p, out.shape)


if __name__ == "__main__":
    main()
