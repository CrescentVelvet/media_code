"""垂手段(105帧) vs 全段(160帧) 的手臂区域视觉对照。"""
import os
import numpy as np
import cv2

ROOT = "/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou"
OUT = "/mnt/d/output/som_human_hand/diag"
os.makedirs(OUT, exist_ok=True)


def grab(model, idx):
    base = os.path.join(ROOT, model, "test/ours_10000")
    r = cv2.imread(os.path.join(base, "renders", f"{idx:05d}.png"))
    g = cv2.imread(os.path.join(base, "gt", f"{idx:05d}.png"))
    return r, g


def crop_pair(g, full, trim, tag, fid):
    h, w = g.shape[:2]
    # 手臂区：身体两侧中部（垂手时手在 y 40-85%，x 左右各 30%）
    # 用 |full-g| 在两侧中部找最差点
    d = np.abs(full.astype(np.float32) - g.astype(np.float32)).sum(2)
    ys = slice(int(h * 0.35), int(h * 0.85))
    band = d[ys, :]
    band[:, int(w * 0.32):int(w * 0.68)] = 0  # 排除躯干
    y0 = int(h * 0.35) + int(np.unravel_index(band.argmax(), band.shape)[0]) - 110
    x0 = int(np.unravel_index(band.argmax(), band.shape)[1]) - 110
    x0 = max(0, min(w - 240, x0))
    y0 = max(0, min(h - 240, y0))
    sl = np.s_[y0:y0 + 240, x0:x0 + 240]
    row = np.hstack([g[sl], full[sl], trim[sl]])
    label = np.zeros((30, row.shape[1], 3), np.uint8)
    label[:] = (28, 28, 28)
    cv2.putText(label, f"fid={fid}   GT | full160(2-pose) | trim105(垂手only)",
                (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([label, row])


def main():
    panels = []
    for idx, fid in ((2, 16), (4, 32), (6, 48), (9, 72), (12, 96)):
        r_f, g = grab("model_p07", idx)
        r_t, g2 = grab("model_p07_trim", idx)
        if g is None or r_f is None or r_t is None:
            continue
        panels.append(crop_pair(g, r_f, r_t, "p07", fid))
    if panels:
        out = np.vstack(panels)
        p = os.path.join(OUT, "arm_trim_vs_full.jpg")
        cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("wrote", p, out.shape)


if __name__ == "__main__":
    main()
