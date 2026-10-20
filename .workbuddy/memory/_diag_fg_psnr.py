"""FG PSNR 对比：full160 vs trim105，只算 mask 内、只比共有 test 视角。"""
import os
import numpy as np
import cv2

ROOT = "/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou"
MASKS = os.path.join(ROOT, "colmap_refined/persons/p07/masks")


def fg_psnr(model, idx, fid):
    base = os.path.join(ROOT, model, "test/ours_10000")
    r = cv2.imread(os.path.join(base, "renders", f"{idx:05d}.png"))
    g = cv2.imread(os.path.join(base, "gt", f"{idx:05d}.png"))
    m = cv2.imread(os.path.join(MASKS, f"{fid:05d}.png"), cv2.IMREAD_GRAYSCALE)
    if r is None or g is None or m is None:
        return np.nan, np.nan
    if m.shape != g.shape[:2]:
        m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    mm = m > 127
    if mm.sum() < 100:
        return np.nan, np.nan
    err = (r.astype(np.float32) - g.astype(np.float32)) ** 2
    mse_fg = err[mm].mean()
    psnr_fg = 20 * np.log10(255.0 / np.sqrt(mse_fg)) if mse_fg > 0 else 99
    # 手臂子区：mask 下半部分左右两侧
    ys, xs = np.where(mm)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    ymid = int(y0 + (y1 - y0) * 0.5)
    arm = np.zeros_like(mm)
    arm[ymid:y1, x0:int(x0 + (x1 - x0) * 0.25)] = mm[ymid:y1, x0:int(x0 + (x1 - x0) * 0.25)]
    arm[ymid:y1, int(x1 - (x1 - x0) * 0.25):x1] = mm[ymid:y1, int(x1 - (x1 - x0) * 0.25):x1]
    if arm.sum() < 100:
        return psnr_fg, np.nan
    mse_arm = err[arm].mean()
    psnr_arm = 20 * np.log10(255.0 / np.sqrt(mse_arm)) if mse_arm > 0 else 99
    return psnr_fg, psnr_arm


def main():
    print("idx fid | full_fg trim_fg | full_arm trim_arm")
    fa, ta, fa2, ta2 = [], [], [], []
    for i in range(14):
        fid = i * 8
        f_fg, f_arm = fg_psnr("model_p07", i, fid)
        t_fg, t_arm = fg_psnr("model_p07_trim", i, fid)
        print(f"{i:3d} {fid:3d} | {f_fg:7.2f} {t_fg:7.2f} | {f_arm:7.2f} {t_arm:7.2f}")
        if np.isfinite(f_fg):
            fa.append(f_fg); ta.append(t_fg)
        if np.isfinite(f_arm):
            fa2.append(f_arm); ta2.append(t_arm)
    print(f"\n均值 FG:    full={np.mean(fa):.2f}  trim={np.mean(ta):.2f}  Δ={np.mean(ta)-np.mean(fa):+.2f}")
    print(f"均值 手臂区: full={np.mean(fa2):.2f}  trim={np.mean(ta2):.2f}  Δ={np.mean(ta2)-np.mean(fa2):+.2f}")


if __name__ == "__main__":
    main()
