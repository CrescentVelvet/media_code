"""精确找抬手起点：用 mask 顶部（手举过头→top 上移）+ 末帧渲染的 alpha 看透明度。"""
import os
import glob
import numpy as np
import cv2

ROOT = "/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou"


def main():
    for pid in ("p02", "p07"):
        mdir = os.path.join(ROOT, "colmap_refined", "persons", pid, "masks")
        mp = sorted(glob.glob(os.path.join(mdir, "*.png")))
        tops, areas = [], []
        for p in mp:
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            mm = m > 127
            ys, xs = np.where(mm)
            tops.append(ys.min() / m.shape[0] if ys.size else 1.0)
            areas.append(mm.mean())
        tops = np.array(tops)
        # 找抬手起点：top 从「头部水平」掉到「头以上」的突变点
        head = np.median(tops[:100])
        print(f"\n{pid}: 前 100 帧 top 中位={head:.3f}")
        # 打印 118 之后的 top，找突变
        for i in range(115, len(mp), 2):
            mark = " <==抬手" if tops[i] < head - 0.12 else ""
            print(f"  f{i:3d} top={tops[i]:.3f} area={areas[i]:.4f}{mark}")


if __name__ == "__main__":
    main()
