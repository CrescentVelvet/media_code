"""把整段序列做成 contact sheet，肉眼确认姿态切换点。"""
import os
import glob
import numpy as np
import cv2

ROOT = "/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou"
IMAGES = os.path.join(ROOT, "colmap_scene", "images")
OUT = "/mnt/d/output/som_human_hand/diag"
os.makedirs(OUT, exist_ok=True)


def sheet(pid, cols=10, cell_w=150):
    ps = sorted(glob.glob(os.path.join(IMAGES, "*")))
    n = len(ps)
    sel = list(range(0, n, max(1, n // (cols * 3))))[: cols * 3]
    mdir = os.path.join(ROOT, "colmap_refined", "persons", pid, "masks")
    tiles = []
    for i in sel:
        g = cv2.imread(ps[i])
        if g is None:
            continue
        hgt = int(g.shape[0] * cell_w / g.shape[1])
        t = cv2.resize(g, (cell_w, hgt), interpolation=cv2.INTER_AREA)
        # 叠 mask 轮廓（该帧原图名）
        stem = os.path.basename(ps[i]).rsplit(".", 1)[0]
        mp = os.path.join(mdir, stem + ".png")
        if os.path.exists(mp):
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (cell_w, hgt), interpolation=cv2.INTER_NEAREST)
                cnts, _ = cv2.findContours((m > 127).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(t, cnts, -1, (0, 255, 0), 1)
        cv2.putText(t, str(i), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        tiles.append(t)
    if not tiles:
        return
    hh = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
             for t in tiles]
    rows = []
    for r in range(0, len(tiles), cols):
        chunk = tiles[r:r + cols]
        while len(chunk) < cols:
            chunk.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(chunk))
    out = np.vstack(rows)
    p = os.path.join(OUT, f"pose_sheet_{pid}.jpg")
    cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", p, out.shape)


if __name__ == "__main__":
    sheet("p07")
    sheet("p02")
