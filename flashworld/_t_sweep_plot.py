# -*- coding: utf-8 -*-
# 由 sweep_psnr.txt 画「PSNR vs 帧序号」曲线（每条件数一条线，纯 PIL）
import re
from PIL import Image, ImageDraw

TXT = r"D:/output/flashworld_results/06_multi_cond/sweep_psnr.txt"
OUT = r"D:/output/flashworld_results/06_multi_cond/sweep_curve.png"
HEADER = ["1cond", "2cond", "3cond", "5cond", "8cond", "12cond", "24cond"]
# 配色（matplotlib viridis 近似采样）
COLORS = [(68, 1, 84), (49, 54, 149), (42, 120, 142), (53, 183, 121),
          (144, 215, 67), (253, 200, 40), (200, 80, 40)]

lines = [l for l in open(TXT, encoding="utf-8").read().splitlines()
         if re.match(r"^\d\d", l)]
data = {}
for l in lines:
    parts = l.replace("*", "").split()
    data[int(parts[0])] = [float(x) for x in parts[1:]]
frames = sorted(data)
M = [data[f] for f in frames]

W, H = 1200, 620
L, R, T, B = 90, 30, 60, 70
pw, ph = W - L - R, H - T - B
ymin = min(min(r) for r in M) - 0.5
ymax = max(max(r) for r in M) + 0.8

img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)


def xy(fi, val):
    x = L + pw * (fi - frames[0]) / (frames[-1] - frames[0])
    y = T + ph * (1 - (val - ymin) / (ymax - ymin))
    return x, y


# 网格与坐标轴
d.text((W // 2 - 260, 16), "FlashWorld multi-image condition sweep "
         "(jiabao narrow24, PSNR vs capture GT)", fill=(20, 20, 20))
for v in range(int(ymin) + 1, int(ymax) + 1):
    y = xy(0, v)[1]
    d.line([(L, y), (W - R, y)], fill=(225, 225, 225))
    d.text((L - 34, y - 7), "%d" % v, fill=(60, 60, 60))
for f in range(0, 24, 2):
    x = xy(f, ymin)[0]
    d.line([(x, T), (x, H - B)], fill=(240, 240, 240))
    d.text((x - 6, H - B + 12), "%d" % f, fill=(60, 60, 60))
d.rectangle([L, T, W - R, H - B], outline=(120, 120, 120))
d.text((W // 2 - 40, H - 34), "frame index", fill=(60, 60, 60))
d.text((14, H // 2 - 30), "P\nS\nN\nR\n(\nd\nB\n)", fill=(60, 60, 60))

# 曲线
for j, name in enumerate(HEADER):
    c = COLORS[j]
    pts = [xy(f, M[k][j]) for k, f in enumerate(frames)]
    d.line(pts, fill=c, width=3)
    for (x, y) in pts:
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=c)

# 图例（右上两列）
lx, ly = W - R - 330, T + 14
for j, name in enumerate(HEADER):
    x = lx + (j % 4) * 86
    y = ly + (j // 4) * 26
    d.line([(x, y + 7), (x + 22, y + 7)], fill=COLORS[j], width=3)
    d.text((x + 28, y), name, fill=(30, 30, 30))

img.save(OUT)
print("saved", OUT)
