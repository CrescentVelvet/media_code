# -*- coding: utf-8 -*-
# 长轨迹实验：同一采集帧的 5 组渲染 + GT 脸部裁剪对比
import glob
import sys

from PIL import Image, ImageDraw

R = "D:/output/flashworld_results/07_long_cond/render_"
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SLOT41 = {f: i for i, f in enumerate(
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38,
     40, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75,
     77, 79, 81])}

groups = [("traj41_cond12", SLOT41[FRAME]), ("traj41_cond41", SLOT41[FRAME]),
          ("traj82_cond24", FRAME), ("traj82_cond41", FRAME),
          ("traj82_cond82", FRAME)]

gt_p = glob.glob("D:/output/flashworld_input/jiabao_png/%03d_*.png" % FRAME)[0]
gt = Image.open(gt_p).convert("RGB")
gt = gt.resize((int(gt.width * 704 / gt.height), 704))
x0 = (gt.width - 480) // 2
gt = gt.crop((x0, 0, x0 + 480, 704))

imgs = [("GT frame%03d" % FRAME, gt)]
for name, slot in groups:
    im = Image.open(R + name + "/render_%02d.png" % slot).convert("RGB")
    imgs.append((name, im))

# 脸部/上身裁剪：(120,60)-(360,340)，放大 1.6x
crops = []
for label, im in imgs:
    c = im.crop((120, 60, 360, 340)).resize((384, 448), Image.LANCZOS)
    crops.append((label, c))

W = sum(c.width for _, c in crops) + 10 * (len(crops) + 1)
H = 448 + 40 + 20
canvas = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(canvas)
x = 10
for label, c in crops:
    d.text((x + 4, 12), label, fill=(0, 0, 0))
    canvas.paste(c, (x, 40))
    x += c.width + 10
out = "D:/output/flashworld_results/07_long_cond/crop_compare_f%03d.png" % FRAME
canvas.save(out)
print("saved", out)
