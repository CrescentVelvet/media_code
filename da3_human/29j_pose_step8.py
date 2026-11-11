# -*- coding: utf-8 -*-
"""位姿核查 step8（判决）：PLY 按视角分块 × c2w 变换，验证「各视角高斯在自身相机系」假说。

若假说成立：chunk_k 施加 c2w_k 后，三个视角渲染 PSNR 应全面提升（人物回画面）。
若 PLY 本就在 ref/world 系：v0 会恶化（v0 原本对齐）。
"""
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("EVAL_W", "378")
os.environ.setdefault("EVAL_H", "504")

CAPTURES = Path.home() / "output/da3_results/_captures"
TASK_3V = next(p for p in CAPTURES.iterdir() if p.name.endswith("_3v"))
meta = json.loads((TASK_3V / "meta.json").read_text(encoding="utf-8"))

code = open("/mnt/c/code/media_code/da3_human/29_d1_diag_3view.py").read()
mod = {}
exec(compile(code, "29.py", "exec"), mod)

from PIL import Image

TASK_OUT = Path.home() / "output/da3_results" / TASK_3V.name
ply = TASK_OUT / "d1_ff_export/gs_max/0000.ply"
data = mod["read_ply"](ply)
n = data["x"].shape[0]
nv = 3
per = n // nv
print("总高斯 %d = %d × %d" % (n, nv, per))

npz_path = TASK_OUT / "d1_ff_export/cameras.npz"
d = np.load(npz_path)
ext = np.asarray(d["extrinsics"]).astype(np.float64)
sf = float(np.asarray(d["scale_factor"]))
Ks = np.asarray(d["intrinsics"]).reshape(3, 3, 3).astype(np.float64)
w, h = 378, 504

cams = json.loads((TASK_OUT / "04_model_3dgs/cameras.json").read_text(encoding="utf-8"))
cams = sorted(cams, key=lambda c: c["img_name"])
gts = [np.array(Image.open(TASK_3V / "images" / cams[k]["img_name"]).convert("RGB").resize((w, h)))
       for k in range(3)]

w2c = np.zeros((3, 4, 4))
for k in range(3):
    w2c[k] = np.eye(4)
    w2c[k][:3] = ext[k]
    w2c[k][:3, 3] /= sf

xyz = np.stack([data["x"], data["y"], data["z"]], -1).astype(np.float64)

for tag in ["raw", "perview_c2w"]:
    xyz_t = xyz.copy()
    if tag == "perview_c2w":
        for k in range(nv):
            sl = slice(k * per, (k + 1) * per)
            c2w = np.linalg.inv(w2c[k])
            # 高斯已乘 sf（metric），相机平移已除 sf → 同尺度，直接应用
            xyz_t[sl] = (c2w[:3, :3] @ xyz[sl].T).T + c2w[:3, 3]
    d2 = dict(data)
    d2["x"], d2["y"], d2["z"] = xyz_t[:, 0], xyz_t[:, 1], xyz_t[:, 2]
    rends, _ = mod["gsplat_render"](d2, w2c, Ks, w, h, sh_degree=2)
    ps = [mod["psnr"](rends[k], gts[k]) for k in range(3)]
    print("%s: v0=%.2f v1=%.2f v2=%.2f dB" % (tag, *ps))
    if tag == "perview_c2w":
        out = TASK_OUT / "d1_diag"
        for k in range(3):
            Image.fromarray(rends[k]).save(out / ("perview_c2w_v%d_render.png" % k))
