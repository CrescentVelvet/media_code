# -*- coding: utf-8 -*-
"""位姿核查 step7：验证 cameras.json 旋转约定（渲染判决）+ 清洗后的 GT vs 预测相对位姿。

04_model_3dgs 是用 cameras.json 训练的（train renders 55~59 dB 铁证）。
用本脚本渲染器分别以 {w2c, c2w} 两种约定解释 rotation，渲 v0/v1/v2 比 GT：
  PSNR 高的约定 = 真值。然后重跑约定无关的相对位姿对比。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("EVAL_W", "378")
os.environ.setdefault("EVAL_H", "504")

CAPTURES = Path.home() / "output/da3_results/_captures"
TASK_3V = next(p for p in CAPTURES.iterdir() if p.name.endswith("_3v"))
meta = json.loads((TASK_3V / "meta.json").read_text(encoding="utf-8"))
PICK = meta["picked_idx"]

code = open("/mnt/c/code/media_code/da3_human/29_d1_diag_3view.py").read()
mod = {}
exec(compile(code, "29.py", "exec"), mod)

from PIL import Image

REND = {}

TASK_OUT = Path.home() / "output/da3_results" / TASK_3V.name
ply = TASK_OUT / "04_model_3dgs/point_cloud/iteration_30000/point_cloud.ply"
if not ply.exists():
    ply = next((TASK_OUT / "04_model_3dgs").rglob("point_cloud.ply"))
data = mod["read_ply"](ply)

cams = json.loads((TASK_OUT / "04_model_3dgs/cameras.json").read_text(encoding="utf-8"))
cams = sorted(cams, key=lambda c: c["img_name"])
w, h = 378, 504
sc = w / cams[0]["width"]  # 1439 → 378
Ks = np.array([[[c["fx"] * sc, 0, w / 2.0], [0, c["fy"] * sc, h / 2.0], [0, 0, 1]] for c in cams])

gts = [np.array(Image.open(TASK_3V / "images" / cams[k]["img_name"]).convert("RGB").resize((w, h)))
       for k in range(3)]

W = {}
for conv in ["w2c", "c2w"]:
    vms = []
    for k in range(3):
        R = np.array(cams[k]["rotation"], np.float64)
        pos = np.array(cams[k]["position"], np.float64)
        m = np.eye(4)
        if conv == "w2c":
            m[:3, :3] = R
            m[:3, 3] = -R @ pos
        else:
            m[:3, :3] = R.T
            m[:3, 3] = -R.T @ pos
        vms.append(m)
    rends, _ = mod["gsplat_render"](data, np.array(vms), Ks, w, h, sh_degree=3)
    ps = [mod["psnr"](rends[k], gts[k]) for k in range(3)]
    W[conv] = np.array(vms)
    REND[conv] = rends
    print("%s 约定: v0=%.2f v1=%.2f v2=%.2f dB" % (conv, *ps))

conv = max(["w2c", "c2w"], key=lambda c: sum(
    mod["psnr"](REND[c][k], gts[k]) for k in range(3)))
print("→ 真值约定:", conv)

# ---------- 清洗后的相对位姿对比（GT=json 真值约定，预测=npz）----------
npz_path = TASK_OUT / "d1_ff_export/cameras.npz"
d = np.load(npz_path)
ext = np.asarray(d["extrinsics"]).astype(np.float64)
sf = float(np.asarray(d["scale_factor"]))
W_pred = np.zeros((3, 4, 4))
for k in range(3):
    W_pred[k] = np.eye(4)
    W_pred[k][:3] = ext[k]
    W_pred[k][:3, 3] /= sf
W_gt = W[conv]


def rot_ang(R1, R2):
    return np.degrees(np.arccos(np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1)))


print("\n== 相对位姿对比（约定无关）：v0→vi 相对运动，GT vs 预测 ==")
for i in [1, 2]:
    Tr = W_gt[i] @ np.linalg.inv(W_gt[0])
    Tp = W_pred[i] @ np.linalg.inv(W_pred[0])
    a_rot = rot_ang(Tr[:3, :3], Tp[:3, :3])
    tr, tp = Tr[:3, 3], Tp[:3, 3]
    a_dir = np.degrees(np.arccos(np.clip((tr / np.linalg.norm(tr)) @ (tp / np.linalg.norm(tp)), -1, 1)))
    print("  v0→v%d: 相对旋转差=%.1f°  平移方向差=%.1f°  |t| GT=%.3f 预测=%.3f(m)" % (
        i, a_rot, a_dir, np.linalg.norm(tr), np.linalg.norm(tp)))

# 单应性检验：人物在哪——把 30k 模型质心(人物)投到两类 v1 相机里
ctr = np.array([data["x"].mean(), data["y"].mean(), data["z"].mean()])
print("\n人物质心(30k 模型, json 世界):", np.round(ctr, 3))
for tag, Wset, K in [("GT", W_gt, Ks), ("PRED→json(未对齐,仅参考)", W_pred, Ks)]:
    for i in [0, 1, 2]:
        m = Wset[i]
        pc = m[:3, :3] @ ctr + m[:3, 3]
        if pc[2] < 0.05:
            print("  %s v%d: 人物在相机后方" % (tag, i))
            continue
        u = K[i][0, 0] * pc[0] / pc[2] + w / 2.0
        vv = K[i][1, 1] * pc[1] / pc[2] + h / 2.0
        print("  %s v%d: 人物投影像素=(%.0f, %.0f) 画面 378x504 中心(189,252)" % (tag, i, u, vv))
