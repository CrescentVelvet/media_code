# -*- coding: utf-8 -*-
"""位姿核查 step9（终审渲染）：FF_max PLY × {预测位姿, GT(json,c2w约定)映射位姿}。

json 相机（已验证 c2w 约定）经 允许镜像的 Umeyama 映射进预测世界系。
镜像情形下渲染图会左右翻转 → PSNR 对原图和水平翻转图各算一次取大。
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
gts_flip = [gt[:, ::-1].copy() for gt in gts]

W_gt = np.zeros((3, 4, 4))
C_gt = np.zeros((3, 3))
for k in range(3):
    R = np.array(cams[k]["rotation"], np.float64)
    pos = np.array(cams[k]["position"], np.float64)
    W_gt[k] = np.eye(4)
    W_gt[k][:3, :3] = R.T          # c2w 约定（step7 已验证）
    W_gt[k][:3, 3] = -R.T @ pos
    C_gt[k] = pos

W_pred = np.zeros((3, 4, 4))
C_pred = np.zeros((3, 3))
for k in range(3):
    W_pred[k] = np.eye(4)
    W_pred[k][:3] = ext[k]
    W_pred[k][:3, 3] /= sf
    C_pred[k] = -W_pred[k][:3, :3].T @ W_pred[k][:3, 3]


def umeyama_row(src, dst):
    """行向量约定：dst ≈ s · src @ Q + t。Q 经 Kabsch 求得。"""
    cs, cd = src.mean(0), dst.mean(0)
    X, Y = src - cs, dst - cd
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    Q = U @ Vt                      # 行映射 Q（允许镜像时不修正 det）
    var = (X ** 2).sum() / len(src)
    s = S.sum() / len(src) / var
    t = cd - s * cs @ Q
    return s, Q, t


s, Q, tg = umeyama_row(C_gt, C_pred)
res = np.linalg.norm((s * (C_gt @ Q) + tg) - C_pred, axis=1)
print("对齐: s=%.3f det(Q)=%.3f 中心残差=%s" % (s, np.linalg.det(Q), np.round(res, 3)))

vms = {}
vms["PRED"] = W_pred
vms["GT"] = np.zeros((3, 4, 4))
for k in range(3):
    R_j, t_j = W_gt[k][:3, :3], W_gt[k][:3, 3]
    # 世界行映射 p_pred = s·Q^T·p_gt + tg（列约定）
    # w2c' = R_j·Q/s ; t' = t_j − R_j·Q·tg/s
    R2 = R_j @ Q / s
    t2 = t_j - R2 @ tg
    vms["GT"][k] = np.eye(4)
    vms["GT"][k][:3, :3] = R2
    vms["GT"][k][:3, 3] = t2

out = TASK_OUT / "d1_diag"
for tag in ["PRED", "GT"]:
    rends, _ = mod["gsplat_render"](data, vms[tag], Ks, w, h, sh_degree=2)
    for k in range(3):
        p1 = mod["psnr"](rends[k], gts[k])
        p2 = mod["psnr"](rends[k], gts_flip[k])
        print("%s v%d: PSNR 原图=%.2f 翻转图=%.2f %s" % (
            tag, k, p1, p2, "(镜像)" if p2 > p1 else ""))
        Image.fromarray(rends[k]).save(out / ("final_%s_v%d.png" % (tag.lower(), k)))
print("渲染图存 d1_diag/final_{pred,gt}_v*.png")
