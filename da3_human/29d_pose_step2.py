# -*- coding: utf-8 -*-
"""位姿核查 step2：判 transforms.json 约定 + 预测相机是否指向人物。"""
import json
from pathlib import Path

import numpy as np

CAPTURES = Path.home() / "output/da3_results/_captures"
TASK_3V = next(p for p in CAPTURES.iterdir() if p.name.endswith("_3v"))
meta = json.loads((TASK_3V / "meta.json").read_text(encoding="utf-8"))
SRC = Path(meta["source"]) / "transforms.json"
PICK = meta["picked_idx"]

tj = json.loads(SRC.read_text(encoding="utf-8"))
frames = tj["frames"]
flip = np.diag([1, -1, -1, 1]).astype(np.float64)

npz_path = Path.home() / "output/da3_results" / TASK_3V.name / "d1_ff_export/cameras.npz"
d = np.load(npz_path)
ext = np.asarray(d["extrinsics"]).astype(np.float64)
sf = float(np.asarray(d["scale_factor"]))


def view_dir(w2c):
    return (w2c[:3, :3].T @ np.array([0, 0, 1.0]))


def center(w2c):
    return -w2c[:3, :3].T @ w2c[:3, 3]


def ang(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(a @ b, -1, 1)))


# ---- 1) GT 约定判定：哪种转换让三帧相机都"看向场景中心" ----
cands = {
    "A c2w_gl=M, flip右乘": lambda M: np.linalg.inv(M @ flip),
    "B c2w_gl=M, flip左乘": lambda M: np.linalg.inv(flip @ M),
    "C M即w2c_gl": lambda M: M @ flip,
    "D M即w2c_cv": lambda M: M,
    "E c2w_cv=M(无需flip)": lambda M: np.linalg.inv(M),
}
print("== GT 约定判定（角度 = 相机朝向 vs 指向相机中心均值，越小越合理）==")
for name, f in cands.items():
    errs = []
    Cs = []
    for i in PICK:
        M = np.array(frames[i]["transform_matrix"], np.float64)
        w2c = f(M)
        Cs.append(center(w2c))
    mean_c = np.mean(Cs, axis=0)
    for i in PICK:
        M = np.array(frames[i]["transform_matrix"], np.float64)
        w2c = f(M)
        errs.append(ang(view_dir(w2c), mean_c - center(w2c)))
    print("  %-22s v0=%5.1f° v1=%5.1f° v2=%5.1f°" % (name, *errs))

# ---- 2) 预测相机（cameras.npz）是否指向场景中心 ----
print("\n== 预测相机 vs 预测场景中心 ==")
Cs, Ws = [], []
for k in range(3):
    w = ext[k].copy()
    if w.shape == (3, 4):
        m = np.eye(4)
        m[:3] = w
        w = m
    w = w.copy()
    w[:3, 3] = w[:3, 3] / sf
    Ws.append(w)
    Cs.append(center(w))
mean_c = np.mean(Cs, axis=0)
for k in range(3):
    print("  v%d: 朝向误差 %5.1f°（中心 %s）" % (k, ang(view_dir(Ws[k]), mean_c - Cs[k]), np.round(Cs[k], 2)))
