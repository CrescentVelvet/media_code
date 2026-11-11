# -*- coding: utf-8 -*-
"""D1 位姿核查：DA3 预测相机（cameras.npz）vs Remy 采集包 GT 位姿（transforms.json）。

3v 输入帧 = 源包第 0/40/81 帧（见 _captures/佳佳抱枕_3v/meta.json picked_idx）。
判读：
  - 朝向角误差大（几十度）→ "人物偏出画面"是位姿预测错，不是渲染 bug；
  - 误差小但渲染仍错位 → 才是几何一致性问题。

用法（WSL，querysplat env）：
    python 29c_pose_check_3view.py
"""
import json
import sys
from pathlib import Path

import numpy as np

CAPTURES = Path.home() / "output/da3_results/_captures"
TASK_3V = next(p for p in CAPTURES.iterdir() if p.name.endswith("_3v"))
SRC_ROOT = Path("/mnt/d/dataset")
SRC = next(SRC_ROOT.rglob("transforms.json")) if False else None
# 源包目录 = meta.json 里的 source 字段
meta = json.loads((TASK_3V / "meta.json").read_text(encoding="utf-8"))
SRC = Path(meta["source"]) / "transforms.json"
PICK = meta["picked_idx"]

npz_path = Path.home() / "output/da3_results" / (TASK_3V.name) / "d1_ff_export/cameras.npz"

tj = json.loads(SRC.read_text(encoding="utf-8"))
frames = tj["frames"]
flip = np.diag([1, -1, -1, 1]).astype(np.float64)
gt_w2c = []
for i in PICK:
    M = np.array(frames[i]["transform_matrix"], np.float64)  # c2w OpenGL
    c2w_cv = M @ flip                                        # → opencv 约定（实测公式）
    gt_w2c.append(np.linalg.inv(c2w_cv))

d = np.load(npz_path)
ext = np.asarray(d["extrinsics"]).astype(np.float64)  # (3,3,4|4,4) w2c opencv
sf = float(np.asarray(d["scale_factor"]))
print("scale_factor =", round(sf, 4))


def center_dir(w2c):
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    C = -R.T @ t                          # 相机中心（世界系）
    view = R.T @ np.array([0, 0, 1.0])    # 相机朝向（opencv +z 前）
    return C, view / np.linalg.norm(view)


print("%-4s %-24s %-24s %-10s %-12s" % ("view", "GT中心(m)", "预测中心(÷sf)", "中心距(m)", "朝向角误差"))
for k in range(3):
    Cg, dg = center_dir(gt_w2c[k])
    w2c_p = ext[k].copy()
    if w2c_p.shape == (3, 4):
        m = np.eye(4)
        m[:3] = w2c_p
        w2c_p = m
    w2c_p[:3, 3] = w2c_p[:3, 3] / sf
    Cp, dp = center_dir(w2c_p)
    dist = np.linalg.norm(Cg - Cp)
    ang = np.degrees(np.arccos(np.clip(dg @ dp, -1, 1)))
    print("v%-3d %-24s %-24s %-10.3f %-8.2f°" % (k, np.round(Cg, 2), np.round(Cp, 2), dist, ang))

# 两两夹角核对（确认相机顺序没串）
print("\n相机中心两两夹角（相对场景均值中心）:")
for tag, cams in [("GT", [center_dir(m)[0] for m in gt_w2c])]:
    ctr = np.mean(cams, axis=0)
    dirs = [c - ctr for c in cams]
    dirs = [x / np.linalg.norm(x) for x in dirs]
    for i in range(3):
        for j in range(i + 1, 3):
            a = np.degrees(np.arccos(np.clip(dirs[i] @ dirs[j], -1, 1)))
            print("  %s v%d-v%d: %.1f°" % (tag, i, j, a))
cams_p = []
for k in range(3):
    w2c_p = ext[k].copy()
    if w2c_p.shape == (3, 4):
        m = np.eye(4)
        m[:3] = w2c_p
        w2c_p = m
    w2c_p[:3, 3] = w2c_p[:3, 3] / sf
    cams_p.append(center_dir(w2c_p)[0])
ctr = np.mean(cams_p, axis=0)
dirs = [c - ctr for c in cams_p]
dirs = [x / np.linalg.norm(x) for x in dirs]
for i in range(3):
    for j in range(i + 1, 3):
        a = np.degrees(np.arccos(np.clip(dirs[i] @ dirs[j], -1, 1)))
        print("  PRED v%d-v%d: %.1f°" % (i, j, a))
print("\n(meta.json pairwise_deg: min=%.1f median=%.1f max=%.1f)" % (
    meta["pairwise_deg"]["min"], meta["pairwise_deg"]["median"], meta["pairwise_deg"]["max"]))
