# -*- coding: utf-8 -*-
"""位姿核查 step3：数据驱动判 transforms.json 的旋转约定。

事实锚点：v0 渲染与 GT 图对齐（23 dB）→ 预测 v0 朝向 ≈ 正确朝向。
检验：c2w 的某个旋转轴（或其负轴）应稳定指向场景原点（人物），
      且与预测 v0 朝向的夹角应最小。
"""
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


def ang(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return np.degrees(np.arccos(np.clip(a @ b, -1, 1)))


# 预测 v0 朝向（渲染已验证对齐 GT 图）
npz_path = Path.home() / "output/da3_results" / TASK_3V.name / "d1_ff_export/cameras.npz"
d = np.load(npz_path)
ext = np.asarray(d["extrinsics"]).astype(np.float64)
pred_view = []
for k in range(3):
    w = ext[k]
    R = w[:3, :3]
    pred_view.append(R.T @ np.array([0, 0, 1.0]))

print("== 检验1：M 的各轴 / 翻转后各轴 指向场景原点(-C) 的夹角（3帧应一致且小）==")
labels = ["+x", "-x", "+y", "-y", "+z", "-z"]
for i in PICK:
    M = np.array(frames[i]["transform_matrix"], np.float64)
    C = M[:3, 3]
    to_origin = -C
    errs_gl = [ang(M[:3, j], to_origin) for j in range(3)] + [ang(-M[:3, j], to_origin) for j in range(3)]
    Mf = M @ flip
    errs_cv = [ang(Mf[:3, j], to_origin) for j in range(3)] + [ang(-Mf[:3, j], to_origin) for j in range(3)]
    best_gl = min(range(6), key=lambda j: errs_gl[j])
    best_cv = min(range(6), key=lambda j: errs_cv[j])
    print("  frame %-3d GL轴最优 %s=%.1f° | CV轴最优 %s=%.1f°" % (
        i, labels[best_gl], errs_gl[best_gl], labels[best_cv], errs_cv[best_cv]))

print("\n== 检验2：各候选约定的 GT 朝向 vs 预测朝向（每视角夹角，小=一致）==")
cands = {
    "A inv(M@flip)": lambda M: np.linalg.inv(M @ flip),
    "F inv(M)": lambda M: np.linalg.inv(M),
    "G M@flip(w2c)": lambda M: M @ flip,
    "H flip@inv(M)": lambda M: flip @ np.linalg.inv(M),
}
for name, f in cands.items():
    errs = []
    for n, i in enumerate(PICK):
        M = np.array(frames[i]["transform_matrix"], np.float64)
        w2c = f(M)
        v = w2c[:3, :3].T @ np.array([0, 0, 1.0])
        errs.append(ang(v, pred_view[n]))
    print("  %-14s v0=%6.1f° v1=%6.1f° v2=%6.1f°" % (name, *errs))

print("\n== 检验3：各候选约定下 GT 相机朝向 vs 指向原点(-C) 夹角 ==")
for name, f in cands.items():
    errs = []
    for i in PICK:
        M = np.array(frames[i]["transform_matrix"], np.float64)
        w2c = f(M)
        C = -w2c[:3, :3].T @ w2c[:3, 3]
        v = w2c[:3, :3].T @ np.array([0, 0, 1.0])
        errs.append(ang(v, -C))
    print("  %-14s v0=%6.1f° v1=%6.1f° v2=%6.1f°" % (name, *errs))
