# -*- coding: utf-8 -*-
"""位姿核查 step4：cameras.json(GT, 经居中) 旋转 vs transforms.json 各候选约定的旋转。

cameras.json 来自采集流水线（03 用它训到 55~59 dB → 其旋转即 GT 真值约定）。
位置经「主体居中」平移，不可直接比；旋转不受居中影响。
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

cams = json.loads((Path.home() / "output/da3_results" / TASK_3V.name
                   / "04_model_3dgs/cameras.json").read_text(encoding="utf-8"))
# img_name 000/001/002 → picked 顺序
cams = sorted(cams, key=lambda c: c["img_name"])


def rot_err(R1, R2):
    """两个旋转矩阵的测地夹角（度）。"""
    R = R1 @ R2.T
    c = (np.trace(R) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


cands = {
    "A inv(M@flip)": lambda M: np.linalg.inv(M @ flip)[:3, :3],
    "F inv(M)": lambda M: np.linalg.inv(M)[:3, :3],
    "G M@flip": lambda M: (M @ flip)[:3, :3],
}
print("== cameras.json 旋转(R_json, w2c) vs 各候选 GT 旋转 的测地夹角 ==")
for name, f in cands.items():
    errs = []
    for n, i in enumerate(PICK):
        M = np.array(frames[i]["transform_matrix"], np.float64)
        R_gt = f(M)
        R_json = np.array(cams[n]["rotation"], np.float64)
        errs.append(rot_err(R_json, R_gt))
    print("  %-12s v0=%6.2f° v1=%6.2f° v2=%6.2f°" % (name, *errs))

# 顺带：R_json 是否就是 w2c？（看 rotation 第三行 · position 是否与平移自洽：t = -R@C）
n = 0
R_json = np.array(cams[n]["rotation"], np.float64)
pos = np.array(cams[n]["position"], np.float64)
print("\ncameras.json t=-R@pos 重构检查（仅验证内部自洽，无外部参照）:")
print("  |t| =", round(np.linalg.norm(-R_json @ pos), 3))
