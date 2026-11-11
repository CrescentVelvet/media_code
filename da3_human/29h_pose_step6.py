# -*- coding: utf-8 -*-
"""位姿核查 step6（终判·约定无关）：比较 v0→v1 / v0→v2 相对位姿。

世界系规范、居中、镜像全部在相对位姿下消掉：
  T_rel = w2c_i @ inv(w2c_0)   （相机 i 相对相机 0 的运动， cam0 坐标系表达）
比较 GT(cameras.json) 与预测(cameras.npz) 的相对旋转角、平移方向角、平移距离。
"""
import json
from pathlib import Path

import numpy as np

CAPTURES = Path.home() / "output/da3_results/_captures"
TASK_3V = next(p for p in CAPTURES.iterdir() if p.name.endswith("_3v"))
meta = json.loads((TASK_3V / "meta.json").read_text(encoding="utf-8"))
SRC = Path(meta["source"]) / "transforms.json"
PICK = meta["picked_idx"]
flip = np.diag([1, -1, -1, 1]).astype(np.float64)

tj = json.loads(SRC.read_text(encoding="utf-8"))
frames = tj["frames"]
W_gt = np.array([(np.array(frames[i]["transform_matrix"], np.float64) @ flip) for i in PICK])

cams = json.loads((Path.home() / "output/da3_results" / TASK_3V.name
                   / "04_model_3dgs/cameras.json").read_text(encoding="utf-8"))
cams = sorted(cams, key=lambda c: c["img_name"])
W_json = np.zeros((3, 4, 4))
for k in range(3):
    R = np.array(cams[k]["rotation"], np.float64)
    pos = np.array(cams[k]["position"], np.float64)
    W_json[k] = np.eye(4)
    W_json[k][:3, :3] = R
    W_json[k][:3, 3] = -R @ pos

npz_path = Path.home() / "output/da3_results" / TASK_3V.name / "d1_ff_export/cameras.npz"
d = np.load(npz_path)
ext = np.asarray(d["extrinsics"]).astype(np.float64)
sf = float(np.asarray(d["scale_factor"]))
W_pred = np.zeros((3, 4, 4))
for k in range(3):
    W_pred[k] = np.eye(4)
    W_pred[k][:3] = ext[k]
    W_pred[k][:3, 3] /= sf


def rot_ang(R1, R2):
    return np.degrees(np.arccos(np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1)))


def rel(w2c_i, w2c_0):
    return w2c_i @ np.linalg.inv(w2c_0)


print("== 相对位姿对比（GT 用 cameras.json；验证 transforms 约定一致性后并列）==")
print("%-14s %-10s %-12s %-12s %-12s" % ("pair", "来源", "相对旋转角", "平移方向角", "平移距离(m)"))
for tag, W in [("json", W_json), ("transforms", W_gt)]:
    for i in [1, 2]:
        T_rel = rel(W[i], W[0])
        Rr, tr = T_rel[:3, :3], T_rel[:3, 3]
        P_rel = rel(W_pred[i], W_pred[0])
        Rp, tp = P_rel[:3, :3], P_rel[:3, 3]
        a_rot = rot_ang(Rr, Rp)
        a_dir = np.degrees(np.arccos(np.clip(
            (tr / np.linalg.norm(tr)) @ (tp / np.linalg.norm(tp)), -1, 1)))
        print("%-14s %-10s %-10.1f° %-11.1f° %-8.3f / %.3f" % (
            "v0→v%d" % i, tag, a_rot, a_dir, np.linalg.norm(tr), np.linalg.norm(tp)))

# json 与 transforms 是否同一物理位姿（相对旋转应≈0）
print("\n(json vs transforms 自检：同一对相机的相对旋转角，应≈0°)")
for i in [1, 2]:
    T1 = rel(W_json[i], W_json[0])
    T2 = rel(W_gt[i], W_gt[0])
    print("  v0→v%d: %.2f°" % (i, rot_ang(T1[:3, :3], T2[:3, :3])))
