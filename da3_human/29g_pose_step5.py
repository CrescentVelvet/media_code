# -*- coding: utf-8 -*-
"""位姿核查 step5（终判）：规范无关地比较 GT(cameras.json) 与预测(cameras.npz)。

步骤：
1) 两两中心距：验证 cameras.json 与 transforms.json 是否同一物理位姿（差一个相似变换）；
2) 规范判定：R_delta = R_json @ R_cand^T 应在三视角一致（跨视角 spread 最小者 = 正确约定）；
3) 相似对齐（Umeyama，3 点）把 GT 相机套进预测坐标系 → 每视角真实位姿误差；
4) 渲染判决：FF_max PLY 分别用「预测位姿」和「GT 位姿」渲 v1/v2，比 PSNR。
   GT 位姿渲染若大幅改善 → 「人物偏出画面」= DA3 位姿预测错误实锤。
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
SRC = Path(meta["source"]) / "transforms.json"
PICK = meta["picked_idx"]

tj = json.loads(SRC.read_text(encoding="utf-8"))
frames = tj["frames"]
flip = np.diag([1, -1, -1, 1]).astype(np.float64)

cams = json.loads((Path.home() / "output/da3_results" / TASK_3V.name
                   / "04_model_3dgs/cameras.json").read_text(encoding="utf-8"))
cams = sorted(cams, key=lambda c: c["img_name"])
R_json = np.array([c["rotation"] for c in cams], np.float64)      # (3,3,3) w2c
C_json = np.array([c["position"] for c in cams], np.float64)      # (3,3)

npz_path = Path.home() / "output/da3_results" / TASK_3V.name / "d1_ff_export/cameras.npz"
d = np.load(npz_path)
ext = np.asarray(d["extrinsics"]).astype(np.float64)
sf = float(np.asarray(d["scale_factor"]))
R_pred, C_pred = [], []
for k in range(3):
    w = ext[k]
    R_pred.append(w[:3, :3])
    C_pred.append(-w[:3, :3].T @ (w[:3, 3] / sf))
R_pred = np.array(R_pred)
C_pred = np.array(C_pred)


def rot_err(R1, R2):
    return np.degrees(np.arccos(np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1)))


print("== 1) 两两中心距（米）：cameras.json vs transforms.json 原始平移 ==")
for i in range(3):
    for j in range(i + 1, 3):
        M_i = np.array(frames[PICK[i]]["transform_matrix"], np.float64)[:3, 3]
        M_j = np.array(frames[PICK[j]]["transform_matrix"], np.float64)[:3, 3]
        print("  v%d-v%d: json=%.3f  transforms=%.3f" % (
            i, j, np.linalg.norm(C_json[i] - C_json[j]), np.linalg.norm(M_i - M_j)))

print("\n== 2) 规范判定：R_delta=R_json@R_cand^T 的跨视角 spread ==")
cands = {
    "A inv(M@flip)": lambda M: np.linalg.inv(M @ flip)[:3, :3],
    "F inv(M)": lambda M: np.linalg.inv(M)[:3, :3],
    "G M@flip": lambda M: (M @ flip)[:3, :3],
}
best = None
for name, f in cands.items():
    deltas = []
    for n, i in enumerate(PICK):
        M = np.array(frames[i]["transform_matrix"], np.float64)
        deltas.append(R_json[n] @ f(M).T)
    spread = max(rot_err(deltas[a], deltas[b]) for a in range(3) for b in range(a + 1, 3))
    print("  %-12s spread=%6.2f°" % (name, spread))
    if best is None or spread < best[1]:
        best = (name, spread)
print("  → 采用:", best[0])

# 用胜出约定的 GT w2c（约定 G：w2c = M @ flip，平移列直接取，勿用 -R@C 重构）
f_full = lambda M: (M @ flip) if best[0].startswith("G") else None
assert f_full is not None
W_gt = np.array([f_full(np.array(frames[i]["transform_matrix"], np.float64)) for i in PICK])
R_gt = W_gt[:, :3, :3]
T_gt = W_gt[:, :3, 3]
C_gt_raw = np.array([np.array(frames[i]["transform_matrix"], np.float64)[:3, 3] for i in PICK])

print("\n== 3) Umeyama 相似对齐 GT→预测系，逐视角位姿残差 ==")


def umeyama(src, dst):
    """src->dst 相似变换 (s, R, t)：dst ≈ s R src + t。"""
    cs, cd = src.mean(0), dst.mean(0)
    X, Y = src - cs, dst - cd
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    Rv = U @ Vt
    # 不强制 det>0：允许镜像（若预测世界系与 GT 系互为镜像，强制旋转会留下 ~0.24m 残差）
    print("  Umeyama det(R)=%.3f（负=镜像关系）" % np.linalg.det(Rv))
    var = (X ** 2).sum() / len(src)
    s = (S.sum() / len(src) / var) if var > 0 else 1.0
    t = cd - s * Rv @ cs
    return s, Rv, t


s, Rg, tg = umeyama(C_gt_raw, C_pred)
C_gt_fit = s * (Rg @ C_gt_raw.T).T + tg
for k in range(3):
    dC = np.linalg.norm(C_gt_fit[k] - C_pred[k])
    # 朝向对齐后的残差：GT 朝向转到预测系 vs 预测朝向
    v_gt = (Rg @ (R_gt[k].T @ np.array([0, 0, 1.0])))
    v_pr = R_pred[k].T @ np.array([0, 0, 1.0])
    a = np.degrees(np.arccos(np.clip((v_gt / np.linalg.norm(v_gt)) @ (v_pr / np.linalg.norm(v_pr)), -1, 1)))
    print("  v%d: 中心残差=%.3fm（对齐尺度 s=%.3f） 朝向残差=%.1f°" % (k, dC, s, a))

print("\n== 4) 渲染判决：FF_max PLY × {预测位姿, GT对齐位姿} ==")
sys.path.insert(0, "/mnt/c/code/media_code/da3_human")
code = open("/mnt/c/code/media_code/da3_human/29_d1_diag_3view.py").read()
mod = {}
exec(compile(code, "29.py", "exec"), mod)  # __name__ != __main__

from PIL import Image

ply = Path.home() / "output/da3_results" / TASK_3V.name / "d1_ff_export/gs_max/0000.ply"
data = mod["read_ply"](ply)
Ks = np.asarray(d["intrinsics"]).reshape(3, 3, 3).astype(np.float64)
w, h = 378, 504

# GT 对齐位姿的 w2c：平移 t' 使 C' = -R'^T t'；旋转 R' = R_gt 转预测 gauge：R' = R_pred系下 w2c
# w2c' = w2c_pred_gauge = Rg @ R_gt @ ? ... 直接构造：已知 C' 和 R'（世界旋转随 gauge 变）
# 对 w2c 应用相似变换：点 p' = s Rg p + tg → w2c' = R_gt @ Rg^T / ... 用矩阵构造：
def w2c_gauge(w2c):
    """世界系施加相似 T(p)=s·Rg·p+tg 后，相机的 w2c'。
    p' = s·Rg·p + tg → p = (p' - tg)/s · Rg^T ... 逐点：p = Rg^T (p'-tg)/s
    x_cam = R x + t → x_cam = R Rg^T (p'-tg)/s + t = (R Rg^T/s) p' + (t - R Rg^T tg/s)"""
    R_, t_ = w2c[:3, :3], w2c[:3, 3]
    R2 = R_ @ Rg.T / s
    t2 = t_ - R2 @ tg
    return R2, t2


gts = [np.array(Image.open(TASK_3V / "images" / cams[k]["img_name"]).convert("RGB").resize((w, h)))
       for k in range(3)]

for tag in ["pred", "gt"]:
    vms = []
    for k in range(3):
        if tag == "pred":
            w2c = ext[k][:, :4] if ext[k].shape == (3, 4) else ext[k]
            m = np.eye(4)
            m[:3] = w2c
            m[:3, 3] = m[:3, 3] / sf
        else:
            w2c_full = np.eye(4)
            w2c_full[:3, :3] = R_gt[k]
            w2c_full[:3, 3] = T_gt[k]
            R2, t2 = w2c_gauge(w2c_full)
            m = np.eye(4)
            m[:3, :3] = R2
            m[:3, 3] = t2
        vms.append(m)
    rends, _ = mod["gsplat_render"](data, np.array(vms), Ks, w, h, sh_degree=2)
    for k in range(3):
        p = mod["psnr"](rends[k], gts[k])
        print("  %s v%d: PSNR=%.2f" % (tag.upper(), k, p))
    out = Path.home() / "output/da3_results" / TASK_3V.name / "d1_diag"
    for k in range(3):
        Image.fromarray(rends[k]).save(out / ("poseGT_%s_v%d_render.png" % (tag, k)))
print("\n渲染图已存 d1_diag/poseGT_{pred,gt}_v*_render.png")
