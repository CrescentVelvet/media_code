#!/usr/bin/env python3
"""29m：导出策略 2×2 消融（峰哥 2026-11-11 晚评审要求）。

背景：GPT 评审指出 D1 白洞对照同时改了裁剪和 SH 两个因素，独立贡献未拆开。
    08 脚本四档导出（official/full_sh/noclip/max）D1 时已全部落盘，本脚本补渲中间两档，
    四档同相机同渲染器同底色，测背景空洞率 + 逐视角 PSNR + 人物前景 alpha/饱和度。

    同时补记录：三输入帧（0/40/81）实际方位角（评审指出 np.linspace 抽的是帧索引非方位角），
    以及灰因素A的严格口径——人物 mask 内 alpha 中位（而非全图中位）。

用法（WSL，querysplat env）：
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate querysplat
    python /mnt/c/code/media_code/da3_human/29m_export_2x2_ablation.py
产物：
    <OUT>/ablation2x2.json + ablation2x2.csv
"""
import json
import os
import types
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("EVAL_W", "378")
os.environ.setdefault("EVAL_H", "504")

# 复用 29_d1_diag 的渲染/指标函数（exec 模块文件，__name__!="__main__" 不触发 main）
mod = types.ModuleType("x")
code = open("/mnt/c/code/media_code/da3_human/29_d1_diag_3view.py").read()
exec(compile(code, "29.py", "exec"), mod.__dict__)

FF = Path("/home/velvet/output/da3_results/佳佳抱枕_3v/d1_ff_export")
GT_CAMS = Path("/home/velvet/output/da3_results/_captures/佳佳抱枕/cameras.npz")
OUT = Path("/home/velvet/output/da3_results/佳佳抱枕_3v/d1_diag")

# 2×2 四档：official=裁剪+DC  full_sh=裁剪+全SH  noclip=不裁剪+DC  max=不裁剪+全SH
CASES = {
    "official": dict(ply="gs_official/0000.ply", crop=True, sh="DC"),
    "full_sh":  dict(ply="gs_full_sh/0000.ply",  crop=True, sh="FullSH"),
    "noclip":   dict(ply="gs_noclip/0000.ply",   crop=False, sh="DC"),
    "max":      dict(ply="gs_max/0000.ply",      crop=False, sh="FullSH"),
}

W, H = 378, 504

npz = np.load(FF / "cameras.npz")
ext = np.asarray(npz["extrinsics"]).astype(np.float64)
if ext.ndim == 4:
    ext = ext[:, :3, :]
if ext.shape[1:] == (3, 4):
    m = np.tile(np.eye(4), (3, 1, 1))
    m[:, :3, :] = ext
    ext = m
sf = float(np.asarray(npz["scale_factor"]))
viewmats = np.repeat(np.eye(4)[None], 3, 0)
for k in range(3):
    vm = ext[k].copy()
    vm[:3, 3] = vm[:3, 3] / sf
    viewmats[k] = vm  # extrinsics 本身就是 w2c（gsplat viewmat），勿再求逆
# npz 内参/深度已是 process_res=504 口径（378×504），勿再缩放（D1-B 同款）
K = np.asarray(npz["intrinsics"]).astype(np.float64)[0]
Ks = np.repeat(K[None], 3, 0)

cap = Path("/home/velvet/output/da3_results/_captures/佳佳抱枕_3v/images")
gts = [np.array(Image.open(p).convert("RGB").resize((W, H)))
       for p in sorted(cap.glob("*.png"))[:3]]

# 人物前景 mask：预测深度（相对口径，只要把人/背景分开）
dep = np.asarray(npz["depth"]).reshape(3, 504, 378)
med = np.median(dep)
fg = np.array([np.array(Image.fromarray((d <= 1.5 * med).astype(np.uint8) * 255)
                        .resize((W, H))) > 127 for d in dep])
bg = ~fg

# ---- 三输入帧实际方位角（transforms.json c2w OpenGL，帧 0/40/81）----
# 注意：capture cameras.npz 只有 32 个精选位姿，帧索引对应 transforms.json（81 帧）
tj = json.loads(Path("/mnt/d/dataset/手部微动样例/"
                     "01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕/transforms.json")
                .read_text())
frames = tj["frames"]
target = np.array([1.449, -0.586, -0.500])  # audit_32 拟合的注视点（capture 系）
PICK = [0, 40, 81]
az = []
for i in PICK:
    C = np.array(frames[i]["transform_matrix"], np.float64)[:3, 3]
    d = target - C
    az.append(float(np.degrees(np.arctan2(d[0], d[2]))))
az_sorted = sorted(az)
gaps = [round(az_sorted[1] - az_sorted[0], 1), round(az_sorted[2] - az_sorted[1], 1)]
print("[29m] 三输入帧实际方位角: %s  间距=%s°（不均匀，评审意见已记录）"
      % (np.round(az_sorted, 1), gaps))

# ---- 四档渲染 + 指标 ----
report = {"input_azimuths_deg": np.round(az_sorted, 1).tolist(),
          "input_azimuth_gaps_deg": gaps, "pipelines": {}}
for tag, cfg in CASES.items():
    d = mod.read_ply(str(FF / cfg["ply"]))
    n = d["x"].shape[0]
    # SH 阶数按 PLY 实际 f_rest 数推断：k_rest=(deg+1)^2-1，每通道 k_rest 个
    n_rest = len([k for k in d if k.startswith("f_rest_")])
    sh_deg = int(round((n_rest // 3 + 1) ** 0.5)) - 1 if n_rest else 0
    rends, alphas = [], []
    for k in range(3):
        r, a = mod.gsplat_render(d, viewmats[k:k+1], Ks[k:k+1], W, H,
                                 sh_degree=sh_deg)
        rends.append(r[0])
        alphas.append(a[0])
    psnrs = [round(mod.psnr(rends[k], gts[k]), 2) for k in range(3)]
    # 背景空洞率 + 前景内部 alpha（评审要的严格口径）
    hole, a_fg = [], []
    for k in range(3):
        hole.append(round(float((alphas[k][bg[k]] < 0.05).mean() * 100), 2))
        a_fg.append(round(float(np.median(alphas[k][fg[k]])), 3))
    # 人物前景饱和度 render vs GT
    sat_r, sat_g = [], []
    for k in range(3):
        st_r = mod.color_stats(rends[k], fg[k])
        st_g = mod.color_stats(gts[k], fg[k])
        sat_r.append(st_r.get("mean_sat"))
        sat_g.append(st_g.get("mean_sat"))
    report["pipelines"][tag] = dict(
        n_points=n, crop=cfg["crop"], sh=cfg["sh"], psnr_per_view=psnrs,
        hole_pct_bg=hole, alpha_median_fg=a_fg,
        sat_fg_render=sat_r, sat_fg_gt=sat_g)
    print("[29m] %-9s N=%6d crop=%-5s sh=%-6s PSNR=%s hole_bg=%s%% alpha_fg=%s sat_fg=%s"
          % (tag, n, cfg["crop"], cfg["sh"], psnrs,
             np.round(np.mean(hole), 2), a_fg, sat_r))

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ablation2x2.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("[29m] ->", OUT / "ablation2x2.json")
