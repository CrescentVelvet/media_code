"""汇总 E-01 重测：旧口径 vs 米制，三组前馈 init。

输出 compare_4methods/ppt_assets/p1_retest_metrics.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/c/code/media_code/compare_4methods")
OUT = ROOT / "ppt_assets"
OLDJ = json.load(open(OUT / "p1_metrics.json", encoding="utf-8"))

B = Path("/home/velvet/output/da3_gs_results/佳佳抱枕_p1m")
d = np.load(B / "cameras.npz")
e = d["extrinsics"].astype(float)
C = -np.einsum("nji,nj->ni", e[:, :, :3], e[:, :, 3])
dist = np.linalg.norm(C - C[0], axis=1)
near = dist <= 1.0

GROUP = [
    ("A0 前馈415万+2k", "②前馈 init + 2k", "final_A0_ff2k"),
    ("A1 前馈415万+5k", "③前馈 init + 5k", "final_A1_ff5k"),
    ("A2 前馈57万+5k", "⑥前馈 init 57万 + 5k", "final_A2_ff500k_5k"),
]

res = {}
print(f"{'组':<18} {'旧PSNR':>7} {'新PSNR':>7} {'Δ':>7} | {'旧SSIM':>7} {'新SSIM':>7} | "
      f"{'旧锐度':>7} {'新锐度':>7} | {'旧v0':>6} {'新v0':>6} | {'旧远':>6} {'新远':>6}")
for name, oldkey, sub in GROUP:
    o = OLDJ[oldkey]
    n = json.load(open(B / sub / "metrics_e01_retest.json", encoding="utf-8"))
    pv = [x["psnr"] for x in n["per_view"]]
    pv = np.array(pv)
    row = dict(
        old_psnr=round(float(o["psnr"]), 2), new_psnr=round(float(np.mean(pv)), 2),
        old_ssim=round(float(o["ssim"]), 4), new_ssim=round(float(n["mean"]["ssim"]), 4),
        old_sharp=round(float(o["sharp"]), 1), new_sharp=round(float(n["mean"]["sharp"]), 1),
        old_v0=round(float(o["psnr_v0"]), 2), new_v0=round(float(pv[0]), 2),
        old_far=round(float(o["psnr_far"]), 2), new_far=round(float(pv[~near].mean()), 2),
        new_near=round(float(pv[near].mean()), 2),
        per_view_old=[round(x, 2) for x in o["per_view_psnr"]],
        per_view_new=[round(x, 2) for x in pv],
        corr_old=round(float(np.corrcoef(dist, o["per_view_psnr"])[0, 1]), 3),
        corr_new=round(float(np.corrcoef(dist, pv)[0, 1]), 3),
        train_seconds=None,
    )
    s = B / sub / "short_train_summary.json"
    if s.exists():
        row["train_seconds"] = json.load(open(s, encoding="utf-8"))["train_seconds"]
    res[name] = row
    print(f"{name:<18} {row['old_psnr']:7.2f} {row['new_psnr']:7.2f} "
          f"{row['new_psnr']-row['old_psnr']:+7.2f} | {row['old_ssim']:7.4f} {row['new_ssim']:7.4f} | "
          f"{row['old_sharp']:7.1f} {row['new_sharp']:7.1f} | {row['old_v0']:6.2f} {row['new_v0']:6.2f} | "
          f"{row['old_far']:6.2f} {row['new_far']:6.2f}")

res["__meta__"] = dict(
    distance_m=[round(float(x), 3) for x in dist],
    near_mask=[bool(x) for x in near],
    note="near = 距参考视角相机中心 <=1 m（与 E-01 对比图同口径）",
)
for k, v in res.items():
    if k != "__meta__":
        print(f"  {k}: corr(距离,PSNR) 旧 {v['corr_old']:+.3f} → 新 {v['corr_new']:+.3f}；训练 {v['train_seconds']}s")

with open(OUT / "p1_retest_metrics.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(res, f, indent=4, ensure_ascii=False)
print("✅", OUT / "p1_retest_metrics.json")
