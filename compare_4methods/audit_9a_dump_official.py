#!/usr/bin/env python3
"""AUDIT-9a 导出一份「官方口径」的高斯与相机，供另一环境用 gsplat 忠实复刻 render_3dgs。

为什么分两段跑：da3_human 有模型没 gsplat，querysplat 有 gsplat 没模型；torch 版本
不同（2.14 vs 2.8），不能把 site-packages 混着加进 PYTHONPATH。

存的东西（全部沿用官方语义，不做任何额外变换）：
  means/scales/quats/opacities/harmonics   ← pred.gaussians 原样（**未乘 scale_factor**）
  extrinsics_scaled                        ← pred.extrinsics（官方 _apply_depth_alignment 之后）
  extrinsics_unscaled                      ← 官方视频导出的做法：t /= scale_factor
  intrinsics                               ← pred.intrinsics（处理分辨率）
  processed_images                         ← 模型实际看到的 378x504 输入（uint8，官方口径的 GT）
  depth / scale_factor / is_metric
"""
import json
import os
import time
from pathlib import Path

import numpy as np


def log(*a):
    print(*a, flush=True)


model_dir = os.environ.get("DA3_MODEL_DIR", "/mnt/d/wheel/da3_human_ms")
images_dir = os.path.expanduser(os.environ["IMAGES_DIR"])
out_npz = Path(os.path.expanduser(os.environ["OUT_NPZ"]))
process_res = int(os.environ.get("PROCESS_RES", "504"))
ref_strategy = os.environ.get("REF_STRATEGY", "first")

files = []
for e in ("*.png", "*.jpg", "*.jpeg"):
    files.extend(sorted(Path(images_dir).glob(e)))
files = sorted(files)
log(f"🖼️ {len(files)} 视角 @ {process_res}  ref={ref_strategy}")

from depth_anything_3.api import DepthAnything3  # noqa: E402
from depth_anything_3.utils import export as da3_export  # noqa: E402

try:
    import importlib.util
    if importlib.util.find_spec("gsplat") is None:
        da3_export.export_to_gs_video = lambda *a, **k: None  # noqa: ARG005
except Exception:
    pass

t0 = time.time()
model = DepthAnything3.from_pretrained(model_dir).to("cuda")
log(f"⏱️ 加载 {time.time()-t0:.1f}s")

t0 = time.time()
pred = model.inference(
    [str(p) for p in files], process_res=process_res, infer_gs=True,
    ref_view_strategy=ref_strategy, export_dir=None, export_format="",
)
log(f"⏱️ 前向 {time.time()-t0:.1f}s")

g = pred.gaussians


def squeeze_b(a, nd):
    """去掉可能存在的 batch=1 前置维：官方容器是 (b, ...)，本仓有些版本是 (...)。"""
    a = np.asarray(a, dtype=np.float32)
    while a.ndim > nd and a.shape[0] == 1:
        a = a[0]
    return a


means = squeeze_b(g.means.detach().float().cpu().numpy(), 2)
scales = squeeze_b(g.scales.detach().float().cpu().numpy(), 2)
quats = squeeze_b(g.rotations.detach().float().cpu().numpy(), 2)
opac = squeeze_b(g.opacities.detach().float().cpu().numpy(), 1).reshape(-1)
harm = squeeze_b(g.harmonics.detach().float().cpu().numpy(), 3)   # (N, 3, n)
ext = np.asarray(pred.extrinsics, dtype=np.float32)               # (V,3,4) 或 (V,4,4) w2c
intr = np.asarray(pred.intrinsics, dtype=np.float32)
dep = np.asarray(pred.depth, dtype=np.float32)
imgs = np.asarray(pred.processed_images)                          # (V,H,W,3) uint8
sf = float(getattr(pred, "scale_factor", 1.0) or 1.0)
is_metric = int(getattr(pred, "is_metric", 0) or 0)

# 官方视频导出的做法：尺度只留在相机上，除回去即回到高斯的坐标系
ext_un = ext.copy()
if is_metric:
    ext_un[:, :3, 3] /= sf

log(f"📊 高斯 {len(means):,}  harmonics {harm.shape}  scale_factor={sf:.4f} is_metric={is_metric}")
log(f"   means |x| 中位 {np.median(np.abs(means)):.3f} m；相机中心到原点 中位 "
    f"{np.median(np.linalg.norm(ext[:, :3, 3], axis=1)):.3f}")
log(f"   processed_images {imgs.shape} 深度 {dep.shape} 内参 fx={intr[0,0,0]:.2f} "
    f"cx={intr[0,0,2]:.1f} cy={intr[0,1,2]:.1f}")

out_npz.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    out_npz, means=means, scales=scales, quats=quats, opacities=opac,
    harmonics=harm, extrinsics=ext, extrinsics_unscaled=ext_un,
    intrinsics=intr, depth=dep, processed_images=imgs,
    scale_factor=np.float32(sf), is_metric=np.int32(is_metric),
)
log(f"💾 {out_npz}  {out_npz.stat().st_size/1024/1024:.0f} MB")
with open(str(out_npz) + ".json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(dict(npz=str(out_npz), n=int(len(means)), harmonics=list(harm.shape),
                   sh_degree=int(round(harm.shape[-1] ** 0.5)) - 1,
                   scale_factor=sf, is_metric=is_metric,
                   images=imgs.shape, depth=list(dep.shape)), f, indent=1)
