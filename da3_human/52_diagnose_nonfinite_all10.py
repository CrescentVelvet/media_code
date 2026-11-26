#!/usr/bin/env python3
"""只读诊断stage1案例DA3高斯非正尺度；不改输入、不导出PLY。

Env: CASE_DIR, DA3_MODEL_DIR, DIAGNOSTIC_PATH, BRANCH=rgb|pose（默认rgb）。
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

spec = importlib.util.spec_from_file_location("cam", Path(__file__).with_name("36_run_hisilicon_cameras.py"))
cam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cam)
files, ext, K, _, _ = cam.load_stage1_inputs(Path(os.environ["CASE_DIR"]))
branch = os.environ.get("BRANCH", "rgb")
if branch not in ("rgb", "pose"):
    raise ValueError("BRANCH只允许rgb/pose")
print("🏋️ 加载模型", flush=True)
model = DepthAnything3.from_pretrained(os.environ["DA3_MODEL_DIR"]).to("cuda").eval()
torch.manual_seed(42)
np.random.seed(42)
print(f"🚀 仅{branch}前向", flush=True)
p = model.inference([str(x) for x in files], extrinsics=ext if branch == "pose" else None,
                    intrinsics=K if branch == "pose" else None,
                    process_res=704, infer_gs=True,
                    ref_view_strategy="saddle_balanced", export_dir=None, export_format="")
print("tensors", {k: tuple(getattr(p.gaussians, k).shape)
                 for k in ("means", "scales", "rotations", "harmonics", "opacities")}, flush=True)
info = {"branch": branch}
# 同时保留尺度符号所来自的深度范围；不能仅靠scales正负反推具体深度根因。
info["depth"] = {"shape": list(p.depth.shape), "dtype": str(p.depth.dtype),
                 "min": float(p.depth.min()), "max": float(p.depth.max()),
                 "nonpositive": int((p.depth <= 0).sum()),
                 "nonfinite": int((~np.isfinite(p.depth)).sum())}
print("depth", info["depth"], flush=True)
# 只看非有限会把负尺度漏掉；标准PLY的log(scale)要求每个分量严格为正。
for key in ("means", "scales", "rotations", "harmonics", "opacities"):
    t = getattr(p.gaussians, key).detach().float()
    bad = ~torch.isfinite(t).all(dim=-1)
    finite = t[torch.isfinite(t)]
    info[key] = {"shape": list(t.shape), "dtype": str(t.dtype),
                 "bad_rows": int(bad.sum()), "min_finite": float(finite.min()),
                 "max_finite": float(finite.max())}
    info[key]["nonpositive"] = int((t <= 0).sum()) if key == "scales" else None
    print(key, info[key], flush=True)
s = p.gaussians.scales
bad = s <= 0
ids = bad.any(dim=-1).nonzero().flatten().cpu()
print("nonpositive_gaussian_rows", len(ids), flush=True)
print("nonpositive_scale_values", s[bad][:20], flush=True)
print("views", torch.unique(ids // 490000).cpu().tolist(), flush=True)
info.update(nonpositive_gaussian_ids=ids, nonpositive_scale_values=s[bad].float().cpu(),
            nonpositive_means=p.gaussians.means[bad.any(dim=-1)].float().cpu(),
            nonpositive_opacity=p.gaussians.opacities[bad.any(dim=-1)].float().cpu())
diagnostic_path = Path(os.environ["DIAGNOSTIC_PATH"])
if diagnostic_path.is_symlink() or diagnostic_path.exists():
    raise FileExistsError(f"拒绝覆盖诊断文件: {diagnostic_path}")
if not diagnostic_path.parent.is_dir():
    raise FileNotFoundError(f"诊断输出目录须预先存在: {diagnostic_path.parent}")
torch.save(info, diagnostic_path)
print("✅ 诊断保存", os.environ["DIAGNOSTIC_PATH"], flush=True)
