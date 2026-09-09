#!/usr/bin/env python3
"""验证 smplx FLAME 加载（smplx 约定布局：FLAME2020/flame/FLAME_NEUTRAL.pkl）。"""
import numpy as np
import torch

import smplx

BASE = "/mnt/d/wheel/flame_human_ms/FLAME2020"

m = smplx.create(model_path=BASE, model_type="flame",
                 num_expression_coeffs=100, use_face_contour=False)
out = m()
v = out.vertices.detach().cpu().numpy().squeeze()
j = out.joints.detach().cpu().numpy().squeeze()
print(f"✅ smplx FLAME 加载成功")
print(f"  vertices: {v.shape}")
print(f"  joints  : {j.shape} (含 68 landmark)")
print(f"  expr 维度: {m.num_expression_coeffs}")
print(f"  faces   : {np.asarray(m.faces).shape}")

# 68 点 landmark 合理性：应落在头骨范围内（FLAME 坐标系，米制）
lm = j[-68:] if j.shape[0] >= 68 else j
print(f"  lm68 范围: x[{lm[:,0].min():.3f},{lm[:,0].max():.3f}] "
      f"y[{lm[:,1].min():.3f},{lm[:,1].max():.3f}] z[{lm[:,2].min():.3f},{lm[:,2].max():.3f}]")
assert v.shape == (5023, 3), v.shape
assert lm.shape[0] == 68, f"68 点 landmark 缺失: {lm.shape}"
print("🎉 全部通过")
