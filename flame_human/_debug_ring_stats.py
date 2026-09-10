"""_debug_ring_stats.py — 人物环带暗度归因（白底单分支渲染上量化）。

读 br_*.png 五联图（GT|scene|body|head|full），在白底分支面板上测：
  - inside: person mask 内
  - ring  : mask 外扩 30px 的环带（人物周围，黑雾高发区）
  - far   : 远处背景
白底下亮度越低 = 该分支在该区域涂的暗色高斯越多。
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

results = Path(os.environ["RESULTS_DIR"])
NAMES = ["gt", "scene", "body", "head", "full"]
frames = [("029", "679455776749000"), ("026", "679451010256156"),
          ("015", "679449710302000")]

for idx, stem in frames:
    p = results / "composite_branch" / f"br_{idx}_{stem}.png"
    img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    H, W5 = img.shape[:2]
    W = W5 // 5
    panels = {nm: img[:, k * W:(k + 1) * W] for k, nm in enumerate(NAMES)}
    m = Image.open(
        results / "01c_sam3_person_masks" / f"{stem}.p00.alpha.png").convert("L")
    if m.size != (W, H):
        m = m.resize((W, H), Image.LANCZOS)
    mask = np.asarray(m, dtype=np.float32) / 255.0
    dil = np.asarray(m.filter(ImageFilter.MaxFilter(61)),
                     dtype=np.float32) / 255.0
    inside = np.broadcast_to((mask > 0.5)[..., None], panels["gt"].shape)
    ring = np.broadcast_to(((dil > 0.5) & (mask <= 0.5))[..., None],
                           panels["gt"].shape)
    far = np.broadcast_to(((dil <= 0.5))[..., None], panels["gt"].shape)
    print(f"--- frame {idx} ({stem}) ---")
    for nm in NAMES:
        a = panels[nm]
        print(f"  {nm:5s}  inside={a[inside].mean():.3f}  "
              f"ring={a[ring].mean():.3f}  far={a[far].mean():.3f}")
