"""_debug_verify_head.py — 验证 body finetune 是否往头部生长（composite 变差根因）。

07a 提取 body 时已用 head_box 剔除头部点（dropped_in_head_bbox=32012），
所以 old body 的 in_head 应 ≈0。若 ft 后 in_head 大增 → body 被 mask 内的
头部 GT 拉着往头部生长 → composite 里 head 分支 + body 分支双重渲染头部
（重影）→ PSNR 下降。这能解释为何 mask 外惩罚无效（头部在 mask 内）。
"""
import os
import sys
import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_body_scene import head_box  # noqa: E402
from plyfile import PlyData  # noqa: E402

rd = os.environ["RESULTS_DIR"]
meshes = sorted(glob.glob(f"{rd}/06_avatar_gs/avatar_mesh_p*.npz"))
binds = sorted(glob.glob(f"{rd}/06_avatar_gs/avatar_bind_p*.npz"))
lo, hi, hc, r = head_box(meshes[0], binds[0], 0.15)
print(f"head_box r={r:.3f} lo={lo.round(3)} hi={hi.round(3)}")


def stats(p):
    v = PlyData.read(p)["vertex"]
    xyz = np.stack([np.asarray(v[k]) for k in "xyz"], 1)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"])))
    in_head = np.all((xyz >= lo) & (xyz <= hi), 1)
    hop = float(op[in_head].mean()) if in_head.any() else 0.0
    return len(xyz), int(in_head.sum()), float(in_head.mean()), \
        float(np.median(op)), hop


for tag, p in (("old   ", f"{rd}/07_body_gs_src/body_gs_p0.ply"),
               ("ft-pen", f"{rd}/08b_finetune_body/body_ft_p0.ply")):
    n, nh, fh, mop, hop = stats(p)
    print(f"{tag}: n={n:,} in_head={nh:,}({fh:.1%}) "
          f"op_p50={mop:.3f} op_in_head={hop:.3f}")
