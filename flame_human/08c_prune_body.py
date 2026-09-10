#!/usr/bin/env python3
"""08c_prune_body.py — body finetune 产物的后处理剪枝（治 floater + 头部重影）。

看图诊断（2026-09-10，多模态目检 composite 三联图）：
  finetune 后的 body 在人物轮廓周围爆出黑色尖刺 floater（暗色高斯从
  silhouette 向外辐射），且 12.6% 高斯长进头盒与 head 分支重影。
  mask 外**亮度**惩罚抓不住暗色 floater（黑针亮度低），故改后处理：

  1. **头盒 3D 剪枝**：head_box 内的点全删（头部归 head 分支）；
  2. **mask 内投票剪枝**（07a 投票的反向）：每个高斯投影到 59 帧，
     统计落在 person mask 内的比例；floater 悬在轮廓外/背景前，多数
     视角投影在 mask 外 → 低分删除。正常 body 表面点多数视角在 mask 内。

Env: IN_PLY / OUT_PLY / ALIGN_JSON / SOURCE_DIR / MASKS_DIR / PID /
     KEEP_RATIO / HEAD_BBOX_MARGIN
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_cameras, proj_matrix  # noqa: E402
from split_body_scene import head_box  # noqa: E402


def log(m):
    print(m, flush=True)


def main():
    results_dir = Path(os.environ.get("RESULTS_DIR", ""))
    in_ply = Path(os.environ.get(
        "IN_PLY", f"{results_dir}/08b_finetune_body/body_ft_p0.ply"))
    out_ply = Path(os.environ.get(
        "OUT_PLY", f"{results_dir}/08c_pruned/body_ft_pruned_p0.ply"))
    align_json = Path(os.environ.get(
        "ALIGN_JSON", f"{results_dir}/05_align/head_align.json"))
    source_dir = os.environ.get("SOURCE_DIR", "")
    masks_dir = Path(os.environ.get(
        "MASKS_DIR", f"{results_dir}/01c_sam3_person_masks"))
    pid = os.environ.get("PID", "0")
    keep_ratio = float(os.environ.get("KEEP_RATIO", "0.5"))
    margin = float(os.environ.get("HEAD_BBOX_MARGIN", "0.15"))

    from plyfile import PlyData, PlyElement
    pd = PlyData.read(str(in_ply))
    v = pd["vertex"]
    names = [p.name for p in v.properties]
    xyz = np.stack([np.asarray(v[k]) for k in "xyz"], 1)
    n = len(xyz)
    log(f"✂️  [08c] prune {in_ply.name}: {n:,} 高斯")

    # ── 1. 头盒 3D 剪枝 ──────────────────────────────────────────────
    mesh_dir = Path(os.environ.get(
        "MESH_DIR", f"{results_dir}/06_avatar_gs"))
    lo, hi, _, r = head_box(mesh_dir / f"avatar_mesh_p{pid}.npz",
                            mesh_dir / f"avatar_bind_p{pid}.npz", margin)
    in_head = np.all((xyz >= lo) & (xyz <= hi), 1)
    log(f"  🗃️ 头盒内删除: {int(in_head.sum()):,} ({in_head.mean():.1%})")

    # ── 2. mask 内投票剪枝 ───────────────────────────────────────────
    import json
    from PIL import Image
    views = read_cameras(source_dir)
    view_by = {vv["stem"]: vv for vv in views}
    stems = json.loads(align_json.read_text())["persons"][pid]["frames"]
    pid2 = pid.zfill(2)

    inside = np.zeros(n, dtype=np.int32)
    front = np.zeros(n, dtype=np.int32)
    for s in stems:
        vv = view_by.get(s)
        if vv is None:
            continue
        mp = masks_dir / f"{s}.p{pid2}.alpha.png"
        if not mp.exists():
            mp = masks_dir / f"{s}.p{pid2}.mask.png"
        if not mp.exists():
            continue
        m = np.asarray(Image.open(mp).convert("L").resize(
            (vv["W"], vv["H"]), Image.LANCZOS), dtype=np.float32) / 255.0
        P, _ = proj_matrix(vv)
        homo = np.c_[xyz, np.ones(n)]
        pr = homo @ P.T                       # (n,3)
        z = pr[:, 2]
        ok = z > 1e-6
        u = np.where(ok, pr[:, 0] / np.clip(z, 1e-6, None), -1).astype(int)
        w = np.where(ok, pr[:, 1] / np.clip(z, 1e-6, None), -1).astype(int)
        inimg = ok & (u >= 0) & (u < vv["W"]) & (w >= 0) & (w < vv["H"])
        front += inimg.astype(np.int32)
        idx = np.nonzero(inimg)[0]
        inside[idx] += (m[w[idx], u[idx]] > 0.5).astype(np.int32)

    score = np.where(front > 0, inside / np.clip(front, 1, None), 1.0)
    low_vote = (front > 0) & (score < keep_ratio)
    log(f"  🗳️  投票剪枝 (keep>={keep_ratio}): 删 {int(low_vote.sum()):,} "
        f"({low_vote.mean():.1%}); 无观测点 {int((front == 0).sum()):,}")

    keep = ~in_head & ~low_vote & (front > 0)
    log(f"  ✅ 保留 {int(keep.sum()):,} / {n:,}")

    arr = np.empty(int(keep.sum()),
                   dtype=[(nm, v[nm].dtype) for nm in names])
    for nm in names:
        arr[nm] = np.asarray(v[nm])[keep]
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")],
            text=False).write(str(out_ply))
    log(f"💾 {out_ply}")


if __name__ == "__main__":
    main()
