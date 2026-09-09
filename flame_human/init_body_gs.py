#!/usr/bin/env python3
"""init_body_gs.py — 阶段七：BodyGaussian 初始化（point-to-triangle 切分 + 显式补缝）。

两件事，都是对参考实现的修正：

1. **切分距离用 point-to-triangle，不是 KNN 到最近顶点**
   `knn_points(body, avatar_v, K=1)` 量的是「到最近 face **顶点**」的距离，
   直接受 face mesh 顶点密度支配。自研 20971 → FLAME 5023，密度降到 1/4：
       平均最近邻间距 × √(20971/5023) = × 2.04，平方距离 × 4.17
   而 max_dist 是按 bbox 对角线算的，没考虑密度 → 同一个点从「该删」变成「留下」，
   **等价于删除半径缩到 49%**。后果不是 gap 变大，而是问题性质翻转：
   从「脖子到肩膀空洞」变成「head/body 重叠 → 重影」。
   点到**表面**的距离与顶点密度无关，换模型/换 LOD 行为一致。

2. **显式补缝**
   face mesh 只到脖子，body 点云又把脖子附近删了，中间一圈没人覆盖。
   参考实现指望 alpha blending 重叠 + densification 兜底，不可靠。
   这里在 face mesh 的下边界（boundary edges，主要是颈部开口）
   沿「法线 + 径向」外推 2~3cm 生成几圈桥接高斯。

Env: BODY_PLY / MESH_DIR / OUT_DIR / PERSONS
     MIN_DIST_M   距头部表面多近就删（默认 0.02 m，等价 head 尺度下的 2cm）
     BRIDGE_MM    补缝外推距离（默认 25 mm）
     BRIDGE_STEPS 外推分几圈（默认 3）
"""
import os
import sys
import json
from pathlib import Path

import numpy as np


def log(m):
    print(m, flush=True)


def load_ply(path):
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    names = [p.name for p in v.properties]
    return names, {n: np.asarray(v[n]) for n in names}


def save_ply(path, names, data):
    from plyfile import PlyData, PlyElement
    n = len(next(iter(data.values())))
    arr = np.empty(n, dtype=[(nm, data[nm].dtype) for nm in names])
    for nm in names:
        arr[nm] = data[nm]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def surface_distance(pts, verts, faces):
    """点到 mesh **表面**的距离（无符号）。优先 trimesh，退回 numpy 近似。"""
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        pq = trimesh.proximity.ProximityQuery(mesh)
        return np.abs(np.asarray(pq.signed_distance(pts)))
    except Exception as e:
        log(f"  ⚠️  trimesh 不可用（{type(e).__name__}: {e}），"
            f"退回 KNN 到顶点近似（精度会受顶点密度影响，仅供兜底）")
        from scipy.spatial import cKDTree
        d, _ = cKDTree(verts).query(pts)
        return d


def boundary_vertex_ring(verts, faces):
    """mesh 开口边界上的顶点（FLAME 的颈部开口就在这里）。"""
    edge_count = {}
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            k = (min(a, b), max(a, b))
            edge_count[k] = edge_count.get(k, 0) + 1
    bnd = [v for (a, b), c in edge_count.items() if c == 1 for v in (a, b)]
    return np.unique(np.asarray(bnd, dtype=np.int64))


def vertex_normals(verts, faces):
    n = np.zeros_like(verts)
    v0 = verts[faces[:, 0]]
    fn = np.cross(verts[faces[:, 1]] - v0, verts[faces[:, 2]] - v0)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def bridge_points(verts, faces, n_per_ring, max_mm, steps, rng, scale):
    """沿 mesh 下边界外推几圈 → 桥接高斯位置（世界单位）。

    max_mm 是**米制毫米**（与 head 尺度无关），乘 scale 转成 SfM 世界单位；
    verts 已经是世界单位，两者不能直接相加。
    """
    bnd = boundary_vertex_ring(verts, faces)
    if len(bnd) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    # 只取下半部分（颈部开口在下方；用 y 中位切）
    cy = np.median(verts[:, 1])
    lower = bnd[verts[bnd, 1] < cy]
    if len(lower) < 8:
        lower = bnd
    nrm = vertex_normals(verts, faces)
    centroid = verts.mean(0)
    radial = verts[lower] - centroid
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    # 外推方向：法线与径向各半（径向保证是"往下往外"而不是纯水平外扩）
    d = 0.5 * nrm[lower] + 0.5 * radial
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)

    out = []
    for si in range(1, steps + 1):
        t = (si / steps) * (max_mm / 1000.0) * scale
        jitter = rng.normal(scale=t * 0.15, size=(n_per_ring, 1))
        idx = rng.integers(0, len(lower), size=n_per_ring)
        out.append(verts[lower][idx] + d[idx] * (t + jitter))
    return np.vstack(out)


def main():
    results_dir = os.environ.get("RESULTS_DIR", "")
    upstream = os.environ.get("UPSTREAM_DIR", "")
    body_ply_tpl = os.environ.get(
        "BODY_PLY", f"{upstream}/03h_person_scene_split/body_gs_p{{pid}}.ply")
    mesh_dir = Path(os.environ.get("MESH_DIR", f"{results_dir}/06_avatar_gs"))
    out_dir = Path(os.environ.get("OUT_DIR", f"{results_dir}/07_body_gs"))
    persons = [p.strip() for p in os.environ.get("PERSONS", "0,1,2").split(",")
               if p.strip() != ""]
    min_dist_mm = float(os.environ.get("MIN_DIST_MM", "20"))
    bridge_mm = float(os.environ.get("BRIDGE_MM", "25"))
    bridge_steps = int(os.environ.get("BRIDGE_STEPS", "3"))
    n_ring = int(os.environ.get("BRIDGE_N", "1500"))
    seed = int(os.environ.get("SEED", "0"))

    log("🧍 [阶段七] BodyGaussian 初始化（point-to-triangle + 补缝）")
    log(f"  📏 切分阈值 {min_dist_mm}mm / 补缝外推 ≤{bridge_mm}mm"
        f"（均为米制，会按各人 scale 换算到 SfM 世界单位）")

    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for pid in persons:
        ply = Path(body_ply_tpl.format(pid=pid.zfill(2)))
        mesh_npz = mesh_dir / f"avatar_mesh_p{pid}.npz"
        if not ply.exists():
            log(f"  ⚠️ p{pid}: 缺 body 点云 {ply}，跳过")
            continue
        if not mesh_npz.exists():
            log(f"  ⚠️ p{pid}: 缺头部网格 {mesh_npz}，跳过")
            continue

        names, data = load_ply(ply)
        xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
        z = np.load(mesh_npz)
        verts, faces = z["verts_world"], z["faces"]

        # scale：FLAME 米 → SfM 世界单位。阈值/外推都用米制给，这里换算
        bind_npz = mesh_dir / f"avatar_bind_p{pid}.npz"
        scale = 1.0
        if bind_npz.exists():
            scale = float(np.load(bind_npz)["scale"])
        else:
            log(f"  ⚠️ p{pid}: 缺 {bind_npz.name}，scale 用 1.0（阈值可能不准）")
        min_dist = min_dist_mm / 1000.0 * scale

        log(f"  🔧 p{pid}: body {len(xyz):,} 点, head mesh {verts.shape[0]} 顶点, "
            f"scale={scale:.4g}")

        d = surface_distance(xyz, verts, faces)
        # 世界尺度：网格已是 SfM 单位，min_dist 给的也是世界单位
        keep = d > min_dist
        log(f"     距头部表面 ≤ {min_dist:.3f} 的点: {int((~keep).sum()):,} "
            f"（删除），保留 {int(keep.sum()):,}")

        kept = {n: data[n][keep] for n in names}

        # ── 补缝：桥接高斯 ───────────────────────────────────────────────
        bp = bridge_points(verts, faces, n_ring, bridge_mm, bridge_steps, rng,
                           scale)
        if len(bp) == 0:
            log("     ⚠️  头部 mesh 无开口边界，跳过补缝")
        else:
            n_new = len(bp)
            extra = {}
            for n in names:
                if n in ("x", "y", "z"):
                    continue
                if n.startswith("f_dc"):
                    extra[n] = np.full(n_new, 0.5, dtype=data[n].dtype)
                elif n == "opacity":
                    extra[n] = np.full(n_new, 0.1, dtype=data[n].dtype)
                elif n.startswith("scale_"):
                    extra[n] = np.full(n_new, np.log(min_dist * 0.5),
                                       dtype=data[n].dtype)
                elif n.startswith("rot_"):
                    extra[n] = np.zeros(n_new, dtype=data[n].dtype)
                elif n in ("nx", "ny", "nz"):
                    extra[n] = np.zeros(n_new, dtype=data[n].dtype)
                else:
                    extra[n] = np.zeros(n_new, dtype=data[n].dtype)
            extra["x"], extra["y"], extra["z"] = bp[:, 0], bp[:, 1], bp[:, 2]
            merged = {n: np.concatenate([kept[n], extra[n]]) for n in names}
            log(f"     ➕ 补缝桥接高斯 {n_new:,} 点 "
                f"（{bridge_steps} 圈 × {n_ring}, 外推 ≤{bridge_mm}mm）")
            kept = merged

        dst = out_dir / f"body_p{pid.zfill(2)}.ply"
        save_ply(dst, names, kept)
        stats[pid] = {"body_in": int(len(xyz)), "dropped": int((~keep).sum()),
                      "bridge": int(len(bp)), "out": int(len(kept["x"]))}
        log(f"     ✅ {dst.name}: {len(kept['x']):,} 点")

    (out_dir / "body_stats.json").write_text(json.dumps(stats, indent=1))
    log(f"💾 {out_dir}")
    log("🎉 阶段七完成。下一步：bash 08_train.sh")


if __name__ == "__main__":
    main()
