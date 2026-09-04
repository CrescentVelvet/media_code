#!/usr/bin/env python3
"""face_center_3d.py — Compute 3D face center from 3DGS point cloud + face masks.

Strategy:
  1. Load 3DGS point cloud (PLY) — millions of 3D gaussians.
  2. Load face masks (SAM2 `<stem>.mask.png` 或 SAM3 多人 `<stem>.p{pid}.mask.png`).
  3. For each 3D point, project to all camera views using COLMAP poses.
  4. A point is a "face point" if it falls inside the mask in >= N frames
     (multi-view consistency, default N=15).
  5. Face 3D center = centroid of all face points.

Multi-person (SAM3 video backend 输出):
  masks 目录含 `<stem>.p{pid}.mask.png` 时自动进入多人模式（或 --multi_person 强制）。
  每个 pid 独立执行投影→命中→质心→sanity 流程, 输出 per-person 中心:
    persons: {pid: {center, n_face_points, sanity_*, per_view_coverage}}
  顶层 center 兼容旧消费者 = 面部点数最多那人的中心。

Output: 06c_face_center.json with {center: [x,y,z], n_face_points, n_total_points,
       per_view_coverage: {frame: coverage%}} (single) / persons{...} (multi)

Env vars:
  PLY_PATH     : 3DGS point cloud PLY
  SOURCE_DIR   : COLMAP source (sparse/0/ has cameras.txt, images.txt)
  MASKS_DIR    : face masks folder
  OUTPUT_JSON  : output JSON path
  MIN_HITS     : min frames a point must hit to count as face (default 15)

min_hits 的取舍:
  太低(<=3)会把「人脸后方的背景点」也算进来 —— 背景点从不同视角投影同样
  能落进人脸的 2D 区域, 于是质心被拉向人脸后方的墙/物体。
  默认 15 (77 个 mask 视角中命中 >=15) 能压住这类噪声。
"""
import os
import re
import sys
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_ply_points(ply_path):
    """Load PLY as (N, 3) xyz array. Minimal PLY parser for gaussian-splatting output."""
    print(f"📥 loading PLY: {ply_path}")
    t0 = time.time()
    with open(ply_path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"element vertex" in line:
                n_vertices = int(line.split()[-1])
            if b"end_header" in line:
                break
        # Read binary data
        # gaussian-splatting PLY: x,y,z, ... (float32) + opacity + SH ...
        # We only need x,y,z — first 3 float32 per vertex
        # But need to skip the rest. Parse property list from header.
        props = []
        for line in header.split(b"\n"):
            line = line.strip()
            if line.startswith(b"property") and b"float" in line:
                props.append(("float", 4))
            elif line.startswith(b"property") and b"uchar" in line:
                props.append(("uchar", 1))

        stride = sum(s for _, s in props)
        n_floats_before_xyz = 0  # x,y,z are first 3 in GS PLY
        xyz_offset = 0
        # Actually, parse properly: find x, y, z property positions
        prop_names = []
        for line in header.split(b"\n"):
            line = line.strip().decode("ascii", errors="ignore")
            if line.startswith("property"):
                parts = line.split()
                if len(parts) >= 3:
                    prop_names.append(parts[2])

        xyz_idx = [prop_names.index("x"), prop_names.index("y"), prop_names.index("z")]
        # Build dtype
        dtype_fields = []
        offset = 0
        for i, (pname, (ptype, psize)) in enumerate(zip(prop_names, props)):
            if ptype == "float":
                dtype_fields.append((pname, "<f4"))
            else:
                dtype_fields.append((pname, "u1"))
        dt = np.dtype(dtype_fields)

        data = np.frombuffer(f.read(n_vertices * stride), dtype=dt)
        xyz = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float64)
    print(f"  {len(xyz)} points, {time.time()-t0:.1f}s")
    return xyz


def parse_colmap_cameras(source_dir):
    """Parse COLMAP cameras.txt + images.txt → list of (stem, qvec, tvec, fx, fy, cx, cy, W, H).

    Handles layouts: sparse/0/cameras.txt, sparse/0_text/cameras.txt,
    or source_dir/cameras.txt.
    """
    cam_file = Path(source_dir) / "sparse" / "0" / "cameras.txt"
    img_file = Path(source_dir) / "sparse" / "0" / "images.txt"
    if not cam_file.exists():
        # Try sparse/0_text/ (BA output text format)
        cam_file = Path(source_dir) / "sparse" / "0_text" / "cameras.txt"
        img_file = Path(source_dir) / "sparse" / "0_text" / "images.txt"
    if not cam_file.exists():
        cam_file = Path(source_dir) / "cameras.txt"
        img_file = Path(source_dir) / "images.txt"

    # cameras.txt
    cameras = {}
    with open(cam_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cid = int(parts[0])
            model = parts[1]
            W, H = int(parts[2]), int(parts[3])
            params = list(map(float, parts[4:]))
            if model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            elif model == "SIMPLE_PINHOLE":
                f, cx, cy = params[:3]
                fx = fy = f
            else:
                continue
            cameras[cid] = dict(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy)

    # images.txt
    views = []
    with open(img_file) as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        img_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)  # qw,qx,qy,qz
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        cid = int(parts[8])
        name = parts[9]
        if cid not in cameras:
            continue
        cam = cameras[cid]
        stem = os.path.splitext(name)[0]
        views.append(dict(stem=stem, qvec=qvec, tvec=tvec, **cam))

    print(f"  parsed {len(views)} camera views")
    return views


def load_masks(masks_dir, stems):
    """Load SAM2 binary masks. Returns dict stem -> bool array (H, W)."""
    masks = {}
    for stem in stems:
        mask_path = os.path.join(masks_dir, f"{stem}.mask.png")
        if os.path.exists(mask_path):
            m = np.array(Image.open(mask_path).convert("L"))
            masks[stem] = m > 127
    return masks


def load_masks_multi(masks_dir, stems):
    """Load per-person masks `<stem>.p{pid}.mask.png`.

    Returns {pid_int: {stem: bool array (H, W)}}. Only p-masks are read;
    旧命名的 {stem}.mask.png 在多人模式下被忽略 (避免单/多混用错配).
    """
    stem_set = set(stems)
    pat = re.compile(r"^(.+)\.p(\d+)\.mask\.png$")
    persons = {}
    for f in os.listdir(masks_dir):
        m = pat.match(f)
        if not m or m.group(1) not in stem_set:
            continue
        pid = int(m.group(2))
        arr = np.array(Image.open(os.path.join(masks_dir, f)).convert("L"))
        persons.setdefault(pid, {})[m.group(1)] = arr > 127
    return persons


def project_points(xyz, view):
    """Project 3D points (N,3) to 2D pixel coords. Returns (N,2) and validity mask."""
    q = view["qvec"]
    t = view["tvec"]
    # Rotation from quaternion (COLMAP: world->camera)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])
    # World->camera
    cam_pts = (R @ xyz.T).T + t  # (N, 3)
    # Only points in front of camera (z > 0)
    valid = cam_pts[:, 2] > 0
    # Project
    z = cam_pts[:, 2:3].copy()
    z[z == 0] = 1e-6
    uv = np.zeros((len(xyz), 2))
    uv[:, 0] = (view["fx"] * cam_pts[:, 0] / z[:, 0] + view["cx"])
    uv[:, 1] = (view["fy"] * cam_pts[:, 1] / z[:, 0] + view["cy"])
    return uv, valid


def compute_face_center(xyz, views, masks, min_hits, min_face_points, label=""):
    """Core per-person flow: project → hit counting → threshold → centroid → sanity.

    Returns dict with center / spread / n_face_points / mh / per_view_coverage /
    sanity results. 打印过程日志 (带 label 前缀便于多人区分).
    """
    tag = f"[p{label}] " if label != "" else ""

    # 1. Project + count hits
    print(f"\n{tag}🎯 projecting {len(xyz)} points to {len(masks)} views...")
    t0 = time.time()
    hit_counts = np.zeros(len(xyz), dtype=np.int32)
    per_view_cov = {}

    for vi, view in enumerate(views):
        if view["stem"] not in masks:
            continue
        mask = masks[view["stem"]]
        uv, valid = project_points(xyz, view)
        in_bounds = (
            valid &
            (uv[:, 0] >= 0) & (uv[:, 0] < view["W"]) &
            (uv[:, 1] >= 0) & (uv[:, 1] < view["H"])
        )
        mask_hits = np.zeros(len(xyz), dtype=bool)
        valid_idx = np.where(in_bounds)[0]
        if len(valid_idx) > 0:
            px = uv[valid_idx, 0].astype(int)
            py = uv[valid_idx, 1].astype(int)
            px = np.clip(px, 0, mask.shape[1] - 1)
            py = np.clip(py, 0, mask.shape[0] - 1)
            mask_hits[valid_idx] = mask[py, px]
        hit_counts += mask_hits.astype(np.int32)
        cov = mask.sum() / mask.size * 100
        per_view_cov[view["stem"]] = round(cov, 2)
        if (vi + 1) % 20 == 0 or vi == 0:
            n_hit = mask_hits.sum()
            print(f"{tag}  [{vi+1}/{len(views)}] {view['stem']}: cov={cov:.1f}%, pts_in_mask={n_hit}")

    print(f"{tag}  projection done in {time.time()-t0:.1f}s")

    # 2. Threshold
    print(f"\n{tag}📊 hit-count 分布 (命中 >=N 个 mask 视角的点数):")
    for th in (1, 3, 5, 10, 15, 20, 30, 40):
        print(f"{tag}    hits>={th:2d}: {(hit_counts >= th).sum():>8d} pts")

    mh = min_hits
    face_pts_mask = hit_counts >= mh
    n_face = int(face_pts_mask.sum())
    print(f"\n{tag}  face points (hits>={mh}): {n_face} / {len(xyz)} ({n_face/len(xyz)*100:.2f}%)")

    # 逐级放宽 (下限 5)。刻意不退回 min_hits=1:
    # 命中 1~2 次的点绝大多数是背景, 退回 1 等于放弃了多视角一致性约束,
    # 会让质心重新飘到人脸后方的墙上。
    FLOOR = 5
    while n_face < min_face_points and mh > FLOOR:
        prev = mh
        mh = max(FLOOR, mh // 2)
        face_pts_mask = hit_counts >= mh
        n_face = int(face_pts_mask.sum())
        print(f"{tag}  ⚠️  hits>={prev} 只有 {n_face} 点 (<{min_face_points}), "
              f"放宽到 >= {mh} → {n_face} 点")

    if n_face == 0:
        return None

    if mh != min_hits:
        print(f"{tag}  ⚠️ 最终 min_hits={mh} (请求值 {min_hits}), 中心可信度下降")

    # 3. Centroid = face 3D center
    center = xyz[face_pts_mask].mean(axis=0)
    spread = xyz[face_pts_mask].std(axis=0)
    print(f"\n{tag}  face 3D center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"{tag}  spread (std):   [{spread[0]:.4f}, {spread[1]:.4f}, {spread[2]:.4f}]")

    # 稳健性参考: 命中次数最高的 10% 点的质心。
    cnts = hit_counts[face_pts_mask]
    thr = float(np.percentile(cnts, 90)) if len(cnts) else 0.0
    top_mask = face_pts_mask & (hit_counts >= thr)
    center_top = xyz[top_mask].mean(axis=0) if top_mask.sum() else center.copy()
    d_top = float(np.linalg.norm(center_top - center))
    print(f"{tag}  robust ref (top-10% 命中点质心): "
          f"[{center_top[0]:.4f}, {center_top[1]:.4f}, {center_top[2]:.4f}]  偏差 {d_top:.4f}")

    # 4. Sanity check: 候选中心回投到各 mask 视角, 自动选命中更好的那个。
    def sanity(F):
        c_xyz = np.asarray(F).reshape(1, 3)
        n_in, n_tot, offs = 0, 0, []
        for view in views:
            if view["stem"] not in masks:
                continue
            n_tot += 1
            uv, valid = project_points(c_xyz, view)
            if not valid[0]:
                continue
            u, v = float(uv[0, 0]), float(uv[0, 1])
            if not (0 <= u < view["W"] and 0 <= v < view["H"]):
                continue
            m = masks[view["stem"]]
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            offs.append(math.hypot(u - xs.mean(), v - ys.mean()))
            if m[int(v), int(u)]:
                n_in += 1
        return n_in, n_tot, float(np.mean(offs)) if offs else 1e9

    print(f"\n{tag}🔍 sanity check: 候选中心回投到 mask 视角...")
    n_in, n_tot, mean_off = sanity(center)
    pct = n_in / max(n_tot, 1) * 100
    print(f"{tag}  完整质心   [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]: "
          f"{n_in}/{n_tot} ({pct:.0f}%), 平均像素距离 {mean_off:.0f}px")
    n_in_t, n_tot_t, mean_off_t = sanity(center_top)
    pct_t = n_in_t / max(n_tot_t, 1) * 100
    print(f"{tag}  top10%质心 [{center_top[0]:.4f}, {center_top[1]:.4f}, {center_top[2]:.4f}]: "
          f"{n_in_t}/{n_tot_t} ({pct_t:.0f}%), 平均像素距离 {mean_off_t:.0f}px")

    use_top = (pct_t, -mean_off_t) > (pct, -mean_off)
    if use_top:
        center, pct, mean_off = center_top, pct_t, mean_off_t
        print(f"{tag}  ✅ 选用 top10% 质心作为最终 face center")
    else:
        print(f"{tag}  ✅ 选用完整质心作为最终 face center")
    if pct < 30:
        print(f"{tag}  ⚠️ 仅 {pct:.0f}% 视角命中 —— 中心仍偏, 建议实现深度过滤 (方案 A)")

    return {
        "center": [float(center[0]), float(center[1]), float(center[2])],
        "center_full_centroid": [float(xyz[face_pts_mask].mean(axis=0)[0]),
                                  float(xyz[face_pts_mask].mean(axis=0)[1]),
                                  float(xyz[face_pts_mask].mean(axis=0)[2])],
        "spread": [float(spread[0]), float(spread[1]), float(spread[2])],
        "center_top10pct": [float(center_top[0]), float(center_top[1]), float(center_top[2])],
        "center_source": "top10pct" if use_top else "full_centroid",
        "n_face_points": int(n_face),
        "n_total_points": int(len(xyz)),
        "min_hits": int(mh),
        "min_hits_requested": int(min_hits),
        "n_mask_views": len(masks),
        "sanity_center_in_mask_views": f"{n_in}/{n_tot}",
        "sanity_center_pct": round(pct, 1),
        "sanity_mean_offset_px": round(mean_off, 1),
        "per_view_coverage": per_view_cov,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--source_dir", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min_hits", type=int, default=15)
    ap.add_argument("--min_face_points", type=int, default=200,
                    help="若 face points 少于此值则逐级放宽 min_hits (下限 5)")
    ap.add_argument("--multi_person", action="store_true",
                    help="强制多人模式 (默认自动检测: 有 <stem>.p{pid}.mask.png 即多人)")
    args = ap.parse_args()

    print(f"🧑 face 3D center computation")
    print(f"  📂 PLY: {args.ply}")
    print(f"  📂 source: {args.source_dir}")
    print(f"  📂 masks: {args.masks_dir}")
    print(f"  📐 min_hits: {args.min_hits}")
    print("")

    # 1. Load points
    xyz = load_ply_points(args.ply)

    # 2. Parse cameras
    print("\n📷 parsing COLMAP cameras...")
    views = parse_colmap_cameras(args.source_dir)
    stems = [v["stem"] for v in views]

    # 3. Load masks — 自动检测多人
    multi = args.multi_person
    persons_masks = {}
    masks = {}
    if not multi:
        print("\n🎭 loading face masks...")
        masks = load_masks(args.masks_dir, stems)
        if not masks:
            print("  无 {stem}.mask.png, 尝试多人 p-mask ...")
            multi = True
        else:
            print(f"  {len(masks)} masks found (of {len(views)} views)")
    if multi:
        print("\n🎭 loading per-person face masks (multi-person mode)...")
        persons_masks = load_masks_multi(args.masks_dir, stems)
        if not persons_masks:
            sys.exit("no masks found (既无 {stem}.mask.png 也无 {stem}.p{pid}.mask.png)")
        pids = sorted(persons_masks)
        print(f"  {len(pids)} person(s): p{pids[0]:02d}..p{pids[-1]:02d}, "
              f"mask 帧数 {[len(persons_masks[p]) for p in pids]}")

    if not multi:
        if not masks:
            sys.exit("no masks found")
        result = compute_face_center(xyz, views, masks, args.min_hits, args.min_face_points)
        if result is None:
            sys.exit("no face points — 检查 MASKS_DIR 与 PLY 是否配套")
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 saved: {args.output}")
        c = result["center"]
        print(f"\n✅ face 3D center: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]")
        return

    # 4. Multi-person: 每个 pid 独立计算
    persons_result = {}
    for pid in sorted(persons_masks):
        print(f"\n{'='*60}\n👤 person p{pid:02d}")
        res = compute_face_center(xyz, views, persons_masks[pid],
                                  args.min_hits, args.min_face_points,
                                  label=f"{pid:02d}")
        if res is None:
            print(f"  ⚠️ p{pid:02d}: 无 face points, 跳过")
            continue
        persons_result[str(pid)] = res

    if not persons_result:
        sys.exit("multi-person 模式下没有任何 person 算出 face center")

    # 主人物 = face points 最多者; 顶层字段兼容旧单人流消费者
    primary_pid = max(persons_result, key=lambda k: persons_result[k]["n_face_points"])
    primary = persons_result[primary_pid]
    result = dict(primary)
    result["multi_person"] = True
    result["primary_person"] = primary_pid
    result["n_persons"] = len(persons_result)
    result["persons"] = persons_result

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 saved: {args.output}")

    print(f"\n✅ multi-person face centers ({len(persons_result)} 人):")
    for pid in sorted(persons_result, key=lambda k: -persons_result[k]["n_face_points"]):
        c = persons_result[pid]["center"]
        mark = " ← primary" if pid == primary_pid else ""
        print(f"  p{int(pid):02d}: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]  "
              f"pts={persons_result[pid]['n_face_points']}  "
              f"sanity={persons_result[pid]['sanity_center_in_mask_views']}{mark}")


if __name__ == "__main__":
    main()
