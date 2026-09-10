#!/usr/bin/env python3
"""split_body_scene.py — 从上游 3DGS 场景点云提取 p0 身体点云（07 的输入）。

上游 vggt_human 的 03h 需要 04b_model_3dgs_ba（iteration_30000）+ 03e 头网格
等一串前置产物，verify 数据上没有跑过。本链路只要 body_gs_p{pid}.ply，
做法对齐 vggt_human split_person_scene.py 的多视角投票，但取材不同：

  输入: 上游 model_3dgs_nn/point_cloud/iteration_7000/point_cloud.ply
        （nn = no-noise？完整训练的场景模型，7k iter）
        本链路 01c_sam3_person_masks/{stem}.p{pid}.alpha.png
        本链路 06_avatar_gs/avatar_mesh_p{pid}.npz（头网格，世界坐标）
        + avatar_bind_p{pid}.npz 的 head_center/头半径（建头包围盒）
  输出: $RESULTS_DIR/07_body_gs_src/body_gs_p{pid}.ply（只含 x/y/z + 颜色 SH，
        07 init_body_gs 只读 xyz；mask/alpha 等属性原样保留）

投票规则（对齐 vggt_human 03h）：
  - 点投到各帧，落入 p{pid} mask 且不落其他人 mask → votes[pid]+1
  - 深度门限: 距该帧头中心深度 ±DEPTH_GATE 相对量，防止远处人/物乱入
  - 归属: votes ≥ MIN_VOTES 且 votes/visible ≥ VOTE_RATIO
  - 头包围盒（06 头网格 bbox 外扩 HEAD_BBOX_MARGIN*头半径）内丢弃
    ——这部分点将由 08 的 head_gs 分支渲染替代，避免头/身重影

Env: RESULTS_DIR / UPSTREAM_DIR / SOURCE_DIR / FRAME_STRIDE / MIN_VOTES /
     VOTE_RATIO / DEPTH_GATE / HEAD_BBOX_MARGIN / PERSONS
"""
import os
import json
from pathlib import Path

import numpy as np

from colmap_io import read_cameras, qvec2rotmat

FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", "2"))
MIN_VOTES = int(os.environ.get("MIN_VOTES", "3"))
VOTE_RATIO = float(os.environ.get("VOTE_RATIO", "0.30"))
DEPTH_GATE = float(os.environ.get("DEPTH_GATE", "0.15"))
HEAD_BBOX_MARGIN = float(os.environ.get("HEAD_BBOX_MARGIN", "0.15"))
NEAR = 1e-6


def log(m):
    print(m, flush=True)


def load_gaussian_ply(path):
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    names = [p.name for p in v.properties]
    return names, {n: np.asarray(v[n]) for n in names}


def save_gaussian_ply(path, names, data, keep_idx):
    from plyfile import PlyData, PlyElement
    n = len(keep_idx)
    arr = np.empty(n, dtype=[(nm, data[nm].dtype) for nm in names])
    for nm in names:
        arr[nm] = data[nm][keep_idx]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def load_person_masks(mask_dir, stems, pids):
    """{pid: {stem: bool mask}}，读 01c 的 .p{pid:02d}.alpha.png。"""
    from PIL import Image
    out = {}
    for pid in pids:
        d = {}
        for stem in stems:
            p = mask_dir / f"{stem}.p{int(pid):02d}.alpha.png"
            if p.exists():
                im = np.asarray(Image.open(p).convert("L"))
                d[stem] = im > 127
        out[pid] = d
    return out


def head_box(mesh_npz, bind_npz, margin_mult):
    """06 头网格 bbox（世界坐标）外扩 margin_mult*头半径。"""
    z = np.load(mesh_npz)
    V = z["verts_world"]
    b = np.load(bind_npz)
    # bind 里存的是 05 对齐产物: head_center/scale 等
    hc = np.asarray(b["head_center"]) if "head_center" in b else V.mean(0)
    r = float(b["head_radius"]) if "head_radius" in b else \
        float(np.linalg.norm(V - hc, axis=1).max())
    mg = r * margin_mult
    return V.min(0) - mg, V.max(0) + mg, hc, r


def main():
    results_dir = Path(os.environ.get("RESULTS_DIR", ""))
    upstream = Path(os.environ.get("UPSTREAM_DIR", ""))
    source_dir = os.environ.get("SOURCE_DIR", f"{upstream}/source")
    mask_dir = Path(os.environ.get(
        "PERSON_MASKS_DIR", f"{results_dir}/01c_sam3_person_masks"))
    mesh_dir = Path(os.environ.get("MESH_DIR", f"{results_dir}/06_avatar_gs"))
    out_dir = Path(os.environ.get(
        "OUT_DIR", f"{results_dir}/07_body_gs_src"))
    persons = [p.strip() for p in os.environ.get("PERSONS", "0").split(",")
               if p.strip() != ""]

    # 场景高斯: 优先 iteration_7000 的 nn 模型（完整训练），退回 iteration_100
    cands = [
        upstream / "model_3dgs_nn" / "point_cloud" / "iteration_7000" / "point_cloud.ply",
        upstream / "model_3dgs" / "point_cloud" / "iteration_100" / "point_cloud.ply",
    ]
    gply = next((c for c in cands if c.exists()), None)
    if gply is None:
        log(f"❌ 找不到场景高斯: {[str(c) for c in cands]}")
        sys.exit(1)
    log(f"🧩 场景高斯: {gply}")

    views = read_cameras(source_dir)
    view_by_stem = {v["stem"]: v for v in views}
    stems = [v["stem"] for v in views][::FRAME_STRIDE]
    log(f"📷 相机 {len(views)} 帧, 投票用 {len(stems)} 帧 (stride={FRAME_STRIDE})")

    log("🧩 载入点云 ...")
    names, data = load_gaussian_ply(gply)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    N = len(xyz)
    log(f"   {N:,} 点, {len(names)} 属性")

    masks = load_person_masks(mask_dir, set(stems), persons)
    for pid in persons:
        log(f"   p{pid}: {len(masks[pid])} 帧 mask")

    votes = {pid: np.zeros(N, dtype=np.int16) for pid in persons}
    total = np.zeros(N, dtype=np.int16)

    # 头中心（深度门限用）
    head_info = {}
    for pid in persons:
        mesh_npz = mesh_dir / f"avatar_mesh_p{pid}.npz"
        bind_npz = mesh_dir / f"avatar_bind_p{pid}.npz"
        if not mesh_npz.exists() or not bind_npz.exists():
            log(f"  ⚠️ p{pid}: 缺 {mesh_npz.name} 或 {bind_npz.name}，跳过该人")
            continue
        head_info[pid] = head_box(mesh_npz, bind_npz, HEAD_BBOX_MARGIN)

    for si, stem in enumerate(stems):
        v = view_by_stem.get(stem)
        if v is None:
            continue
        R = qvec2rotmat(v["qvec"])
        T = np.asarray(v["tvec"])
        fx, fy = v["fx"], v["fy"]
        cx, cy, W, H = v["cx"], v["cy"], v["W"], v["H"]

        Xc = (R @ xyz.T).T + T
        z = Xc[:, 2]
        vis = z > NEAR
        u = np.where(vis, fx * Xc[:, 0] / np.where(vis, z, 1) + cx, -1)
        vv = np.where(vis, fy * Xc[:, 1] / np.where(vis, z, 1) + cy, -1)
        vis &= (u >= 0) & (u < W) & (vv >= 0) & (vv < H)

        ui = u.astype(np.int32).clip(0, W - 1)
        vi = vv.astype(np.int32).clip(0, H - 1)

        for pid in persons:
            if pid not in head_info or stem not in masks[pid]:
                continue
            mk = masks[pid][stem]
            if mk.shape != (H, W):
                # mask 与相机画幅不一致 → 缩放（01c 生成的 mask 应同画幅）
                from PIL import Image
                mk_im = Image.fromarray((mk * 255).astype(np.uint8))
                mk_im = mk_im.resize((W, H), Image.NEAREST)
                mk = np.asarray(mk_im) > 127
            inside = vis & mk[vi, ui]
            # 落入其他人 mask → 本帧不投票
            for pid2 in persons:
                if pid2 == pid or stem not in masks[pid2]:
                    continue
                m2 = masks[pid2][stem]
                if m2.shape != (H, W):
                    from PIL import Image
                    m2_im = Image.fromarray((m2 * 255).astype(np.uint8))
                    m2_im = m2_im.resize((W, H), Image.NEAREST)
                    m2 = np.asarray(m2_im) > 127
                inside &= ~m2[vi, ui]
            if not inside.any():
                continue
            # 深度门限: 该帧头中心深度
            hc = head_info[pid][2]
            z_head = float((R @ hc + T)[2])
            inside &= np.abs(z - z_head) < DEPTH_GATE * max(z_head, 1e-6)
            votes[pid][inside] += 1
        total[vis] += 1
        if (si + 1) % 10 == 0:
            log(f"   投票 {si + 1}/{len(stems)} 帧")

    log("🧮 归属判定 ...")
    best_p = np.full(N, -1, dtype=np.int64)
    best_v = np.zeros(N, dtype=np.int16)
    for pid in persons:
        v_ = votes[pid]
        better = v_ > best_v
        best_v[better] = v_[better]
        best_p[better] = int(pid)
    ok = (best_v >= MIN_VOTES) & (best_v / np.maximum(total, 1) >= VOTE_RATIO)
    person_of = np.where(ok, best_p, -1)

    # 头包围盒内丢弃
    in_head = np.zeros(N, dtype=bool)
    for pid in persons:
        if pid not in head_info:
            continue
        lo, hi, _, _ = head_info[pid]
        sel = person_of == int(pid)
        if not sel.any():
            continue
        in_head |= sel & np.all((xyz >= lo) & (xyz <= hi), axis=1)

    stats = {"total": int(N), "params": {
        "FRAME_STRIDE": FRAME_STRIDE, "MIN_VOTES": MIN_VOTES,
        "VOTE_RATIO": VOTE_RATIO, "DEPTH_GATE": DEPTH_GATE,
        "HEAD_BBOX_MARGIN": HEAD_BBOX_MARGIN}}
    for pid in persons:
        sel = person_of == int(pid)
        body = sel & ~in_head
        stats[f"p{pid}"] = {
            "votes_total": int(sel.sum()),
            "dropped_in_head_bbox": int((sel & in_head).sum()),
            "body": int(body.sum())}
        save_gaussian_ply(str(out_dir / f"body_gs_p{pid}.ply"),
                          names, data, np.where(body)[0])
        log(f"   p{pid}: mask 命中 {int(sel.sum()):,} → 头盒丢弃 "
            f"{int((sel & in_head).sum()):,} → body {int(body.sum()):,}")
    scene = person_of < 0
    stats["scene"] = int(scene.sum())
    save_gaussian_ply(str(out_dir / "scene_gs.ply"), names, data,
                      np.where(scene)[0])
    log(f"   scene: {int(scene.sum()):,}")

    with open(out_dir / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log(f"✅ 输出: {out_dir}")


if __name__ == "__main__":
    import sys
    main()
