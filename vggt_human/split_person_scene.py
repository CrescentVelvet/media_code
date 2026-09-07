#!/usr/bin/env python3
"""split_person_scene.py — 3DGS 点云按「人 / 场景」拆分 (目标架构: 头/身/景三模型)。

头模型已由 3DMM 拟合 + head_gs 构建 (03f/03g) 替代，因此本脚本只负责:
  body_gs_p{pid}.ply : 属于人的 SfM/GS 点 —— 多视角投票落入 person mask，
                       且不在 3DMM 头包围盒内 (头包围盒内的点被 head_gs 替代，丢弃)
  scene_gs.ply       : 其余点

点 → 人 归属规则 (多视角投票):
  对下采样的帧集合，把每个 GS 中心投影到相机:
    可视 (z>near 且在画幅内) 且落在 person p 的 mask 内 且
    深度接近该帧人头深度 (|z_pt - z_head| < DEPTH_GATE·z_head, 过滤人背后的背景点)
  → 给 p 投一票。票数 ≥ MIN_VOTES 且占该点总票数 ≥ VOTE_RATIO → 归 p。
  多人并列 → 归头中心最近者。

person track (video 模式 obj_id) 与 face pid 的对齐:
  用既有 face mask (清洗后) 与 person mask 的逐帧 IoM 累计, 一一映射 (贪心)。

Env vars: 见 03h_split_person_scene.sh
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_center_3d import parse_colmap_cameras
from fit_head_3dmm import qvec2rotmat

RESULTS_DIR = os.environ.get("RESULTS_DIR", "/mnt/d/output/vggt_human_ms")
SOURCE_DIR = os.environ.get("SOURCE_DIR", f"{RESULTS_DIR}/03b_source_ba")
GAUSSIAN_PLY = os.environ.get(
    "GAUSSIAN_PLY", f"{RESULTS_DIR}/04b_model_3dgs_ba/point_cloud/iteration_30000/point_cloud.ply")
PERSON_MASKS_DIR = os.environ.get("PERSON_MASKS_DIR", f"{RESULTS_DIR}/03_sam3_person_masks")
FACE_MASKS_DIR = os.environ.get("FACE_MASKS_DIR", f"{RESULTS_DIR}/03_sam3_face_masks")
HEAD_FIT_JSON = os.environ.get("HEAD_FIT_JSON", f"{RESULTS_DIR}/03e_head_3dmm/head_fit.json")
HEAD_MESH_DIR = os.environ.get("HEAD_MESH_DIR", f"{RESULTS_DIR}/03e_head_3dmm")
OUT_DIR = os.environ.get("SPLIT_OUT_DIR", f"{RESULTS_DIR}/03h_person_scene_split")

FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", "3"))       # 每 3 帧取 1 帧投票
MIN_VOTES = int(os.environ.get("MIN_VOTES", "4"))             # 最少票数
VOTE_RATIO = float(os.environ.get("VOTE_RATIO", "0.6"))       # 票数/总票数 下限
DEPTH_GATE = float(os.environ.get("DEPTH_GATE", "0.5"))       # |z_pt-z_head| < GATE·z_head
HEAD_BBOX_MARGIN = float(os.environ.get("HEAD_BBOX_MARGIN", "0.15"))  # 头包围盒外扩 15%
NEAR = float(os.environ.get("NEAR", "0.01"))
CHUNK = 200_000

PIDS = [int(s) for s in os.environ.get("PERSONS", "0,1,2").split(",") if s.strip() != ""]


def log(msg):
    print(msg, flush=True)


def load_gaussian_ply(path):
    from plyfile import PlyData
    ply = PlyData.read(path)
    v = ply["vertex"]
    names = [p.name for p in v.properties]
    data = {n: np.asarray(v[n]) for n in names}
    return names, data


def save_gaussian_ply(path, names, data, keep_idx):
    """按 keep_idx 抽取子集并写完整属性 PLY。"""
    from plyfile import PlyData, PlyElement
    n = len(keep_idx)
    arr = np.empty(n, dtype=[(nm, data[nm].dtype) for nm in names])
    for nm in names:
        arr[nm] = data[nm][keep_idx]
    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(path)


def load_person_masks(stems):
    """→ {obj_id: {stem: bool_mask}}。文件名 {stem}.p{oid}.mask.png。"""
    out = {}
    for f in os.listdir(PERSON_MASKS_DIR):
        if not f.endswith(".mask.png") or ".p" not in f:
            continue
        stem, rest = f.split(".p", 1)
        oid_s = rest.split(".")[0]
        if not oid_s.isdigit() or stem not in stems:
            continue
        m = np.asarray(Image.open(Path(PERSON_MASKS_DIR) / f).convert("L")) > 127
        out.setdefault(int(oid_s), {})[stem] = m
    return out


def load_face_masks(stems, pids):
    """→ {pid: {stem: bool_mask}}。"""
    out = {}
    for f in os.listdir(FACE_MASKS_DIR):
        if not f.endswith(".mask.png") or ".p" not in f:
            continue
        stem, rest = f.split(".p", 1)
        pid_s = rest.split(".")[0]
        if not pid_s.isdigit() or int(pid_s) not in pids or stem not in stems:
            continue
        m = np.asarray(Image.open(Path(FACE_MASKS_DIR) / f).convert("L")) > 127
        out.setdefault(int(pid_s), {})[stem] = m
    return out


def map_person_to_face(person_masks, face_masks):
    """IoM 累计贪心一对一映射: face_pid → person_obj_id。"""
    iom = {}
    for pid, pm in face_masks.items():
        for stem, fm in pm.items():
            for oid, om in person_masks.items():
                if stem not in om:
                    continue
                inter = np.logical_and(fm, om[stem]).sum()
                if inter == 0:
                    continue
                iom[(pid, oid)] = iom.get((pid, oid), 0) + inter / max(fm.sum(), 1)
    pairs = sorted(iom.items(), key=lambda kv: -kv[1])
    p2f, used = {}, set()
    for (pid, oid), s in pairs:
        if pid in p2f.values() or oid in used:
            continue
        p2f[oid] = pid
        used.add(oid)
    log(f"  🔗 person track → face pid 映射: "
        f"{ {f'obj{o}': f'p{p:02d}' for o, p in sorted(p2f.items())} }")
    return p2f  # person obj_id → face pid


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    views = parse_colmap_cameras(SOURCE_DIR)
    stems_all = [v["stem"] for v in views]
    stems = stems_all[::FRAME_STRIDE]
    view_by_stem = {v["stem"]: v for v in views}
    log(f"📷 相机 {len(views)} 帧, 投票用 {len(stems)} 帧 (stride={FRAME_STRIDE})")

    log("🧩 载入 3DGS 点云 ...")
    names, data = load_gaussian_ply(GAUSSIAN_PLY)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    N = len(xyz)
    log(f"   {N:,} 点, {len(names)} 属性")

    # 头包围盒 (3DMM mesh bbox 外扩) → 内部点由 head_gs 替代
    fit = json.load(open(HEAD_FIT_JSON))
    head_boxes = {}
    for pid in PIDS:
        zm = np.load(Path(HEAD_MESH_DIR) / f"head_mesh_p{pid:02d}.npz")
        V = zm["verts_world"]
        r = fit["persons"][f"{pid:02d}"]["head_radius_p90"]
        mg = r * HEAD_BBOX_MARGIN
        head_boxes[pid] = (V.min(0) - mg, V.max(0) + mg)
        log(f"   p{pid:02d} 头包围盒 margin {mg:.4f}")

    person_masks = load_person_masks(set(stems))
    log(f"🎭 person tracks: {sorted(person_masks)} "
        f"({sum(len(v) for v in person_masks.values())} 张 mask)")
    face_masks = load_face_masks(set(stems_all), set(PIDS))
    p2f = map_person_to_face(person_masks, face_masks)

    votes = {pid: np.zeros(N, dtype=np.int16) for pid in PIDS}
    total = np.zeros(N, dtype=np.int16)

    for si, stem in enumerate(stems):
        v = view_by_stem.get(stem)
        if v is None:
            continue
        R = qvec2rotmat(v["qvec"])
        T = np.asarray(v["tvec"])
        fxs, fys = v["fx"], v["fy"]
        cx, cy, W, H = v["cx"], v["cy"], v["W"], v["H"]

        Xc = (R @ xyz.T).T + T
        z = Xc[:, 2]
        vis = z > NEAR
        u = np.where(vis, fxs * Xc[:, 0] / np.where(vis, z, 1) + cx, -1)
        vv = np.where(vis, fys * Xc[:, 1] / np.where(vis, z, 1) + cy, -1)
        vis &= (u >= 0) & (u < W) & (vv >= 0) & (vv < H)

        for oid, om in person_masks.items():
            pid = p2f.get(oid)
            if pid is None or stem not in om:
                continue
            mk = om[stem]
            inside = vis & mk[vv.astype(np.int32).clip(0, H - 1),
                              u.astype(np.int32).clip(0, W - 1)]
            if not inside.any():
                continue
            # 深度门限: 该帧人头深度
            c = np.array(fit["persons"][f"{pid:02d}"]["head_center"])
            z_head = float((R @ c + T)[2])
            inside &= np.abs(z - z_head) < DEPTH_GATE * max(z_head, 1e-6)
            votes[pid][inside] += 1
        total[vis] += 1
        if (si + 1) % 10 == 0:
            log(f"   投票 {si + 1}/{len(stems)} 帧")

    # 归属判定
    log("🧮 归属判定 ...")
    stack = np.stack([votes[p] for p in PIDS], axis=0)          # (P, N)
    best_p = stack.argmax(axis=0)
    best_v = stack.max(axis=0)
    sum_v = stack.sum(axis=0)
    person_of = np.full(N, -1, dtype=np.int64)
    ok = (best_v >= MIN_VOTES) & (best_v / np.maximum(sum_v, 1) >= VOTE_RATIO)
    person_of[ok] = best_p[ok]

    # 头包围盒内的点丢弃 (head_gs 替代)
    in_head = np.zeros(N, dtype=bool)
    for pid, (lo, hi) in head_boxes.items():
        sel = person_of == pid
        if not sel.any():
            continue
        in_head |= sel & np.all((xyz >= lo) & (xyz <= hi), axis=1)

    stats = {"total": int(N)}
    for k, pid in enumerate(PIDS):
        sel = person_of == pid
        body = sel & ~in_head
        stats[f"p{pid:02d}"] = {
            "votes_total": int(sel.sum()),
            "dropped_in_head_bbox": int((sel & in_head).sum()),
            "body": int(body.sum()),
        }
        save_gaussian_ply(str(Path(OUT_DIR) / f"body_gs_p{pid:02d}.ply"),
                          names, data, np.where(body)[0])
        log(f"   p{pid:02d}: mask 命中 {int(sel.sum()):,} → 头盒丢弃 "
            f"{int((sel & in_head).sum()):,} → body {int(body.sum()):,}")
    scene = person_of < 0
    stats["scene"] = int(scene.sum())
    save_gaussian_ply(str(Path(OUT_DIR) / "scene_gs.ply"), names, data, np.where(scene)[0])
    log(f"   scene: {int(scene.sum()):,}")
    assert (in_head.sum() + scene.sum() +
            sum(stats[f"p{p:02d}"]["body"] for p in PIDS)) == N, "拆分不完整"

    with open(Path(OUT_DIR) / "split_stats.json", "w") as f:
        json.dump({"stats": stats, "person_to_face": {str(o): p for o, p in p2f.items()},
                   "params": {"FRAME_STRIDE": FRAME_STRIDE, "MIN_VOTES": MIN_VOTES,
                              "VOTE_RATIO": VOTE_RATIO, "DEPTH_GATE": DEPTH_GATE,
                              "HEAD_BBOX_MARGIN": HEAD_BBOX_MARGIN}}, f, indent=2)
    log(f"✅ 输出: {OUT_DIR}")


if __name__ == "__main__":
    main()
