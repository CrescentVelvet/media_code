#!/usr/bin/env python3
"""init_avatar_gs.py — 阶段五：AvatarGaussian 初始化（重心绑定）。

产出三样东西：
  1. avatar_p{pid}.ply   标准 3DGS 初始 PLY（x,y,z + 法线 + f_dc + opacity/scale/rot）
  2. avatar_bind_p{pid}.npz   (face_id, bary, is_free) —— **绑定的核心**
  3. avatar_mesh_p{pid}.npz   世界坐标下的中性脸网格（阶段七切分要用）

为什么必须绑：exp_base 形状是 (100, 5023, 3)，只能驱动 5023 个顶点位置。
3DGS 训练会 densify，点数长到几万，新点没有 exp_base 索引 → 形变时头部撕裂。
绑定的做法：每个高斯记住自己在哪个三角面 + 重心坐标，
    p_f = bary @ V_f[face_id]，V_f = mean + B_id@id + B_exp@exp_f
densify 时子点继承父点的 (face_id, bary)（clone 直接继承，split 加微小扰动）——
这部分在训练脚本里实现，本脚本只负责初始那批。

自由高斯（头发/睫毛）：mesh 表达不了的结构，不绑定（face_id = -1），
只跟随刚体变换不叠 exp（参照参考实现里 glass 的 vert_ids 虚拟顶点机制）。
采样位置：头皮区顶点沿法线外推，而不是整个头部——脸上放自由高斯会和绑定点打架。

Env: ALIGN_JSON / OUT_DIR / FLAME_MODEL / N_BOUND / N_FREE
     FREE_OFFSET_MM / SCALP_Y_PCT / INIT_OPACITY
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

N_SHAPE = 300
N_EXPR = 100


def log(m):
    print(m, flush=True)


def quat_to_mat_np(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def median_quat_np(qs):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_quat(np.c_[qs[:, 1:4], qs[:, 0]]).mean().as_quat()
    return np.concatenate([q[3:4], q[:3]])


def compute_vertex_normals(verts, faces):
    n = np.zeros_like(verts)
    v0 = verts[faces[:, 0]]
    e1 = verts[faces[:, 1]] - v0
    e2 = verts[faces[:, 2]] - v0
    fn = np.cross(e1, e2)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


def area_weighted_sample(verts, faces, n_points, rng):
    """面积加权的三角面内均匀采样 → (pts, normals, face_id, bary, spacing)。"""
    v0 = verts[faces[:, 0]]
    e1 = verts[faces[:, 1]] - v0
    e2 = verts[faces[:, 2]] - v0
    cross = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    prob = area / area.sum()
    fi = rng.choice(len(faces), size=n_points, p=prob)

    u = rng.random(n_points)
    v = rng.random(n_points)
    over = (u + v) > 1.0
    u[over] = 1.0 - u[over]
    v[over] = 1.0 - v[over]
    w = 1.0 - u - v
    bary = np.stack([w, u, v], axis=1)

    tri = faces[fi]
    pts = (verts[tri] * bary[:, :, None]).sum(1)

    vn = compute_vertex_normals(verts, faces)
    nrm = (vn[tri] * bary[:, :, None]).sum(1)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(ln, 1e-12)

    # 相邻点间距估计：给初始 scale 用（3DGS 用最近邻距离的 log）
    spacing = np.sqrt(area[fi] * 2.0)
    return pts, nrm, fi, bary, spacing


def write_3dgs_ply(path, xyz, nrm, rgb=None, opacity=0.1, scale=None):
    """写标准 3DGS 初始 PLY（属性齐全，官方 create_from_ply 能直接吃）。"""
    from plyfile import PlyData, PlyElement
    n = len(xyz)
    if rgb is None:
        rgb = np.full((n, 3), 0.5, dtype=np.float32)
    if scale is None:
        scale = np.full((n, 3), np.log(0.005), dtype=np.float32)
    else:
        scale = np.log(np.maximum(scale, 1e-6)).astype(np.float32)
    rot = np.zeros((n, 4), dtype=np.float32)
    rot[:, 0] = 1.0
    op = np.full((n, 1), opacity, dtype=np.float32)

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    for i in range(3):
        dtype.append((f"f_dc_{i}", "f4"))
    dtype += [("opacity", "f4")]
    for i in range(3):
        dtype.append((f"scale_{i}", "f4"))
    for i in range(4):
        dtype.append((f"rot_{i}", "f4"))

    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["nx"], arr["ny"], arr["nz"] = nrm[:, 0], nrm[:, 1], nrm[:, 2]
    for i in range(3):
        arr[f"f_dc_{i}"] = rgb[:, i]
    arr["opacity"] = op[:, 0]
    for i in range(3):
        arr[f"scale_{i}"] = scale[:, i]
    for i in range(4):
        arr[f"rot_{i}"] = rot[:, i]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def main():
    results_dir = os.environ.get("RESULTS_DIR", "")
    align_json = Path(os.environ.get(
        "ALIGN_JSON", f"{results_dir}/05_align/head_align.json"))
    out_dir = Path(os.environ.get("OUT_DIR", f"{results_dir}/06_avatar_gs"))
    flame_model = os.environ.get("FLAME_MODEL", "")
    n_bound = int(os.environ.get("N_BOUND", "50000"))
    n_free = int(os.environ.get("N_FREE", "20000"))
    free_off_mm = float(os.environ.get("FREE_OFFSET_MM", "8"))
    scalp_pct = float(os.environ.get("SCALP_Y_PCT", "60"))
    init_op = float(os.environ.get("INIT_OPACITY", "0.1"))
    # 初始 scale 缩放（提锐度用）：<1 让高斯更细，细节需靠更多点/densify 补。
    # 2026-09-10：head 残留柔化的一个来源是 scale 被"面片间距"锚住偏大。
    scale_factor = float(os.environ.get("SCALE_FACTOR", "1.0"))
    seed = int(os.environ.get("SEED", "0"))

    if not align_json.exists():
        sys.exit(f"❌ 缺少阶段四输出: {align_json}")
    if not flame_model or not Path(flame_model).exists():
        sys.exit(f"❌ 缺少 FLAME: {flame_model}")

    log("🎨 [阶段五] AvatarGaussian 初始化（重心绑定）")
    import smplx
    dev = torch.device("cpu")
    # smplx 约定：model_path 传目录（拼 <dir>/flame/FLAME_NEUTRAL.pkl）
    flame = smplx.create(model_path=str(Path(flame_model).parent),
                         model_type="flame",
                         num_betas=N_SHAPE, num_expression_coeffs=N_EXPR,
                         use_face_contour=False).to(dev)
    for p in flame.parameters():
        p.requires_grad_(False)
    faces = np.asarray(flame.faces, dtype=np.int64)

    align = json.loads(align_json.read_text())
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    for oid in sorted(align.get("persons", {})):
        P = align["persons"][oid]
        idc = torch.tensor(P["id_coeff"], dtype=torch.float32).reshape(1, N_SHAPE)
        with torch.no_grad():
            # 中性脸：exp = 0（不是"平均表情"；表情基底是相对中性的偏移）
            out = flame(betas=idc,
                        expression=torch.zeros(1, N_EXPR, dtype=torch.float32))
            verts_can = out.vertices.squeeze(0).numpy().astype(np.float64)

        # 参考姿态：局部旋转取四元数平均、平移取逐维中位（与 vggt_human 一致）
        lq = np.asarray(P["local_q"], dtype=np.float64)
        lt = np.asarray(P["local_t"], dtype=np.float64)
        q_ref = median_quat_np(lq)
        t_ref = np.median(lt, axis=0)
        R_ref = quat_to_mat_np(q_ref)

        R_g = quat_to_mat_np(np.asarray(P["global_q"], dtype=np.float64))
        t_g = np.asarray(P["global_t"], dtype=np.float64)
        s = float(P["scale"])

        p_local = (R_ref @ verts_can.T).T + t_ref
        verts_w = s * ((R_g @ p_local.T).T + t_g)

        # 绑定点：整个头部表面
        pts, nrm, fid, bary, spacing = area_weighted_sample(
            verts_w, faces, n_bound, rng)

        # 自由点：头皮区（y 分位数以上）沿法线外推，不绑定
        y_thr = np.percentile(verts_w[:, 1], scalp_pct)
        scalp_mask = faces[
            (verts_w[faces][:, :, 1] > y_thr).any(axis=1)]
        if len(scalp_mask) == 0:
            scalp_mask = faces
        fpts, fnrm, _, _, fsp = area_weighted_sample(
            verts_w, scalp_mask, n_free, rng)
        off = rng.uniform(0.0, free_off_mm / 1000.0, size=(len(fpts), 1))
        # off 是米（FLAME 尺度），乘 s 转成 SfM 世界单位
        fpts = fpts + fnrm * off * s

        all_pts = np.vstack([pts, fpts])
        all_nrm = np.vstack([nrm, fnrm])
        all_fid = np.concatenate([fid, np.full(len(fpts), -1, dtype=np.int64)])
        all_bary = np.vstack([bary, np.zeros((len(fpts), 3), dtype=np.float64)])
        is_free = np.concatenate([np.zeros(len(pts), bool),
                                  np.ones(len(fpts), bool)])
        sc = np.concatenate([spacing, fsp]) * scale_factor

        ply_path = out_dir / f"avatar_p{oid}.ply"
        npz_path = out_dir / f"avatar_bind_p{oid}.npz"
        write_3dgs_ply(ply_path, all_pts.astype(np.float32),
                       all_nrm.astype(np.float32), opacity=init_op,
                       scale=sc[:, None].repeat(3, 1).astype(np.float32))
        np.savez_compressed(
            str(npz_path), face_id=all_fid, bary=all_bary, is_free=is_free,
            ref_local_q=q_ref, ref_local_t=t_ref,
            global_q=np.asarray(P["global_q"]), global_t=t_g, scale=s,
            id_coeff=np.asarray(P["id_coeff"]))
        np.savez_compressed(
            str(out_dir / f"avatar_mesh_p{oid}.npz"),
            verts_world=verts_w, faces=faces)

        log(f"  ✅ p{oid}: {len(pts)} 绑定 + {len(fpts)} 自由 "
            f"→ {ply_path.name}")
        log(f"     scale={s:.4g}  中性脸顶点 {verts_w.shape[0]}  "
            f"ref_local_q={np.round(q_ref,3).tolist()}")

    log(f"💾 输出目录: {out_dir}")
    log("🎉 阶段五完成。下一步：bash 07_init_body_gs.sh")


if __name__ == "__main__":
    main()
