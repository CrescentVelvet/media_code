#!/usr/bin/env python3
"""build_lm468_embedding.py — 生成 MediaPipe 468 点 → FLAME 顶点的重心嵌入。

阶段四要用 468 个 2D landmark 算重投影残差，而 FLAME 只自带 68 点嵌入
（68 点解不了 100 维 exp，见设计文档的可解性表）。所以必须自己建 468 点的对应。

做法（与 vggt_human/prepare_3dmm_template.py 建 lm468_idx/lm468_bary 同源）：
  1. 读 MediaPipe canonical_face_model.obj（468 个 3D 点，随 mediapipe 包自带）
  2. 读 FLAME mean face（shape=0, expr=0）+ 三角面
  3. 两者都是"标准正脸"，但单位与朝向约定不同 → 先归一化，再暴力搜旋转粗对齐，
     再用 ICP 精修
  4. 每个 468 点在 FLAME 表面找最近点 → 记 (face_id, bary)
  5. 语义自检：鼻尖要在两眼中点附近、在眼与嘴之间、且是最靠前的点

Env:
  FLAME_MODEL            FLAME2020 generic_model.pkl
  OUT_NPZ                输出（默认 $MODEL_DIR/flame_lm468_embedding.npz）
  CANONICAL_FACE_MODEL   MediaPipe canonical_face_model.obj（默认自动找）
  ICP_ITERS              ICP 迭代次数（默认 50）
"""
import os
import sys
from pathlib import Path

import numpy as np

N_LM = 468
# MediaPipe 468 布局里的语义点索引（用于自检）
IDX_NOSE_TIP = 1
IDX_EYE_L, IDX_EYE_R = 33, 263     # 内眼角
IDX_MOUTH_L, IDX_MOUTH_R = 61, 291  # 嘴角


def log(m):
    print(m, flush=True)


def find_canonical_face_model():
    """MediaPipe canonical_face_model.obj（468 顶点 + 三角面）。

    新版 mediapipe wheel 不再内置该文件（2026-09-09 踩坑），优先级：
    env CANONICAL_FACE_MODEL → mediapipe 包内 → $MODEL_DIR/mediapipe_aux/（01 脚本
    从官方仓镜像下载的副本）→ 报错。
    """
    env = os.environ.get("CANONICAL_FACE_MODEL", "")
    if env and Path(env).exists():
        return env
    cands = []
    try:
        import mediapipe as mp
        root = Path(mp.__file__).parent
        cands.append(root / "modules/face_geometry/data/canonical_face_model.obj")
        cands += sorted(root.rglob("canonical_face_model.obj"))
    except ImportError:
        pass
    model_dir = os.environ.get("MODEL_DIR", "")
    if model_dir:
        cands.append(Path(model_dir) / "mediapipe_aux" / "canonical_face_model.obj")
    for c in cands:
        if c.exists():
            return str(c)
    sys.exit("❌ 找不到 canonical_face_model.obj：新版 mediapipe wheel 不内置，"
             "下载到 $MODEL_DIR/mediapipe_aux/ 或用 CANONICAL_FACE_MODEL= 指定"
             "（来源 google-ai-edge/mediapipe 仓 "
             "mediapipe/modules/face_geometry/data/canonical_face_model.obj）")


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                # 只取每项的顶点索引，忽略 vt/vn
                faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def normalize(pts):
    """去质心 + 缩放到单位 RMS 半径。"""
    c = pts.mean(0)
    p = pts - c
    s = np.sqrt((p ** 2).sum(1).mean())
    return p / max(s, 1e-12), c, s


def rot_mat(yaw_deg, pitch_deg, roll_deg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("zyx", [yaw_deg, pitch_deg, roll_deg],
                               degrees=True).as_matrix()


def chamfer(a, b_tree, b, a_tree=None):
    """对称 Chamfer 距离（a 已变换过）。"""
    d_ab = b_tree.query(a)[0]
    if a_tree is None:
        from scipy.spatial import cKDTree
        a_tree = cKDTree(a)
    d_ba = a_tree.query(b)[0]
    return float(d_ab.mean() + d_ba.mean())


def icp_refine(src, dst_tree, dst, iters=50):
    """点到点 ICP：把 src 刚体变换到 dst。只微调，初始姿态已由粗搜给出。"""
    from scipy.spatial.transform import Rotation
    R = np.eye(3)
    t = np.zeros(3)
    cur = src.copy()
    for _ in range(iters):
        d, j = dst_tree.query(cur)
        if not np.isfinite(d).all():
            break
        tgt = dst[j]
        # 用有效配对解 Umeyama（不带尺度，粗搜已归一化过）
        cs, ct = cur.mean(0), tgt.mean(0)
        H = (cur - cs).T @ (tgt - ct)
        U, _, Vt = np.linalg.svd(H)
        dR = Vt.T @ U.T
        if np.linalg.det(dR) < 0:
            Vt[-1] *= -1
            dR = Vt.T @ U.T
        dt = ct - dR @ cs
        cur = (dR @ cur.T).T + dt
        R, t = dR @ R, dR @ t + dt
        if np.abs(dR - np.eye(3)).max() < 1e-7 and np.abs(dt).max() < 1e-9:
            break
    return R, t, cur


def nearest_on_surface(queries, verts, faces, vert_faces):
    """每个查询点 → (face_id, bary)。

    最近点一定落在「最近顶点的邻接三角形」上（或至少很接近），
    所以先用 KD-tree 找最近顶点，再只查它的 1-ring 三角形，避免全网格暴力。
    """
    from scipy.spatial import cKDTree
    v_tree = cKDTree(verts)
    v0 = verts[faces[:, 0]]
    e1 = verts[faces[:, 1]] - v0
    e2 = verts[faces[:, 2]] - v0

    out_fid = np.zeros(len(queries), dtype=np.int64)
    out_bary = np.zeros((len(queries), 3), dtype=np.float64)
    out_pts = np.zeros((len(queries), 3), dtype=np.float64)
    for i, q in enumerate(queries):
        best_d, best_f, best_b, best_p = np.inf, -1, None, None
        # 取最近若干个顶点，扩大 1-ring 覆盖，防止落在 1-ring 缝隙外
        _, nn = v_tree.query(q, k=8)
        cand = set()
        for vi in np.atleast_1d(nn):
            cand.update(vert_faces[vi].tolist())
        for fi in cand:
            a, b = e1[fi], e2[fi]
            n = np.cross(a, b)
            nn2 = n @ n
            if nn2 < 1e-18:
                continue
            # 投影到三角形平面
            w = q - v0[fi]
            u = (w @ a) * (b @ b) - (w @ b) * (a @ b)
            v = (w @ b) * (a @ a) - (w @ a) * (a @ b)
            denom = (a @ a) * (b @ b) - (a @ b) ** 2
            if abs(denom) < 1e-18:
                continue
            u, v = u / denom, v / denom
            #  clamp 到三角形内（重心坐标 ∈ [0,1] 且 u+v ≤ 1）
            if u < 0 or v < 0 or u + v > 1:
                cu, cv = np.clip(u, 0, 1), np.clip(v, 0, 1)
                if cu + cv > 1:
                    s = 0.5 if (cu + cv) == 0 else 1.0 / (cu + cv)
                    cu, cv = cu * s, cv * s
                u, v = cu, cv
            p = v0[fi] + u * a + v * b
            d = float(((p - q) ** 2).sum())
            if d < best_d:
                best_d, best_f = d, fi
                best_b = np.array([1.0 - u - v, u, v])
                best_p = p
        out_fid[i], out_bary[i], out_pts[i] = best_f, best_b, best_p
    return out_fid, out_bary, out_pts


def main():
    flame_model = os.environ.get("FLAME_MODEL", "")
    model_dir = os.environ.get("MODEL_DIR", "")
    out_npz = Path(os.environ.get(
        "OUT_NPZ", f"{model_dir}/flame_lm468_embedding.npz"))
    icp_iters = int(os.environ.get("ICP_ITERS", "50"))

    if not flame_model or not Path(flame_model).exists():
        sys.exit(f"❌ FLAME_MODEL 不存在: {flame_model}")

    log("🔗 构建 MediaPipe 468 → FLAME 重心嵌入")

    # ── FLAME mean face ──────────────────────────────────────────────────
    import smplx
    # smplx 约定：model_path 传目录（拼 <dir>/flame/FLAME_NEUTRAL.pkl）
    flame = smplx.create(model_path=str(Path(flame_model).parent),
                         model_type="flame",
                         num_expression_coeffs=100, use_face_contour=False)
    verts = flame().vertices.detach().cpu().numpy().squeeze().astype(np.float64)
    faces = np.asarray(flame.faces, dtype=np.int64)
    log(f"  🗿 FLAME: {verts.shape[0]} 顶点, {faces.shape[0]} 面")

    # vertex → incident faces 邻接表
    vert_faces = [[] for _ in range(len(verts))]
    for fi, tri in enumerate(faces):
        for vi in tri:
            vert_faces[vi].append(fi)
    vert_faces = [np.asarray(x, dtype=np.int64) for x in vert_faces]

    # ── MediaPipe canonical 468 ──────────────────────────────────────────
    cpath = find_canonical_face_model()
    mp_v, _ = load_obj(cpath)
    if len(mp_v) != N_LM:
        sys.exit(f"❌ canonical_face_model 顶点数 {len(mp_v)} != {N_LM}")
    log(f"  🎯 MediaPipe canonical: {cpath}")

    # ── 归一化 ───────────────────────────────────────────────────────────
    V_n, V_c, V_s = normalize(verts)
    M_n, _, _ = normalize(mp_v)

    # ── 粗搜旋转（yaw 全周 + pitch/roll 微调）──────────────────────────
    from scipy.spatial import cKDTree
    V_tree = cKDTree(V_n)
    log("  🔍 粗搜朝向 ...")
    best = (np.inf, np.eye(3))
    for yaw in range(0, 360, 10):
        for pitch in (-20, 0, 20):
            for roll in (-15, 0, 15):
                R = rot_mat(yaw, pitch, roll)
                d = chamfer(M_n @ R.T, V_tree, V_n)
                if d < best[0]:
                    best = (d, R)
    log(f"     粗搜 Chamfer = {best[0]:.5f}")
    M_c = M_n @ best[1].T

    # ── ICP 精修 ─────────────────────────────────────────────────────────
    _, _, M_icp = icp_refine(M_c, V_tree, V_n, iters=icp_iters)
    log(f"  📐 ICP 后 Chamfer = {chamfer(M_icp, V_tree, V_n):.5f}")

    # 回到 FLAME 原始尺度/位置（V_n = (V - V_c)/V_s）
    M_flame = M_icp * V_s + V_c

    # ── 最近点 → (face_id, bary) ─────────────────────────────────────────
    fid, bary, proj = nearest_on_surface(M_flame, verts, faces, vert_faces)
    resid = np.linalg.norm(proj - M_flame, axis=1)
    log(f"  📌 投影残差: 中值 {np.median(resid)*1000:.2f}mm, "
        f"p95 {np.percentile(resid,95)*1000:.2f}mm")

    # ── 语义自检 ─────────────────────────────────────────────────────────
    # 鼻尖：x 接近两眼中点、y 在眼与嘴之间、z 是最靠前的（FLAME +Z 朝前）
    eye_mid = proj[[IDX_EYE_L, IDX_EYE_R]].mean(0)
    mouth_mid = proj[[IDX_MOUTH_L, IDX_MOUTH_R]].mean(0)
    nose = proj[IDX_NOSE_TIP]
    ok_x = abs(nose[0] - eye_mid[0]) < 0.02
    ok_y = (mouth_mid[1] < nose[1] < eye_mid[1]) or (eye_mid[1] < nose[1] < mouth_mid[1])
    ok_z = nose[2] >= np.percentile(proj[:, 2], 95)
    log(f"  🩺 自检: 鼻尖 x 偏差 {abs(nose[0]-eye_mid[0])*1000:.1f}mm "
        f"{'✅' if ok_x else '❌'} | y 在眼嘴之间 {'✅' if ok_y else '❌'} | "
        f"z 最靠前 {'✅' if ok_z else '❌'}")
    if not (ok_x and ok_y and ok_z):
        log("  ⚠️ 自检未全过：对齐可能失败，检查 FLAME 朝向约定（+Z 是否朝前）"
            "或 ICP_ITERS 是否够")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_npz),
        lm_face_id=fid,        # (468,) 每个 landmark 落在哪个三角面
        lm_bary=bary,          # (468,3) 重心坐标
        lm_verts=faces[fid],   # (468,3) 该三角面的三个顶点索引（便于直接取点）
        verts_mean=verts,      # (5023,3) 生成时用的 FLAME mean（shape=0, expr=0）
        faces=faces,           # (F,3)
        proj_resid=resid,      # (468,) 投影残差，大值说明该点对应不可靠
        meta=np.array([f"flame={flame_model}", f"canonical={cpath}"]),
    )
    log(f"💾 {out_npz}")
    log("🎉 嵌入生成完成")


if __name__ == "__main__":
    main()
