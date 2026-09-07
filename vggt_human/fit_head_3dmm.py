#!/usr/bin/env python3
"""fit_head_3dmm.py — 多视角 3DMM 头拟合（头/身/景拆分 step 3）。

输入：3DMM 模板 npz（03d 产出）+ 每帧 468 landmark（03e 产出）+ COLMAP 相机
输出：每人一个「世界坐标系下的完整人头网格」（含后脑），供 build_head_gs 用。

优化变量：
  - 全局尺度 s      （把 cm 级模板缩放到 SfM 世界尺度；VGGT 输出尺度是任意的，必须拟合）
  - 全局身份系数 β  （identity PCA，个性化头型；无 PCA 时维度为 0）
  - 每帧刚体 (R,t)  （人会在视频里转头，所以每帧独立姿态）

观测：468 个 landmark 的 2D 重投影（只在 MediaPipe 检出的帧上）

初始化流程（直接联合优化容易跑飞，先给个靠谱初值）：
  1. 用所有检出帧的投影矩阵做线性三角化 → landmark 的世界坐标（世界尺度正确）
  2. 与模板 landmark 做 Umeyama（带尺度）→ s0, R0, t0
  3. 每帧姿态初值 = (R0, t0)
  4. scipy.optimize.least_squares 联合 refine（带 jac_sparsity + soft_l1 鲁棒损失）

参考姿态（head_gs 用）：所有帧姿态的「位置中位 + 旋转四元数平均」。
3DGS 是静态表示，人头在视频里微动，取中位比取某一帧稳健。

Env:
  TEMPLATE_NPZ  3dmm 模板（默认 $MODEL_3DMM_DIR/3dmm_template.npz）
  LANDMARKS     landmark json（默认 $RESULTS_DIR/03e_head_3dmm/face_landmarks.json）
  SOURCE_DIR    COLMAP 场景（默认 $RESULTS_DIR/03_source）
  OUT_DIR       输出目录（默认 $RESULTS_DIR/03e_head_3dmm）
  MIN_FRAMES    少于该帧数的人跳过（默认 8）
  N_OUTER       交替/联合优化轮数（默认 3）
  FIT_BETA      是否拟合身份系数（默认 1）
"""
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_center_3d import parse_colmap_cameras  # noqa: E402
from prepare_3dmm_template import umeyama  # noqa: E402

N_LM = 468


def log(msg):
    print(msg, flush=True)


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def build_proj_matrices(views):
    """返回 {stem: (P 3x4 世界→像素, K 3x3)}"""
    out = {}
    for v in views:
        K = np.array([[v["fx"], 0, v["cx"]], [0, v["fy"], v["cy"]], [0, 0, 1.0]])
        R = qvec2rotmat(v["qvec"])
        t = np.asarray(v["tvec"]).reshape(3, 1)
        P = K @ np.hstack([R, t])
        out[v["stem"]] = (P, K)
    return out


def triangulate_all(Ps, obs):
    """线性三角化（DLT）：obs (F,468,2), Ps (F,3,4) → X (468,3) 世界坐标。"""
    F = len(Ps)
    X = np.zeros((N_LM, 3))
    for j in range(N_LM):
        A = []
        for i in range(F):
            u, v = obs[i, j]
            P = Ps[i]
            A.append(u * P[2] - P[0])
            A.append(v * P[2] - P[1])
        A = np.asarray(A)
        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[-1]
        X[j] = Xh[:3] / Xh[3]
    return X


def template_landmarks(tmpl, beta):
    """β → 模板 landmark 3D 位置 (468,3)，模板空间。"""
    base = tmpl["id_mean"]
    basis = tmpl["id_basis"]
    lm_idx = tmpl["lm468_idx"]
    bary = tmpl["lm468_bary"]
    if beta is not None and len(beta) > 0 and basis.shape[0] == len(beta):
        V = base + np.einsum("b,bij->ij", beta, basis)
    else:
        V = base
    return (V[lm_idx] * bary[:, :, None]).sum(1), V


def make_residual(lm_tmpl_base, basis, lm_idx, bary, obs, Ps, n_beta, use_beta):
    """残差：x = [log_s, beta(B)?, 每帧 (rvec3, tvec3)]"""
    F = len(Ps)

    def resid(x):
        s = float(np.exp(x[0]))
        if use_beta:
            beta = x[1 : 1 + n_beta]
            V = lm_tmpl_base + np.einsum("b,bij->ij", beta, basis)
        else:
            V = lm_tmpl_base
        lm = (V[lm_idx] * bary[:, :, None]).sum(1)  # (468,3)

        out = np.empty(F * N_LM * 2)
        off = 0
        for i in range(F):
            b = 1 + n_beta + 6 * i
            rv = x[b : b + 3]
            tv = x[b + 3 : b + 6]
            R, _ = cv2.Rodrigues(rv.astype(np.float64))
            Xw = (s * (R @ lm.T)).T + tv
            Xh = np.hstack([Xw, np.ones((N_LM, 1))])
            pr = (Ps[i] @ Xh.T).T
            uv = pr[:, :2] / np.clip(pr[:, 2:3], 1e-9, None)
            d = (uv - obs[i]).ravel()
            out[off : off + d.size] = d
            off += d.size
        return out

    return resid


def build_sparsity(F, n_beta):
    n_res = F * N_LM * 2
    n_x = 1 + n_beta + 6 * F
    S = lil_matrix((n_res, n_x), dtype=int)
    off = 0
    for i in range(F):
        rows = np.arange(off, off + N_LM * 2)
        # 依赖全局（log_s + beta）与该帧 6 个变量
        cols = [0] + list(range(1, 1 + n_beta)) + list(range(1 + n_beta + 6 * i, 1 + n_beta + 6 * i + 6))
        for c in cols:
            S[rows, c] = 1
        off += N_LM * 2
    return S.tocsr()


def fit_person(pid, frames, projs, tmpl, min_frames, use_beta, verbose=True):
    """frames: list of (stem, obs (468,2))"""
    if len(frames) < min_frames:
        return None, f"帧数不足 ({len(frames)} < {min_frames})"

    stems = [f[0] for f in frames]
    Ps = np.stack([projs[s][0] for s in stems])
    obs = np.stack([f[1] for f in frames])
    F = len(frames)

    # ── 初始化 1：三角化 → 世界坐标 landmark ─────────────────────────────
    Xw = triangulate_all(Ps, obs)
    lm_tmpl0, _ = template_landmarks(tmpl, None)
    s0, R0, t0 = umeyama(lm_tmpl0, Xw, with_scale=True)
    if not np.isfinite(s0) or s0 <= 0:
        return None, "三角化初值异常"

    # ── 初始化 2：每帧姿态 = 全局参考姿态 ────────────────────────────────
    r0 = Rotation.from_matrix(R0).as_rotvec()
    x0 = [np.log(s0)]
    basis = tmpl["id_basis"]
    n_beta = int(basis.shape[0]) if use_beta else 0
    if use_beta:
        x0.extend(np.zeros(n_beta))
    for _ in range(F):
        x0.extend(r0)
        x0.extend(t0)
    x0 = np.asarray(x0)

    resid = make_residual(
        tmpl["id_mean"], basis, tmpl["lm468_idx"], tmpl["lm468_bary"],
        obs, Ps, n_beta, use_beta,
    )
    r_init = resid(x0)
    err_init = float(np.sqrt(np.mean(r_init ** 2)))
    if verbose:
        log(f"     初值 RMS 重投影误差: {err_init:.2f} px (s0={s0:.4g})")

    S = build_sparsity(F, n_beta)
    t0t = time.time()
    res = least_squares(
        resid, x0, jac_sparsity=S, loss="soft_l1", f_scale=5.0,
        x_scale="jac", max_nfev=200, verbose=0,
    )
    err = float(np.sqrt(np.mean(res.fun ** 2)))
    if verbose:
        log(f"     优化后 RMS: {err:.2f} px  ({time.time()-t0t:.0f}s, {res.nfev} nfev)")

    # ── 解析结果 ─────────────────────────────────────────────────────────
    s = float(np.exp(res.x[0]))
    beta = res.x[1 : 1 + n_beta].tolist() if use_beta else []
    per_frame = {}
    Rs, ts = [], []
    for i, stem in enumerate(stems):
        b = 1 + n_beta + 6 * i
        rv = res.x[b : b + 3]
        tv = res.x[b + 3 : b + 6]
        R, _ = cv2.Rodrigues(np.asarray(rv, dtype=np.float64))
        per_frame[stem] = {"R": R.tolist(), "t": tv.tolist()}
        Rs.append(R)
        ts.append(tv)
    Rs = np.stack(Rs)
    ts = np.stack(ts)

    # 参考姿态：位置中位 + 旋转四元数平均
    t_ref = np.median(ts, axis=0)
    q_mean = Rotation.from_matrix(Rs).mean()
    R_ref = q_mean.as_matrix()

    # 世界坐标下的完整头网格（含后脑）
    _, V_skin = template_landmarks(tmpl, np.asarray(beta) if use_beta else None)
    mesh_world = (s * (R_ref @ V_skin.T)).T + t_ref

    # 脸中心 = 468 landmark 质心（与既有 face_center_3d 的语义一致：
    # closeup 相机是朝「脸」推近，不是朝头的几何中心）
    face_center = s * (R_ref @ lm_tmpl0.mean(0)) + t_ref
    # 头几何中心 = face 段顶点质心（含后脑的 skin 会带进脖子，偏低）
    geom_center = s * (R_ref @ V_skin[:9409].mean(0)) + t_ref
    radius = float(
        np.percentile(np.linalg.norm(mesh_world[:9409] - geom_center, axis=1), 90)
    )

    return {
        "pid": pid,
        "n_frames": F,
        "scale": s,
        "beta": beta,
        "R_ref": R_ref.tolist(),
        "t_ref": t_ref.tolist(),
        "per_frame": per_frame,
        "reproj_rms_px": err,
        "reproj_rms_px_init": err_init,
        "head_center": face_center.tolist(),      # 脸中心（landmark 质心）
        "head_geom_center": geom_center.tolist(),  # 头几何中心（face 段顶点质心）
        "head_radius_p90": radius,               # 90 分位半径（世界单位）
        "frames_used": stems,
    }, None


def main():
    model_dir = os.environ.get("MODEL_3DMM_DIR", "")
    results_dir = os.environ.get("RESULTS_DIR", "")
    template_npz = Path(
        os.environ.get("TEMPLATE_NPZ", f"{model_dir}/3dmm_template.npz")
    )
    landmarks_json = Path(
        os.environ.get("LANDMARKS", f"{results_dir}/03e_head_3dmm/face_landmarks.json")
    )
    source_dir = Path(os.environ.get("SOURCE_DIR", f"{results_dir}/03_source"))
    out_dir = Path(os.environ.get("OUT_DIR", f"{results_dir}/03e_head_3dmm"))
    min_frames = int(os.environ.get("MIN_FRAMES", "8"))
    use_beta = int(os.environ.get("FIT_BETA", "1"))

    for p in (template_npz, landmarks_json):
        if not p.exists():
            log(f"❌ 缺少输入: {p}")
            return 1

    log("🗿 多视角 3DMM 头拟合")
    tmpl = np.load(str(template_npz), allow_pickle=True)
    log(f"  📦 模板: {template_npz.name}  basis={tmpl['id_basis'].shape}")

    lm = json.loads(landmarks_json.read_text())
    views = parse_colmap_cameras(str(source_dir))
    projs = build_proj_matrices(views)
    log(f"  📷 相机: {len(projs)} views")

    # 汇总每 pid 的观测
    person_obs = {}
    for stem, rec in lm["frames"].items():
        if stem not in projs:
            continue
        for pid, pr in rec.get("persons", {}).items():
            if not pr.get("ok") or not pr.get("lm"):
                continue
            arr = np.asarray(pr["lm"], dtype=np.float64)
            if arr.shape != (N_LM, 2):
                continue
            person_obs.setdefault(pid, []).append((stem, arr))
    log(f"  👥 有观测的人: { {k: len(v) for k, v in sorted(person_obs.items())} }")

    out = {"meta": {
        "template": str(template_npz),
        "landmarks": str(landmarks_json),
        "source_dir": str(source_dir),
        "min_frames": min_frames,
        "fit_beta": bool(use_beta),
    }, "persons": {}}

    out_dir.mkdir(parents=True, exist_ok=True)
    for pid in sorted(person_obs):
        log(f"  🔧 拟合 pid={pid} ({len(person_obs[pid])} 帧)…")
        r, err = fit_person(pid, person_obs[pid], projs, tmpl, min_frames, bool(use_beta))
        if r is None:
            log(f"     ⚠️ 跳过: {err}")
            continue
        out["persons"][pid] = r
        # 世界坐标头网格单独存 npz（不进 json）
        _, V_skin = template_landmarks(
            tmpl, np.asarray(r["beta"]) if r["beta"] else None
        )
        mesh_world = (
            r["scale"] * (np.asarray(r["R_ref"]) @ V_skin.T)
        ).T + np.asarray(r["t_ref"])
        np.savez_compressed(
            str(out_dir / f"head_mesh_p{pid}.npz"),
            verts_world=mesh_world,
            faces=tmpl["skin_faces"],
        )
        log(
            f"     ✅ s={r['scale']:.4g} rms={r['reproj_rms_px']:.2f}px "
            f"center={np.round(r['head_center'],3).tolist()} R_p90={r['head_radius_p90']:.3f}"
        )

    (out_dir / "head_fit.json").write_text(json.dumps(out, indent=1))
    log(f"💾 {out_dir/'head_fit.json'}")
    log(f"✅ 拟合完成，成功 {len(out['persons'])} 人")
    return 0


if __name__ == "__main__":
    sys.exit(main())
