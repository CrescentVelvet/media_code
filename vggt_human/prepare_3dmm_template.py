#!/usr/bin/env python3
"""prepare_3dmm_template.py — 从 ICT-FaceKit 构建 3DMM 头模板（供多视角拟合用）。

背景（vggt_human 头/身/景拆分）:
  3DGS 初始化要拆成三个独立模型：人头 / 人体 / 场景。人头不用 SfM 点（稀疏、
  且有后脑缺失），而是用 3DMM 模板拟合到多视角观测上，这样 head_gs 是完整的
  人头（含后脑），closeup 推近时 alpha 覆盖率才有物理意义。

本脚本只做「离线模板准备」，不碰具体数据：
  1. 从 ICT-FaceKit generic_neutral_mesh.obj 提取 skin 网格
     （拓扑段 #0 Face + #1 Head and Neck = 顶点 [0:11248]、面 [0:11144]，
     含后脑与脖子，排除眼球/牙齿/舌头/睫毛等内部件）
  2. 用 identity000-023（24 个身份，同拓扑）建 identity PCA 形状基
  3. 建立 MediaPipe 468 landmark ↔ ICT skin 三角面的 barycentric 对应
     —— 质心/鼻尖粗对齐 + 鲁棒点面 ICP（yaw 0/180 双假设，取残差小者）
     这样每帧的 468 个 2D landmark 可以直接约束 3DMM 顶点

产出 npz:
  skin_verts   (M,3)    通用头 skin 顶点（ICT 模板空间，单位 cm）
  skin_faces   (K,3)    三角面
  id_mean      (M,3)    identity 均值（无 identity 时 = skin_verts）
  id_basis     (B,M,3)  identity PCA 基（按 std 降序）
  id_std       (B,)     对应标准差
  lm468_idx    (468,3)  landmark → 三角面顶点索引
  lm468_bary   (468,3)  landmark → 重心坐标
  lm468_verts  (468,3)  landmark 在 ICT 模板空间的位置
  T_mp2ict     (4,4)    MediaPipe canonical → ICT 模板空间 的相似变换

Env:
  MODEL_3DMM_DIR  3DMM 模型根（默认 $WEIGHTS_ROOT/ICT-FaceKit，WSL: ~/model/ICT-FaceKit）
  OUT_NPZ         输出 npz（默认 $MODEL_3DMM_DIR/3dmm_template.npz）
  DEBUG_PNG       对齐验证图（默认 $MODEL_3DMM_DIR/3dmm_align_check.png）
  ICP_ITERS       ICP 迭代数（默认 40）
"""
import os
import sys
import time
from pathlib import Path

import numpy as np


# ── ICT-FaceKit 拓扑段（README 公开的固定分段） ──────────────────────────────
# #0 Face          verts [0:9409]     faces [0:9230]
# #1 Head and Neck verts [9409:11248] faces [9230:11144]
SKIN_V_END = 11248
SKIN_F_END = 11144

# MediaPipe FaceMesh 常用语义点（canonical_face_model 索引，固定拓扑）
MP_LM = {
    "nose_tip": 1,
    "chin": 199,
    "forehead": 10,
    "eye_out_L": 33,     # 图像左侧 = 被摄者右眼
    "eye_out_R": 263,
    "mouth_L": 61,
    "mouth_R": 291,
}


def log(msg):
    print(msg, flush=True)


def load_obj(path, triangulate=False):
    """最小 obj 解析：只要 v / f，保持顶点文件顺序。

    triangulate=True 时多边形面做扇形三角化（identity*.obj 混有 quad）。
    默认 False：generic_neutral_mesh 的面段划分（#0 Face / #1 Head and Neck）
    是按「多边形数」给的，三角化会打乱面序，破坏 skin 段切片。
    """
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                if len(idx) < 3:
                    continue
                if triangulate and len(idx) > 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
                else:
                    faces.append(idx[:3])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def extract_skin(verts, faces):
    """切出 skin 段（Face + Head and Neck），丢弃内部件。"""
    skin_f = faces[:SKIN_F_END]
    assert skin_f.max() < SKIN_V_END, f"skin face 索引越界: {skin_f.max()}"
    return verts[:SKIN_V_END].copy(), skin_f.copy()


def closest_point_on_triangle(p, a, b, c):
    """点 p 到三角形 abc 的最近点 + 重心坐标（Ericson 实时碰撞检测 5.1.5）。"""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab @ ap
    d2 = ac @ ap
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.array([1.0, 0.0, 0.0])

    bp = p - b
    d3 = ab @ bp
    d4 = ac @ bp
    if d3 >= 0.0 and d4 <= d3:
        return b, np.array([0.0, 1.0, 0.0])

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.array([1.0 - v, v, 0.0])

    cp = p - c
    d5 = ab @ cp
    d6 = ac @ cp
    if d6 >= 0.0 and d5 <= d6:
        return c, np.array([0.0, 0.0, 1.0])

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.array([1.0 - w, 0.0, w])

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.array([0.0, 1.0 - w, w])

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + v * ab + w * ac, np.array([1.0 - v - w, v, w])


class SurfaceQuery:
    """顶点 KDTree + 一环邻域三角形的精确点面最近点查询。"""

    def __init__(self, verts, faces):
        from scipy.spatial import cKDTree

        self.verts = verts
        self.faces = faces
        self.tree = cKDTree(verts)
        self.v2f = [[] for _ in range(len(verts))]
        for fi, f in enumerate(faces):
            a, b, c = verts[f]
            # 跳过退化三角形（零面积），否则最近点会出 NaN 让 SVD 崩
            if np.linalg.norm(np.cross(b - a, c - a)) < 1e-12:
                continue
            for vi in f:
                self.v2f[vi].append(fi)

    def query(self, pts):
        n = len(pts)
        out_p = np.empty((n, 3))
        out_fid = np.empty(n, dtype=np.int64)
        out_bary = np.empty((n, 3))
        out_d = np.empty(n)
        # 批量找最近顶点，再在其一环邻域里精查（比全局 BVH 便宜，且够精确）
        _, nn = self.tree.query(pts, k=8)
        nn = np.atleast_2d(nn)
        for i, p in enumerate(pts):
            cand = set()
            for vi in nn[i]:
                cand.update(self.v2f[vi])
            if not cand:  # 邻域面全退化 → 回退到全局最近顶点
                vi = int(nn[i][0])
                out_p[i] = self.verts[vi]
                out_fid[i] = self.v2f[vi][0] if self.v2f[vi] else 0
                out_bary[i] = np.array([1.0, 0.0, 0.0])
                out_d[i] = np.linalg.norm(self.verts[vi] - p)
                continue
            best = (None, -1, None, np.inf)
            for fi in cand:
                a, b, c = self.verts[self.faces[fi]]
                q, bary = closest_point_on_triangle(p, a, b, c)
                d = np.linalg.norm(q - p)
                if d < best[3]:
                    best = (q, fi, bary, d)
            out_p[i], out_fid[i], out_bary[i], out_d[i] = best
        return out_p, out_fid, out_bary, out_d


def umeyama(src, dst, weights=None, with_scale=False):
    """带/不带尺度的 Procrustes：with_scale=False 时返回 (R, t)，s 锁定 1。

    两个模板（ICT skin + MP canonical）都是真实 cm 级，尺度差 < 10%，用
    带尺度的 Procrustes 在迭代中会让 s 累积漂移（每轮 s 略 < 1，40 轮后
    P_std 从 4cm 塌缩到 0.3cm，整体缩成一点），所以默认禁用尺度自由度。
    """
    if weights is None:
        weights = np.ones(len(src))
    w = weights / weights.sum()
    mu_s = (src * w[:, None]).sum(0)
    mu_d = (dst * w[:, None]).sum(0)
    X = src - mu_s
    Y = dst - mu_d
    C = (X * w[:, None]).T @ Y
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    if with_scale:
        var = (w * (X ** 2).sum(1)).sum()
        s = float((S * np.diag(D)).sum() / max(var, 1e-12))
    else:
        s = 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def icp_align(surf, src_pts, yaw_deg, iters=40):
    """把 MediaPipe canonical 点云配到 ICT skin 表面。

    设计选择（每条都有血泪教训）：
      - src 固定为 MP canonical，每轮 P = T@src（不要把 P 当下轮的 src，
        否则经典 ICP 退化为自更新，40 轮内塌缩成一点）
      - 尺度锁 s=1：两个模板都是真实 cm 级，尺度差 < 8%；带尺度会让 Procrustes
        在 C 的某些奇异值退化时把 s 推到 0
      - 鼻尖 + 头顶 + 下巴 3 锚点定初值，再做 6DoF 刚性 Umeyama；
        比纯质心/PCA 初始化稳定，避免 yaw 滑移
      - yaw 双假设（0/180）：让 ICT 的前向自动判定

    返回 (T 4x4, median, mean)
    """
    yaw = np.deg2rad(yaw_deg)
    R_yaw = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    )

    # 找 ICT 鼻尖 / 头顶 / 下巴（face 部分限定）
    # yaw_deg: 0 → 假设 +z 前向（鼻尖 = z 最大）；180 → 假设 -z 前向
    face_v = surf.verts[:9409]
    z_col = face_v[:, 2]
    ict_nose = face_v[z_col.argmax()] if yaw_deg == 0 else face_v[z_col.argmin()]
    ict_chin = face_v[face_v[:, 1].argmin()]
    ict_top = face_v[face_v[:, 1].argmax()]

    # MP canonical 对应 3 点
    mp_nose = src_pts[MP_LM["nose_tip"]]
    mp_chin = src_pts[MP_LM["chin"]]
    mp_top = src_pts[MP_LM["forehead"]]

    # 锚点 6DoF 估计（s=1）：src_anchor 到 dst_anchor
    src_a = np.stack([mp_nose, mp_chin, mp_top])
    dst_a = np.stack([ict_nose, ict_chin, ict_top])
    s, R_a, t_a = umeyama(src_a, dst_a, with_scale=False)
    R_init = R_a
    t_init = ict_nose - R_a @ mp_nose  # 让鼻尖严格对齐

    T = np.eye(4)
    T[:3, :3] = R_init
    T[:3, 3] = t_init

    prev_med = np.inf
    no_improve = 0
    src = src_pts.copy()
    for it in range(iters):
        P = (T[:3, :3] @ src.T).T + T[:3, 3]
        q, fid, bary, d = surf.query(P)
        med = float(np.median(d))
        sigma = max(med, 1e-3)
        w = 1.0 / (1.0 + (d / (2.5 * sigma)) ** 2)
        # A 本身就是完整的 src→q 变换（Umeyama 的 src 是固定的 MP canonical），
        # 所以这里是 **替换** T，不是 T = A @ T（那样会把变换应用两次直接跑飞）
        _, R, t = umeyama(src, q, w, with_scale=False)
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = t
        T = A
        if it < 3 or it == iters - 1 or it % 10 == 0:
            print(
                f"     [yaw={yaw_deg}° iter={it:2d}] med_d={med:.4f} "
                f"P_std={P.std(0).round(2)}"
            )
        if med >= prev_med - 1e-5:
            no_improve += 1
            if no_improve >= 2:
                break
        else:
            no_improve = 0
        prev_med = med

    P = (T[:3, :3] @ src.T).T + T[:3, 3]
    q, fid, bary, d = surf.query(P)
    return T, float(np.median(d)), float(np.mean(d)), fid, bary


def pca(X, n_comp):
    """手写 PCA（环境无 sklearn）。X: (N, D) → mean, basis (n,D), std (n,)"""
    mean = X.mean(0)
    Xc = X - mean
    # N 很小（24）→ 走 Gram 矩阵（D×D 也很大：11248*3=33744 → 用样本空间）
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    basis = Vt[:n_comp]
    std = S[:n_comp] / max(np.sqrt(len(X) - 1), 1)
    return mean, basis, std


def save_debug_png(path, ict_verts, mp_pts, title=""):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        log(f"  ⚠️ 跳过 debug 图（matplotlib 不可用: {e}）")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [("front x-y", 0, 1), ("side z-y", 2, 1), ("top x-z", 0, 2)]
    for ax, (name, ax_x, ax_y) in zip(axes, panels):
        ax.scatter(ict_verts[:, ax_x], ict_verts[:, ax_y], s=0.4, c="0.75", label="ICT skin")
        ax.scatter(mp_pts[:, ax_x], mp_pts[:, ax_y], s=6, c="r", label="MP468 aligned")
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    log(f"  🖼️ 对齐验证图: {path}")


def main():
    weights_root = os.environ.get("WEIGHTS_ROOT", "")
    model_dir = os.environ.get(
        "MODEL_3DMM_DIR",
        os.path.join(weights_root, "ICT-FaceKit") if weights_root else "ICT-FaceKit",
    )
    model_dir = Path(model_dir)
    out_npz = Path(os.environ.get("OUT_NPZ", str(model_dir / "3dmm_template.npz")))
    debug_png = Path(
        os.environ.get("DEBUG_PNG", str(model_dir / "3dmm_align_check.png"))
    )
    icp_iters = int(os.environ.get("ICP_ITERS", "40"))
    skip_identity = int(os.environ.get("SKIP_IDENTITIES", "0"))

    generic = model_dir / "FaceXModel" / "generic_neutral_mesh.obj"
    canonical = model_dir / "mediapipe" / "canonical_face_model.obj"
    for p in (generic, canonical):
        if not p.exists():
            log(f"❌ 缺少模板文件: {p}")
            log("   先运行: bash 03d_prepare_3dmm.sh（会下载）")
            return 1

    log("🧱 准备 3DMM 头模板")
    log(f"  📂 模型根: {model_dir}")

    # ── 1) skin 网格 ───────────────────────────────────────────────────────
    t0 = time.time()
    V, F = load_obj(str(generic))
    log(f"  📥 generic: {V.shape[0]} verts / {F.shape[0]} faces ({time.time()-t0:.1f}s)")
    skin_v, skin_f = extract_skin(V, F)
    log(
        f"  ✂️ skin(face+head&neck): {skin_v.shape[0]} verts / {skin_f.shape[0]} faces"
    )
    log(f"     bbox size (cm): {(skin_v.max(0)-skin_v.min(0)).round(2)}")

    # ── 2) identity PCA ────────────────────────────────────────────────────
    id_paths = sorted((model_dir / "FaceXModel").glob("identity*.obj"))
    if skip_identity:
        log("  ⏭️ SKIP_IDENTITIES=1，跳过身份 PCA")
        id_paths = []
    if len(id_paths) >= 4:
        log(f"  👤 identity 网格: {len(id_paths)} 个 → PCA")
        stack = []
        for p in id_paths:
            try:
                vi, fi = load_obj(str(p))
            except Exception as e:
                log(f"  ⚠️ 跳过 {p.name}（load 失败: {e.__class__.__name__}）")
                continue
            if vi.shape[0] != V.shape[0]:
                log(f"  ⚠️ 跳过 {p.name}（顶点数 {vi.shape[0]} != {V.shape[0]}）")
                continue
            # 只要 skin 顶点（PCA 用）；面序在 quad 三角化后不可靠，不取
            stack.append(vi[:SKIN_V_END].ravel())
        if len(stack) < 4:
            log(f"  ⚠️ 有效 identity 仅 {len(stack)} 个，退化为通用头")
            id_mean = skin_v.copy()
            id_basis = np.zeros((0, skin_v.shape[0], 3))
            std = np.zeros(0)
        else:
            stack = np.asarray(stack)
            n_comp = min(stack.shape[0] - 1, 20)
            mean, basis, std = pca(stack, n_comp)
            id_mean = mean.reshape(-1, 3)
            id_basis = basis.reshape(n_comp, -1, 3)
            log(f"     PCA: {n_comp} 个形状基, std[:5]={std[:5].round(3)}")
    else:
        log("  ⚠️ identity 网格不足（<4），退化为「仅通用头 + 各向同性尺度」")
        id_mean = skin_v.copy()
        id_basis = np.zeros((0, skin_v.shape[0], 3))
        std = np.zeros(0)

    # ── 3) MediaPipe 468 ↔ ICT 对应 ────────────────────────────────────────
    mp_v, _ = load_obj(str(canonical))
    log(f"  📥 MediaPipe canonical: {mp_v.shape[0]} verts")

    surf = SurfaceQuery(skin_v, skin_f)
    log("  🔗 鲁棒 ICP 配准 (yaw 0/180 双假设)…")
    results = []
    for yaw in (0, 180):
        T, med, mean_d, fid, bary = icp_align(surf, mp_v, yaw, iters=icp_iters)
        # 总 s = T 线性部分的总缩放 = 三个奇异值的几何平均
        s_total = float(np.linalg.det(T[:3, :3]) ** (1 / 3))
        # ICP 收敛到「塌缩成一点」（s_total→0）或「爆炸」（s_total→∞）的伪解时，
        # median 反而很小（点全在表面一个区域）会被误选。加 s 范围约束排除。
        results.append((med, mean_d, T, fid, bary, yaw, s_total))
        log(
            f"     yaw={yaw:3d}°: median={med:.3f}cm  mean={mean_d:.3f}cm  s_total={s_total:.3f}"
        )
    # 只在 s 合理范围内选最小 med；塌缩解直接淘汰
    valid = [r for r in results if 0.7 <= r[6] <= 1.3]
    if not valid:
        log("  ⚠️ yaw 假设都跑飞了（s_total 越界），强制放宽到 [0.5, 1.5]")
        valid = [r for r in results if 0.5 <= r[6] <= 1.5]
    if not valid:
        valid = results
    valid.sort(key=lambda r: r[0])
    med, mean_d, T, fid, bary, yaw, _ = valid[0]
    log(
        f"  ✅ 采用 yaw={yaw}° (median={med:.3f}cm, mean={mean_d:.3f}cm, s_total={results[1 if yaw==180 else 0][6]:.3f})"
    )

    # landmark 在 ICT 模板空间的位置（barycentric 插值）
    lm_idx = skin_f[fid]                      # (468,3)
    lm_verts = (skin_v[lm_idx] * bary[:, :, None]).sum(1)   # (468,3)
    # 一致性自检：bary 插值点应与 ICP 后的点重合
    P_aligned = (T[:3, :3] @ mp_v.T).T + T[:3, 3]
    drift = np.linalg.norm(lm_verts - P_aligned, axis=1)
    log(f"     landmark 落面偏差: median={np.median(drift):.4f}cm max={drift.max():.3f}cm")

    # ── 数值 sanity check（无法看图时全靠这些指标） ────────────────────────
    # 1) 鼻尖必须是最靠前的 landmark（前向 = T 把 MP +z 映射到的方向）
    fwd = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    proj = lm_verts @ fwd
    nose_rank = int((proj > proj[MP_LM["nose_tip"]]).sum())
    log(f"     ✔️ 鼻尖前向排名: {nose_rank}/468  (0~5 = 正确)")
    # 2) 尺度一致性：MP canonical 与 ICT landmark 位置的「眼距 / 面高」应 ≈ 1
    def ratios(P):
        eye = np.linalg.norm(P[MP_LM["eye_out_L"]] - P[MP_LM["eye_out_R"]])
        face_h = np.linalg.norm(P[MP_LM["forehead"]] - P[MP_LM["chin"]])
        return eye, face_h

    e_mp, h_mp = ratios(mp_v)
    e_ict, h_ict = ratios(lm_verts)
    log(
        f"     ✔️ 眼距 MP={e_mp:.2f}cm → ICT={e_ict:.2f}cm (比 {e_ict/e_mp:.3f}) | "
        f"面高 {h_mp:.2f}→{h_ict:.2f} (比 {h_ict/h_mp:.3f})  [应 ≈1.0]"
    )
    # 3) landmark 贴合 ICT 表面的残差
    log(
        f"     ✔️ 落面残差: median={np.median(drift):.4f}cm p90={np.percentile(drift,90):.3f}cm"
    )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_npz),
        skin_verts=skin_v,
        skin_faces=skin_f,
        id_mean=id_mean,
        id_basis=id_basis,
        id_std=std,
        lm468_idx=lm_idx,
        lm468_bary=bary,
        lm468_verts=lm_verts,
        T_mp2ict=T,
        mp_canonical=mp_v,
        meta=np.array(
            [
                f"ict={generic.name}",
                f"n_identities={len(id_paths)}",
                f"icp_yaw={yaw}",
                f"icp_median={med:.4f}",
            ]
        ),
    )
    log(f"  💾 模板已保存: {out_npz}  ({out_npz.stat().st_size/1e6:.1f} MB)")

    save_debug_png(
        str(debug_png),
        skin_v,
        lm_verts,
        title=f"ICT skin (grey) + MediaPipe 468 aligned (red), yaw={yaw}, median={med:.3f}cm",
    )
    log("✅ 模板准备完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
