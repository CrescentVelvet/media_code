#!/usr/bin/env python3
# qc_gravity_axis.py — 量化「模型在训练坐标系里歪了多少」。
#
# 背景：01c（POS  E_ADJUST）把场景重参数化到「右=X、上=+Y、-视线=Z」的重力对齐系，
# 训练出的 point_cloud.ply 就在这个系里。若重力估计有残差，SuperSplat 里看着就歪。
# 本工具用三路独立证据互证，判断倾斜是「估计误差」还是「查看器约定」：
#
#   ① 相机 down 向量（w2c R 的第 1 行 = 图像下方在世界系的方向）
#      —— 只看 roll 影响，pitch 不影响；对齐良好时应 ≈ (0,1,0)
#   ② 相机轨迹平面法向（PCA 最小特征向量）
#      —— 手持环绕拍摄的高度近似不变时，轨迹平面 ≈ 水平面，法向 ≈ 重力
#      —— 若拍摄高度变化大（俯拍/蹲下），此法向自身有偏差，看 RMS 判断可信度
#   ③ 主体长轴（原点附近点云的 PCA 最大特征向量）
#      —— 站立/坐姿主体长轴 ≈ 竖直；蹲/卧姿态此路无效（脚本会标注）
#
# 用法：
#   MODEL_PATH=<model 目录> python deformable_human/qc_gravity_axis.py
#   MODEL_PATH=<model 目录> python deformable_human/qc_gravity_axis.py --subject-radius 2.5
#
# 输出：各角度（度）。① ② 都明显小于 2° → 模型本身不歪，问题在查看器约定；
# 否则即为重力估计残差，量级 = 需要修正的角度。
import argparse
import json
import math
import os
import sys

import numpy as np


# ── 极简 PLY 读取（只取 xyz；3DGS 点云无面片，binary_little_endian）───────────
def read_ply_xyz(path):
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            sys.exit(f"❌ 不是 PLY: {path}")
        fmt = None
        n_vertex = 0
        props = []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                sys.exit(f"❌ PLY 头异常: {path}")
            s = line.decode("ascii", "replace").strip()
            if s.startswith("format"):
                fmt = s.split()[1]
            elif s.startswith("element vertex"):
                n_vertex = int(s.split()[2])
                in_vertex = True
            elif s.startswith("element"):
                in_vertex = False
            elif s.startswith("property") and in_vertex:
                parts = s.split()
                if parts[1] == "list":
                    sys.exit("❌ 暂不支持 list property")
                props.append((parts[2], parts[1]))  # (name, type)
            elif s == "end_header":
                break
        if fmt != "binary_little_endian":
            sys.exit(f"❌ 只支持 binary_little_endian，实为 {fmt}")

        np_types = {"float": "<f4", "float32": "<f4", "double": "<f8",
                    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
                    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
                    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4"}
        dtype = np.dtype([(n, np_types[t]) for n, t in props])
        buf = f.read(n_vertex * dtype.itemsize)
        data = np.frombuffer(buf, dtype=dtype, count=n_vertex)
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)


def angle_to_y(v):
    """v 与 +Y 的夹角（度）。"""
    v = v / (np.linalg.norm(v) + 1e-12)
    return math.degrees(math.acos(min(1.0, max(-1.0, float(v[1])))))


def pca_axis(pts, largest=True):
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    w, V = np.linalg.eigh(cov)          # 升序
    return V[:, -1] if largest else V[:, 0], w, c


def person_points(model_path, pts, cams, R, Cpos, conv, n_views=8, stride=2):
    """用 SAM3 mask 反投影挑出人体高斯点（不依赖位姿约定的硬证据来源）。

    model_path = <scene>/model → mask 在 <scene>/colmap_scene/masks/{img_name}.png
    """
    scene_root = os.path.dirname(model_path.rstrip("/"))
    masks_dir = os.path.join(scene_root, "colmap_scene", "masks")
    if not os.path.isdir(masks_dir):
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    sub = pts[::stride]
    picked = []
    idxs = np.linspace(0, len(cams) - 1, min(n_views, len(cams))).astype(int)
    for i in idxs:
        c = cams[i]
        mp_path = os.path.join(masks_dir, f"{c['img_name']}.png")
        if not os.path.exists(mp_path):
            continue
        m = np.asarray(Image.open(mp_path).convert("L"))
        H, W = m.shape
        rel = sub - Cpos[i]
        cam = rel @ R[i] if conv == "c2w" else rel @ R[i].T
        z = cam[:, 2]
        ok = z > 1e-6
        u = np.full(len(cam), -1)
        v = np.full(len(cam), -1)
        u[ok] = c["fx"] * cam[ok, 0] / z[ok] + W / 2
        v[ok] = c["fy"] * cam[ok, 1] / z[ok] + H / 2
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        sel = np.zeros(len(cam), dtype=bool)
        sel[ok] = m[vi[ok], ui[ok]] > 127
        picked.append(sub[sel])
    if not picked:
        return None
    return np.concatenate(picked, axis=0)


def ransac_plane(pts, iters=300, tol=0.03, max_pts=60000, seed=0):
    """RANSAC 求主导平面（室内场景的地面）。返回 (法向, 内点数, RMS)。"""
    rng = np.random.default_rng(seed)
    p = pts[:: max(1, len(pts) // max_pts)]
    best = (None, 0, None)
    n = len(p)
    for _ in range(iters):
        a, b, c = p[rng.choice(n, 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        d = (p - a) @ nrm
        inl = np.abs(d) < tol
        cnt = int(inl.sum())
        if cnt > best[1]:
            best = (nrm, cnt, float(d[inl].std()))
    if best[0] is None:
        return None, 0, float("nan")
    return best[0], best[1], best[2]


def gravity_from_mask_orientation(model_path, cams, R, Cpos, conv):
    """用图像空间「人体主轴」反解世界竖直方向（不依赖任何位姿约定假设）。

    原理：人体长轴 ≈ 竖直。世界竖直方向 g 经相机投影后，在图像上的方向应为
    J_i g，其中 J_i 是 (u,v) 对世界点的雅可比。求 |g|=1 使 Σ‖(I-u_i u_iᵀ) J_i g‖²
    最小的 g —— 即 Σ J_iᵀ(I-u_i u_iᵀ)J_i 的最小特征向量。

    主体定位：POSE_ADJUST 后主体居中在原点，故取世界原点作为参考 3D 点算雅可比，
    深度用该相机到原点的距离（不是精确点，但雅可比只依赖深度量级，已足够）。
    """
    scene_root = os.path.dirname(model_path.rstrip("/"))
    masks_dir = os.path.join(scene_root, "colmap_scene", "masks")
    if not os.path.isdir(masks_dir):
        return None, 0
    try:
        from PIL import Image
    except ImportError:
        return None, 0

    M = np.zeros((3, 3))
    used = 0
    for i, c in enumerate(cams):
        p = os.path.join(masks_dir, f"{c['img_name']}.png")
        if not os.path.exists(p):
            continue
        m = np.asarray(Image.open(p).convert("L")) > 127
        if m.sum() < 200:
            continue
        ys, xs = np.where(m)
        # 图像空间主轴（2x2 协方差最小特征向量 = 长轴方向）
        q = np.stack([xs - xs.mean(), ys - ys.mean()], axis=1).astype(np.float64)
        cov = q.T @ q / len(q)
        w, V = np.linalg.eigh(cov)
        u = V[:, -1]                                  # (2,) 单位
        aniso = math.sqrt(w[-1] / max(w[-2], 1e-12))
        if aniso < 1.15:                              # 接近各向同性（人形不明确）→ 跳过
            continue

        C = Cpos[i]
        Rc2w = R[i] if conv == "c2w" else R[i].T      # c2w 旋转
        z0 = float(np.linalg.norm(C))                 # 原点到相机距离 ≈ 深度
        # X_cam = Rc2wᵀ (X - C)，在 X=0 处：x_cam = -Rc2wᵀ C
        xc = -Rc2w.T @ C
        fx, fy = c["fx"], c["fy"]
        # d(u,v)/dX_cam 在 X=0
        Ju = np.array([fx / z0, 0.0, -fx * xc[0] / z0 ** 2])
        Jv = np.array([0.0, fy / z0, -fy * xc[1] / z0 ** 2])
        Jcam = np.stack([Ju, Jv], axis=0)             # (2,3)
        J = Jcam @ Rc2w.T                             # (2,3) 对世界坐标
        P = np.eye(2) - np.outer(u, u)                # 去掉沿 u 的分量
        M += J.T @ P @ J
        used += 1
    if used < 5:
        return None, used
    w, V = np.linalg.eigh(M)
    return V[:, 0], used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    ap.add_argument("--iteration", default="10000")
    ap.add_argument("--subject-radius", type=float, default=2.5,
                    help="取原点附近该半径内的点做主体长轴估计（居中后原点≈主体）")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    mp = os.path.expanduser(args.model_path)
    if not mp or not os.path.isdir(mp):
        sys.exit("❌ 需要 MODEL_PATH（model 目录）")
    label = args.label or os.path.basename(os.path.dirname(mp))

    cams = json.load(open(os.path.join(mp, "cameras.json")))
    R = np.array([c["rotation"] for c in cams], dtype=np.float64)   # (n,3,3)
    Cpos = np.array([c["position"] for c in cams], dtype=np.float64)

    # ── 约定自检：cameras.json 的 rotation 是 w2c 还是 c2w ──
    # 判据 A：用 input.ply（稀疏点，训练坐标系）投影到第 0 个相机，
    #         正确约定下「z>0 且落在像幅内」的点占多数。
    # 判据 B：POS_ADJUST 后主体居中在原点，相机前向（camera +z）应指向原点。
    conv = "c2w"
    inp0 = os.path.join(mp, "input.ply")
    if os.path.exists(inp0):
        sp = read_ply_xyz(inp0)
        c0, w0, h0 = Cpos[0], cams[0]["width"], cams[0]["height"]
        fx0 = cams[0]["fx"]
        scores = {}
        for name, camv in (("w2c", (sp - c0) @ R[0].T), ("c2w", (sp - c0) @ R[0])):
            z = camv[:, 2]
            u = fx0 * camv[:, 0] / np.where(z == 0, 1e-9, z) + w0 / 2
            v = fx0 * camv[:, 1] / np.where(z == 0, 1e-9, z) + h0 / 2
            frac = float(np.mean((z > 0) & (u > 0) & (u < w0) & (v > 0) & (v < h0)))
            scores[name] = frac
            print(f"   [自检A] 按 {name} 解译: 落在首帧像幅内 {frac*100:.1f}%")
        conv = max(scores, key=scores.get)
        # 判据 B：两种解译下「前向 vs 指向原点」的平均余弦
        to_origin = -Cpos / (np.linalg.norm(Cpos, axis=1, keepdims=True) + 1e-12)
        fwd_c2w = R[:, :, 2]                    # c2w：第 2 列 = 相机 +z 在世界系
        fwd_w2c = R[:, 2, :]                    # w2c：第 2 行 = 世界系里映射到相机 +z 的方向
        cos_c2w = float(np.mean((fwd_c2w * to_origin).sum(axis=1)))
        cos_w2c = float(np.mean((fwd_w2c * to_origin).sum(axis=1)))
        print(f"   [自检B] 前向·指向原点 平均余弦: c2w {cos_c2w:+.3f} | w2c {cos_w2c:+.3f}")
        conv = "c2w" if cos_c2w >= cos_w2c else "w2c"
        print(f"   → 采用 {conv} 解译（A 判定 {max(scores, key=scores.get)}）")

    print(f"══ {label} ══  ({len(cams)} 相机)")

    # ① 相机「图像下方」方向（COLMAP/OpenCV 约定相机 y 轴朝下）
    #    POS_ADJUST 把 +Y 定义为「上」（pred_rot 第 2 列 = -gravity_down），
    #    故对齐良好时 图像下方 ≈ -Y，即 angle_to_y ≈ 180°；这里报 |180-θ| = 残差。
    down = R[:, :, 1] if conv == "c2w" else R[:, 1, :]
    mean_down = down.mean(axis=0)
    resid_down = 180.0 - angle_to_y(mean_down)
    rolls = np.array([180.0 - angle_to_y(d) for d in down])
    print(f"① 相机图像下方 vs -Y 残差   : {resid_down:6.2f}°   "
          f"（逐相机 {rolls.mean():.2f}° ± {rolls.std():.2f}°，max {rolls.max():.2f}°）")
    print("   ⚠️ ① 只在相机「接近平视」时才等于残差；俯拍场景里图像下方本就≈水平，"
          "会被俯仰角污染（见 ①b）。判倾斜以 ② ④ ⑤ 为准。")

    # ② 相机轨迹平面法向
    n, eig, ccen = pca_axis(Cpos, largest=False)
    # 法向只定到符号，取与 -Y 或 +Y 更接近的方向
    if n[1] < 0:
        n = -n
    resid = (Cpos - ccen) @ n
    print(f"② 轨迹平面法向与 +Y 夹角   : {angle_to_y(n):6.2f}°   "
          f"（平面拟合残差 RMS {resid.std():.3f}，相机距原点中位数 "
          f"{np.median(np.linalg.norm(Cpos, axis=1)):.2f}）")

    # ①b 俯仰角分布：俯拍场景里「图像下方」本就不指向重力，解释 ① 的大角度
    fwd = R[:, :, 2] if conv == "c2w" else R[:, 2, :]
    pitches = np.array([90.0 - angle_to_y(f) for f in fwd])
    print(f"①b 俯仰角 pitch（0=平视, -90=正下看）: "
          f"均值 {pitches.mean():+6.1f}° 标准差 {pitches.std():.1f}°，"
          f"范围 [{pitches.min():.1f}, {pitches.max():.1f}]")

    # ③ 人体点云（用 SAM3 mask 反投影挑高斯）+ 体轴方向
    #    站立/坐姿人体长轴 ≈ 竖直 → 这是不依赖任何位姿约定的硬证据。
    ply = os.path.join(mp, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    pts = None
    if os.path.exists(ply):
        pts = read_ply_xyz(ply)
        body = person_points(mp, pts, cams, R, Cpos, conv, n_views=8)
        if body is not None and len(body) > 200:
            ax, w, _ = pca_axis(body, largest=True)
            aa = angle_to_y(ax)
            ratio = math.sqrt(w[-1] / max(w[-2], 1e-12))
            # 长轴无符号 → 与 +Y 或 -Y 取小者（站立/倒立都算对齐）
            aa = min(aa, 180.0 - aa)
            print(f"③ 人体长轴 vs 竖直轴        : {aa:6.2f}°   "
                  f"（mask 反投影得 {len(body)} 点，各向异性 {ratio:.2f}）")
            print("   ⚠️ 蹲/卧/躺姿态此路不可信（长轴非竖直），矩阵里会标注")
        else:
            print("③ 人体长轴：mask 反投影点不足，跳过（检查 colmap_scene/masks）")

    # ④ 地面法向（RANSAC 主导平面）——室内场景的强重力线索
    if pts is not None:
        n_floor, inl, rms = ransac_plane(pts, iters=300, tol=0.03)
        if n_floor is not None:
            if n_floor[1] < 0:
                n_floor = -n_floor
            print(f"④ 地面法向 vs +Y            : {angle_to_y(n_floor):6.2f}°   "
                  f"（内点 {inl}，RMS {rms:.4f}）")

    # ⑤ 由 mask 图像主轴反解的重力方向（语义级硬参考：人体长轴≈竖直）
    g, used = gravity_from_mask_orientation(mp, cams, R, Cpos, conv)
    if g is not None:
        a = angle_to_y(g)
        a = min(a, 180.0 - a)          # 竖直方向无符号
        print(f"⑤ mask 主轴反解重力 vs 竖直轴: {a:6.2f}°   （用 {used} 帧）")
        print("   ← 这一路只用「人体在画面里是竖的」这一语义事实，最可信")

    inp = os.path.join(mp, "input.ply")
    if os.path.exists(inp):
        sp = read_ply_xyz(inp)
        print(f"   稀疏点云 {len(sp)} 点，质心 {np.round(sp.mean(axis=0), 3).tolist()}")



if __name__ == "__main__":
    main()
