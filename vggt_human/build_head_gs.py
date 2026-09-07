#!/usr/bin/env python3
"""build_head_gs.py — 3DMM 头网格 → 头部高斯点云 (head_gs.ply)。

头/身/景拆分第 4 步 (A): 用 03f 拟合出的头网格生成高密度、带真实颜色的
头部高斯, 供 render_closeup 的覆盖率闭环推近与近景 loss mask 使用。

流程:
  1. 载入 head_mesh_p{pid}.npz (参考帧世界坐标 skin 网格, 含后脑+脖子)。
  2. 面积加权在三角面内均匀采样 N 个点 (默认 60k) → 位置 + 面法线。
  3. 多视角颜色采样: 每个点按「每帧拟合出的头位姿」变换到该帧, 投影取色,
     过滤 (在相机前 + 法线朝向相机 + 落在图像内 + 落在 SAM3 face mask 内),
     取有效观测的中位色 (对偶发遮挡鲁棒)。
  4. 各向异性高斯: 切向 σ = 0.5×点距, 法向 σ = 0.25×点距, 旋转把局部 z 轴
     对齐面法线 → 薄片贴合表面, 覆盖率比各向同性更实。
  5. 写 3DGS 标准 PLY (sh_degree=3) + 可视化 (黑底渲染 / 原图叠加 / 拼图)。

Env:
  RESULTS_DIR    输出根 (默认 /mnt/d/output/vggt_human_ms)
  SOURCE_DIR     COLMAP 源 (默认 $RESULTS_DIR/03_source)
  FIT_DIR        拟合输出 (默认 $RESULTS_DIR/03e_head_3dmm)
  MASKS_DIR      SAM3 face mask (默认 $RESULTS_DIR/03_sam3_face_masks_cleaned)
  PERSONS        只跑指定 pid (默认全部)
  HEAD_GS_POINTS 采样点数 (default 60000)
  HEAD_GS_OPACITY 不透明度 (default 0.9)
  HEAD_GS_SCALE_K 尺度系数 (default 1.0, 乘在切向/法向 σ 上)
  VIS_FRAMES     每人可视化帧数 (default 6)
  SEED           随机种子 (default 0)
  GS_DIR         gaussian-splatting 仓库
  DEVICE         cuda
"""
import os
import sys
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from face_center_3d import parse_colmap_cameras  # noqa: E402
from fit_head_3dmm import build_proj_matrices, qvec2rotmat  # noqa: E402

RESULTS_DIR = os.environ.get("RESULTS_DIR", "/mnt/d/output/vggt_human_ms")
SOURCE_DIR = os.environ.get("SOURCE_DIR", f"{RESULTS_DIR}/03_source")
FIT_DIR = os.environ.get("FIT_DIR", f"{RESULTS_DIR}/03e_head_3dmm")
MASKS_DIR = os.environ.get("MASKS_DIR", f"{RESULTS_DIR}/03_sam3_face_masks")
IMAGES_DIR = os.environ.get("IMAGES_DIR", f"{SOURCE_DIR}/images")
GS_DIR = os.environ.get("GS_DIR", os.path.expanduser("~/repos/gaussian-splatting"))
DEVICE = os.environ.get("DEVICE", "cuda")

N_POINTS = int(os.environ.get("HEAD_GS_POINTS", "60000"))
OPACITY = float(os.environ.get("HEAD_GS_OPACITY", "0.9"))
SCALE_K = float(os.environ.get("HEAD_GS_SCALE_K", "1.0"))
VIS_FRAMES = int(os.environ.get("VIS_FRAMES", "6"))
SEED = int(os.environ.get("SEED", "0"))

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")
_SH_C0 = 0.28209479177387814


def log(m):
    print(m, flush=True)


# ── 网格采样 ──────────────────────────────────────────────────────────────
def sample_surface(verts, faces, n_points, rng):
    """面积加权在三角面内均匀采样。返回 (pts, normals, face_ids, bary, spacing)。"""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    total = float(area.sum())
    prob = area / total
    fi = rng.choice(len(faces), size=n_points, p=prob)

    u = rng.random(n_points)
    v = rng.random(n_points)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    w = 1.0 - u - v
    bary = np.stack([w, u, v], axis=1)

    tri = faces[fi]
    pts = (verts[tri] * bary[:, :, None]).sum(1)
    nrm = cross[fi]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    spacing = math.sqrt(total / max(n_points, 1))
    return pts, nrm, fi, bary, spacing


def quat_from_z_axis(n):
    """把局部 z 轴对齐到 n (N,3, 单位向量) 的四元数 (w,x,y,z, N×4)。"""
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    # 选一个与 n 不平行的参考轴
    ref = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([0.0, 0.0, 1.0]),
                   np.array([0.0, 1.0, 0.0]))
    ref = np.broadcast_to(ref, n.shape).copy()
    x = np.cross(ref, n)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = np.cross(n, x)
    # 列向量基 → 旋转矩阵 M = [x, y, n] (局部→世界)
    M = np.stack([x, y, n], axis=2)          # (N,3,3)
    quats = np.empty((len(n), 4))
    tr = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
    m = tr > 0
    s = np.zeros(len(n))
    s[m] = np.sqrt(np.maximum(tr[m] + 1.0, 1e-12)) * 2
    quats[m, 0] = 0.25 * s[m]
    quats[m, 1] = (M[m, 2, 1] - M[m, 1, 2]) / s[m]
    quats[m, 2] = (M[m, 0, 2] - M[m, 2, 0]) / s[m]
    quats[m, 3] = (M[m, 1, 0] - M[m, 0, 1]) / s[m]
    # 退化分支（tr<=0），逐个处理（极少）
    for i in np.where(~m)[0]:
        Mi = M[i]
        if Mi[0, 0] > Mi[1, 1] and Mi[0, 0] > Mi[2, 2]:
            ss = math.sqrt(max(1.0 + Mi[0, 0] - Mi[1, 1] - Mi[2, 2], 1e-12)) * 2
            quats[i] = [(Mi[2, 1] - Mi[1, 2]) / ss, 0.25 * ss,
                        (Mi[0, 1] + Mi[1, 0]) / ss, (Mi[0, 2] + Mi[2, 0]) / ss]
        elif Mi[1, 1] > Mi[2, 2]:
            ss = math.sqrt(max(1.0 + Mi[1, 1] - Mi[0, 0] - Mi[2, 2], 1e-12)) * 2
            quats[i] = [(Mi[0, 2] - Mi[2, 0]) / ss, (Mi[0, 1] + Mi[1, 0]) / ss,
                        0.25 * ss, (Mi[1, 2] + Mi[2, 1]) / ss]
        else:
            ss = math.sqrt(max(1.0 + Mi[2, 2] - Mi[0, 0] - Mi[1, 1], 1e-12)) * 2
            quats[i] = [(Mi[1, 0] - Mi[0, 1]) / ss, (Mi[0, 2] + Mi[2, 0]) / ss,
                        (Mi[1, 2] + Mi[2, 1]) / ss, 0.25 * ss]
    return quats


def find_image(stem):
    for ext in IMG_EXTS:
        p = Path(IMAGES_DIR) / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


# ── 多视角颜色采样 ────────────────────────────────────────────────────────
def sample_colors(pts, nrm, fit, projs, masks, verbose=True):
    """返回 (rgb (N,3) 0..1, n_mask (N,), n_vis (N,))。

    每个点按每帧头位姿变换后投影取色。两级观测:
      A. 落在 SAM3 face mask 内 (最可信, 正脸直接可见)
      B. 仅通过几何可见性 (相机前 + 法线朝外 + 在画幅内)
    优先用 A (需 ≥3 个观测), 否则退化到 B —— 后脑/头顶/头发不在 face mask 内,
    只能靠 B。两级都取中位数, 对偶发遮挡鲁棒。
    """
    from scipy.ndimage import map_coordinates

    N = len(pts)
    R_ref = np.asarray(fit["R_ref"])
    t_ref = np.asarray(fit["t_ref"])
    c_mask = np.full((0, N, 3), np.nan, dtype=np.float32)
    c_vis = np.full((0, N, 3), np.nan, dtype=np.float32)
    n_mask = np.zeros(N, dtype=np.int32)
    n_vis = np.zeros(N, dtype=np.int32)

    stems = [s for s in fit["per_frame"] if s in projs]
    for stem in stems:
        img_path = find_image(stem)
        if img_path is None:
            continue
        pf = fit["per_frame"][stem]
        R_f = np.asarray(pf["R"])
        t_f = np.asarray(pf["t"])
        P, K = projs[stem]

        # 参考帧 → 该帧的刚体: p_f = A p_ref + b
        A = R_f @ R_ref.T
        b = t_f - A @ t_ref
        p_f = (A @ pts.T).T + b
        n_f = (A @ nrm.T).T

        # 相机中心 C (world): P = K[R|t] → C = -R^T t
        R_c = P[:, :3]
        t_c = P[:, 3]
        R_c = np.linalg.inv(K) @ R_c
        t_c = np.linalg.inv(K) @ t_c
        C = -R_c.T @ t_c

        hom = (P[:, :3] @ p_f.T).T + P[:, 3]      # (N,3) 齐次
        z = hom[:, 2]
        valid = z > 1e-6
        uv = np.zeros((N, 2))
        uv[valid] = hom[valid, :2] / z[valid, None]

        img = np.asarray(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]
        inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & \
                 (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)

        # 法线朝向相机 (点朝外的面才可见)
        view = C - p_f
        view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
        facing = (n_f * view).sum(1) > 0.1

        ok = inside & facing
        inmask = np.zeros(N, dtype=bool)
        if masks is not None and stem in masks:
            mk = masks[stem]
            ix = np.clip(uv[:, 0].astype(np.int32), 0, W - 1)
            iy = np.clip(uv[:, 1].astype(np.int32), 0, H - 1)
            inmask = ok & mk[iy, ix]

        if not (ok.any() or inmask.any()):
            continue

        imgf = img.astype(np.float32) / 255.0
        # 双线性采样 (map_coordinates 用 (row, col) 顺序)
        def sample_at(sel):
            coords = np.stack([uv[sel, 1], uv[sel, 0]], axis=0)
            return np.stack([
                map_coordinates(imgf[:, :, c], coords, order=1, mode="nearest")
                for c in range(3)
            ], axis=1)

        if inmask.any():
            f = np.full((1, N, 3), np.nan, dtype=np.float32)
            f[0, inmask] = sample_at(inmask)
            c_mask = np.concatenate([c_mask, f], axis=0)
            n_mask[inmask] += 1
        if ok.any():
            f = np.full((1, N, 3), np.nan, dtype=np.float32)
            f[0, ok] = sample_at(ok)
            c_vis = np.concatenate([c_vis, f], axis=0)
            n_vis[ok] += 1
        if verbose:
            log(f"      {stem}: mask内 {int(inmask.sum())} 可见 {int(ok.sum())} / {N}")

    med = np.full((N, 3), 0.5, dtype=np.float32)
    if c_vis.shape[0]:
        with np.errstate(all="ignore"):
            m_vis = np.nanmedian(c_vis, axis=0)
        med = np.where(np.isfinite(m_vis), m_vis, 0.5).astype(np.float32)
    if c_mask.shape[0]:
        with np.errstate(all="ignore"):
            m_mk = np.nanmedian(c_mask, axis=0)
        good = np.isfinite(m_mk).all(axis=1) & (n_mask >= 3)
        med[good] = m_mk[good]

    # 从无任何观测的点（后脑/脖子底等相机从没看到过的区域）: 取几何最近的有色点颜色,
    # 避免灰色斑块。用 KD-tree 近邻填充（头顶→头发色, 后脑→后脑邻域色）。
    no_obs = n_vis == 0
    if no_obs.any() and (~no_obs).any():
        from scipy.spatial import cKDTree
        tree = cKDTree(pts[~no_obs])
        _, nn = tree.query(pts[no_obs], k=1)
        med[no_obs] = med[~no_obs][nn]
        if verbose:
            log(f"   🔧 近邻补色: {int(no_obs.sum())} 个无观测点")
    return np.clip(med, 0.0, 1.0), n_mask, n_vis


# ── PLY 写出 ─────────────────────────────────────────────────────────────
def write_gs_ply(path, xyz, rgb, scale, quat, opacity, sh_degree=3):
    from plyfile import PlyData, PlyElement

    N = len(xyz)
    f_rest_n = ((sh_degree + 1) ** 2 - 1) * 3
    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    attrs += [f"f_dc_{i}" for i in range(3)]
    attrs += [f"f_rest_{i}" for i in range(f_rest_n)]
    attrs += ["opacity"] + [f"scale_{i}" for i in range(3)] + \
             [f"rot_{i}" for i in range(4)]

    f_dc = (rgb - 0.5) / _SH_C0                      # (N,3)
    zeros = np.zeros((N, 3), dtype=np.float32)
    op = np.full((N, 1), math.log(opacity / (1 - opacity)), dtype=np.float32)
    sc = np.log(scale.astype(np.float32))            # (N,3) 存 log
    mat = np.concatenate([xyz, zeros, f_dc, np.zeros((N, f_rest_n), np.float32),
                          op, sc, quat], axis=1).astype(np.float32)
    el = PlyElement.describe(
        np.array([tuple(r) for r in mat], dtype=[(a, "f4") for a in attrs]),
        "vertex",
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([el]).write(str(path))
    return sum(1 for _ in attrs)


# ── 可视化 ───────────────────────────────────────────────────────────────
def visualize(ply_path, fit, views, out_dir, pid, vis_frames):
    """把 head_gs 渲染到若干输入视角: 黑底纯渲染 + 原图叠加 + 横向拼图。"""
    import torch
    sys.path.insert(0, GS_DIR)
    from argparse import Namespace
    from scene import GaussianModel
    from scene.cameras import Camera
    from gaussian_renderer import render

    g = GaussianModel(3)
    g.load_ply(str(ply_path))
    bg_black = torch.zeros(3, device=DEVICE)
    bg_white = torch.ones(3, device=DEVICE)
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                     antialiasing=False, debug=False)

    stems = [s for s in fit["per_frame"] if s in views]
    if not stems:
        return []
    idx = np.linspace(0, len(stems) - 1, min(vis_frames, len(stems))).astype(int)
    picked = [stems[i] for i in sorted(set(idx.tolist()))]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays, renders = [], []
    R_ref = np.asarray(fit["R_ref"])
    t_ref = np.asarray(fit["t_ref"])
    xyz0 = g.get_xyz.detach().clone()
    rot0 = g.get_rotation.detach().clone()
    for stem in picked:
        v = views[stem]
        img_path = find_image(stem)
        if img_path is None:
            continue
        FoVx = 2 * math.atan(v["W"] / (2 * v["fx"]))
        FoVy = 2 * math.atan(v["H"] / (2 * v["fy"]))
        R = qvec2rotmat(v["qvec"])
        T = np.asarray(v["tvec"])
        # 逐帧刚性重摆: head_gs 建在中位姿态, 按该帧拟合位姿变换过去。
        # 不重摆的话, 头动得多的帧(如 p02)会表现为"模型头偏离真人"的假阳性。
        pf_pose = fit["per_frame"].get(stem)
        if pf_pose is not None:
            from scipy.spatial.transform import Rotation as _Rot
            R_f = np.asarray(pf_pose["R"], dtype=np.float64)
            t_f = np.asarray(pf_pose["t"], dtype=np.float64)
            A = R_f @ R_ref.T
            b = t_f - A @ t_ref
            xyz_np = xyz0.cpu().numpy()
            rot_np = rot0.cpu().numpy()  # (N,4) w,x,y,z
            xyz_new = (A @ xyz_np.T).T + b
            qA = _Rot.from_matrix(A)
            q_old = _Rot.from_quat(rot_np[:, [1, 2, 3, 0]])
            q_new = (qA * q_old).as_quat()  # x,y,z,w
            rot_new = np.stack(
                [q_new[:, 3], q_new[:, 0], q_new[:, 1], q_new[:, 2]], axis=1
            )
            g.get_xyz.data = torch.tensor(xyz_new, dtype=xyz0.dtype,
                                          device=xyz0.device)
            g.get_rotation.data = torch.tensor(rot_new, dtype=rot0.dtype,
                                               device=rot0.device)
        else:
            g.get_xyz.data = xyz0.clone()
            g.get_rotation.data = rot0.clone()
        cam = Camera(resolution=(v["W"], v["H"]), colmap_id=0, R=R.T, T=T,
                     FoVx=FoVx, FoVy=FoVy, depth_params=None,
                     image=Image.fromarray(
                         np.zeros((v["H"], v["W"], 3), dtype=np.uint8)),
                     invdepthmap=None, image_name=stem, uid=0, data_device=DEVICE)
        with torch.no_grad():
            rgb = render(cam, g, pipe, bg_black)["render"].clamp(0, 1)
            wht = render(cam, g, pipe, bg_white)["render"].clamp(0, 1)
        alpha = 1 - (wht - rgb).mean(dim=0, keepdim=True).clamp(0, 1)
        rn = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        an = alpha[0].cpu().numpy()

        orig = np.asarray(Image.open(img_path).convert("RGB").resize((v["W"], v["H"])))
        ov = (orig * (1 - an[..., None]) + rn * an[..., None]).astype(np.uint8)

        name = f"p{pid}_{stem}"
        Image.fromarray(rn).save(out_dir / f"head_render_{name}.png")
        Image.fromarray(ov).save(out_dir / f"head_overlay_{name}.png")
        renders.append(rn)
        overlays.append(ov)

    # 拼图 (横向), 便于一眼对比
    for tag, imgs in (("overlay", overlays), ("render", renders)):
        if not imgs:
            continue
        h = min(im.shape[0] for im in imgs)
        w = min(im.shape[1] for im in imgs)
        mos = np.concatenate([im[:h, :w] for im in imgs], axis=1)
        Image.fromarray(mos).save(out_dir / f"head_{tag}_mosaic_p{pid}.png")

    del g
    torch.cuda.empty_cache()
    return [str(out_dir / f"head_overlay_mosaic_p{pid}.png")]


def main():
    t0 = time.time()
    fit_path = Path(FIT_DIR) / "head_fit.json"
    if not fit_path.is_file():
        sys.exit(f"❌ 缺少拟合结果: {fit_path} (先跑 03f_fit_head_3dmm.sh)")
    fit = json.loads(fit_path.read_text())
    views = parse_colmap_cameras(SOURCE_DIR)
    views = {v["stem"]: v for v in views}
    projs = build_proj_matrices(list(views.values()))
    log(f"📷 相机 {len(views)} | 拟合 {fit_path}")

    # masks (可选, 用于过滤取色帧)
    masks = None
    if Path(MASKS_DIR).is_dir():
        masks = {}
        for f in sorted(os.listdir(MASKS_DIR)):
            if ".p" not in f or not f.endswith(".mask.png"):
                continue
            stem = f.split(".p")[0]
            pid_s = f.split(".p")[1].split(".")[0]
            if not pid_s.isdigit():
                continue
            masks.setdefault(int(pid_s), {})[stem] = \
                np.asarray(Image.open(Path(MASKS_DIR) / f).convert("L")) > 127
        log(f"🎭 masks: {sum(len(v) for v in masks.values())} 张 ({len(masks)} 人)")

    want = os.environ.get("PERSONS", "")
    # 接受 "00" / "p00" / "0" 三种写法
    want = {w.strip().lstrip("pP").zfill(2) for w in want.split(",") if w.strip()} or None

    out_meta = {}
    for pid, pf in fit["persons"].items():
        if want and f"{int(pid):02d}" not in want:
            continue
        mesh_path = Path(FIT_DIR) / f"head_mesh_p{int(pid):02d}.npz"
        if not mesh_path.is_file():
            log(f"  ⚠️ 跳过 p{pid}: 无 {mesh_path}")
            continue
        log(f"\n👤 p{pid}: 网格 {mesh_path.name}, 观测 {pf['n_frames']} 帧, "
            f"RMS {pf['reproj_rms_px']:.2f}px")
        z = np.load(mesh_path)
        verts, faces = z["verts_world"], z["faces"]

        # 世界单位 → 毫米: 模板是 cm 级, 拟合尺度 s 把 cm 变世界单位
        w2mm = 10.0 / float(pf["scale"])
        rng = np.random.default_rng(SEED)
        pts, nrm, _, _, spacing = sample_surface(verts, faces, N_POINTS, rng)
        log(f"   🔹 采样 {len(pts)} 点, 点距 {spacing:.5f} 世界单位 "
            f"(≈{spacing * w2mm:.1f}mm)")

        pmasks = masks.get(int(pid)) if masks else None
        rgb, n_mask, n_vis = sample_colors(pts, nrm, pf, projs, pmasks)
        log(f"   🎨 取色: mask内≥3帧 {(n_mask >= 3).sum()} 点, "
            f"仅可见 {(n_mask < 3).__and__(n_vis > 0).sum()} 点, "
            f"无色 {(n_vis == 0).sum()} 点")

        sigma_t = spacing * 0.5 * SCALE_K
        sigma_n = spacing * 0.25 * SCALE_K
        scale = np.tile([sigma_t, sigma_t, sigma_n], (len(pts), 1))
        quat = quat_from_z_axis(nrm)

        ply_path = Path(FIT_DIR) / f"head_gs_p{int(pid):02d}.ply"
        write_gs_ply(ply_path, pts.astype(np.float32), rgb.astype(np.float32),
                     scale, quat.astype(np.float32), OPACITY)
        log(f"   💾 {ply_path.name} ({ply_path.stat().st_size / 1e6:.1f}MB)")

        vis_dir = Path(FIT_DIR) / "vis"
        try:
            outs = visualize(ply_path, pf, views, vis_dir, f"{int(pid):02d}", VIS_FRAMES)
            for o in outs:
                log(f"   🖼️  {o}")
        except Exception as e:
            log(f"   ⚠️ 可视化失败: {type(e).__name__}: {e}")

        out_meta[pid] = dict(
            ply=str(ply_path), n_points=len(pts),
            spacing_world=spacing, spacing_mm=spacing * w2mm,
            sigma_tangent=sigma_t, sigma_normal=sigma_n,
            n_mask_median=int(np.median(n_mask)), n_vis_median=int(np.median(n_vis)),
            n_no_obs=int((n_vis == 0).sum()),
            head_center=pf["head_center"], head_geom_center=pf["head_geom_center"],
            head_radius_p90=pf["head_radius_p90"],
        )

    (Path(FIT_DIR) / "head_gs_meta.json").write_text(
        json.dumps({"meta": dict(points=N_POINTS, opacity=OPACITY,
                                 scale_k=SCALE_K, seed=SEED),
                    "persons": out_meta}, indent=2))
    log(f"\n✅ 完成 {time.time() - t0:.0f}s → {FIT_DIR}/head_gs_meta.json")


if __name__ == "__main__":
    main()
