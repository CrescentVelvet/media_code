#!/usr/bin/env python3
"""render_closeup.py — Generate closeup + pitch-extrapolated views for face enhancement.

Strategy (from grill-me design):
  1. Parse COLMAP cameras, compute azimuth for each (relative to face center).
  2. Bin cameras into 10 azimuth bins (36° each).
  3. Per bin, pick the frame with highest SAM2 mask coverage = 10 base cameras.
  4. For each base camera: move along optical axis toward face center until
     SAM2 mask coverage >= 40% (min distance = 30% of original).
  5. At closeup position, generate 2 extrapolated views: pitch +10°, pitch -10°
     (rotation around face center's horizontal axis).
  6. 3DGS render all 20 extrapolated views → save RGB + alpha + poses.

Output:
  单人: 06c_closeup_renders/closeup_0001.png ... + 06c_closeup_poses.json
  多人 (FACE_CENTER JSON 含 persons, 由 face_center_3d --multi 生成):
       每人独立跑完整流程, 输出 06c_closeup_renders_p{pid:02d}/ ...
       PERSONS=p00,p02 可只跑指定人 (默认全部)

Env vars:
  GAUSSIAN_DIR  : 3DGS model dir (has point_cloud/iteration_*/point_cloud.ply)
  SOURCE_DIR    : COLMAP source (sparse/0/)
  RESULTS_DIR   : output root
  FACE_CENTER   : JSON with face 3D center (face_center_3d.py; 多人 JSON 含 persons)
  MASKS_DIR     : face masks folder (单人 {stem}.mask.png / 多人 {stem}.p{pid}.mask.png)
  PERSONS       : 多人模式下要跑的 pid 逗号列表 (默认全部)
  N_BINS        : azimuth bins (default 10)
  TARGET_COV    : target SAM2 coverage % to stop (default 40)
  MIN_DIST_RATIO: min distance ratio (default 0.3)
  PITCH_DEG     : pitch extrapolation degrees (default 10)
  CLOSEUP_SIZE  : 方形近景分辨率 (如 512; 0=默认用 COLMAP 相机 W/H)
  FACE_FILL     : CLOSEUP_SIZE 模式下人脸半径占半边长比例 (default 0.8)
  FACE_SPREAD_K : 人脸半径 = K × mean(spread) (default 2.5)
  ITERATION     : 3DGS iteration to load (default 30000)
  GS_DIR        : gaussian-splatting repo path
  DEVICE        : cuda
"""
import os
import re
import sys
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Config from env
GAUSSIAN_DIR = os.environ.get("GAUSSIAN_DIR", "")
SOURCE_DIR = os.environ.get("SOURCE_DIR", "")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "")
FACE_CENTER_JSON = os.environ.get("FACE_CENTER", "")
MASKS_DIR = os.environ.get("MASKS_DIR", "")
N_BINS = int(os.environ.get("N_BINS", "10"))
TARGET_COV = float(os.environ.get("TARGET_COV", "40"))
MIN_DIST_RATIO = float(os.environ.get("MIN_DIST_RATIO", "0.3"))
PITCH_DEG = float(os.environ.get("PITCH_DEG", "10"))
ITERATION = int(os.environ.get("ITERATION", "30000"))
GS_DIR = os.environ.get("GS_DIR", os.path.expanduser("~/repos/gaussian-splatting"))
DEVICE = os.environ.get("DEVICE", "cuda")

# 优化点 2 (grill-me A2): 近景渲染固定方形分辨率 (如 512)。
# 等距 fx=fy 从人脸 3D 点簇 spread 推导: 人脸半径 r=K×mean(spread),
# 期望人脸占半边长的 FACE_FILL → fx = FACE_FILL * (S/2) * d / r (d=物距, 逐视角)。
# 人脸像素尺寸跨视角恒定, 整张图即人脸特写 → 下游 WHOLE_IMAGE 整图增强。
CLOSEUP_SIZE = int(os.environ.get("CLOSEUP_SIZE", "0"))   # 0 = 用 COLMAP 相机 W/H (旧行为)
FACE_FILL = float(os.environ.get("FACE_FILL", "0.8"))
FACE_SPREAD_K = float(os.environ.get("FACE_SPREAD_K", "2.5"))

# 优化点 3 (grill-me 决策 C): 中间结果带标注 — sidecar _debug/ 目录, 不污染主输出。
# 渲染图/alpha 的标注副本 + annotations.json (与控制台日志同量级的信息)。
DEBUG_ANNOTATE = os.environ.get("DEBUG_ANNOTATE", "1") == "1"


def draw_annotations(pil_img, lines):
    """在图像副本左上角画标注条 (黑描边 + 黄字, 半透明底), 返回新图。"""
    import cv2
    arr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    # 半透明深色底条
    strip_h = 14 + 17 * len(lines)
    overlay = arr.copy()
    cv2.rectangle(overlay, (0, 0), (370, strip_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, arr, 0.55, 0, arr)
    y = 20
    for line in lines:
        cv2.putText(arr, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(arr, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (60, 220, 255), 1, cv2.LINE_AA)
        y += 17
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def quat_to_rotmat(qw, qx, qy, qz):
    """Quaternion to 3x3 rotation matrix (COLMAP convention: world->camera)."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
    ])


def look_at_rotmat(C, target, up):
    """COLMAP 约定下的 look-at 旋转矩阵 (world -> camera)。

    COLMAP 相机系: +X 右、+Y 下、+Z 前(朝场景), 为右手系,
    故 R 的三行依次是 [right; down; forward], 且 right × down = forward。

    up 传基视图自身的 up (即 -R[1,:]), 以沿用原有 roll, 不引入额外倾斜。
    """
    forward = np.asarray(target, dtype=np.float64) - np.asarray(C, dtype=np.float64)
    nf = np.linalg.norm(forward)
    if nf < 1e-9:
        raise ValueError("look_at: 相机位置与目标重合")
    forward = forward / nf

    up = np.asarray(up, dtype=np.float64)
    right = np.cross(forward, up)
    nr = np.linalg.norm(right)
    if nr < 1e-6:
        # forward 与 up 共线 (相机正对天顶/地底), 换一个参考轴
        alt = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(forward, alt))) > 0.99:
            alt = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, alt)
        nr = np.linalg.norm(right)
    right = right / nr
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=0)
    return R


def rotate_around_axis(v, axis, theta_deg):
    """Rodrigues 旋转: 向量 v 绕轴 axis 旋转 theta_deg 度 (右手定则)。"""
    k = np.asarray(axis, dtype=np.float64)
    k = k / (np.linalg.norm(k) + 1e-12)
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    return v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c)


def parse_colmap(source_dir):
    """Parse COLMAP cameras.txt + images.txt → list of view dicts.

    Handles layouts: sparse/0/cameras.txt, sparse/0_text/cameras.txt,
    or source_dir/cameras.txt.
    """
    sparse = Path(source_dir) / "sparse" / "0"
    cam_file = sparse / "cameras.txt"
    img_file = sparse / "images.txt"
    if not cam_file.exists():
        # Try sparse/0_text/ (BA output text format)
        sparse_text = Path(source_dir) / "sparse" / "0_text"
        cam_file = sparse_text / "cameras.txt"
        img_file = sparse_text / "images.txt"
    if not cam_file.exists():
        cam_file = Path(source_dir) / "cameras.txt"
        img_file = Path(source_dir) / "images.txt"

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
                fx = fy = params[0]; cx, cy = params[1:3]
            else:
                continue
            cameras[cid] = dict(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy)

    views = []
    with open(img_file) as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        qvec = np.array(list(map(float, parts[1:5])))
        tvec = np.array(list(map(float, parts[5:8])))
        cid = int(parts[8])
        name = parts[9]
        if cid not in cameras:
            continue
        cam = cameras[cid]
        R = quat_to_rotmat(*qvec)
        C = -R.T @ tvec  # camera position in world
        stem = os.path.splitext(name)[0]
        views.append(dict(
            stem=stem, name=name, R=R, T=tvec, C=C,
            **cam
        ))
    return views


def load_sam2_coverage(masks_dir, stems):
    """Load SAM2 mask coverage (%) per stem."""
    cov = {}
    for stem in stems:
        p = os.path.join(masks_dir, f"{stem}.mask.png")
        if os.path.exists(p):
            m = np.array(Image.open(p).convert("L"))
            cov[stem] = m.sum() / m.size * 100 / 255  # fraction * 100
    return cov


def load_coverage_multi(masks_dir, stems):
    """Load per-person coverage from {stem}.p{pid}.mask.png.

    Returns {pid_int: {stem: coverage%}}."""
    stem_set = set(stems)
    pat = re.compile(r"^(.+)\.p(\d+)\.mask\.png$")
    cov = {}
    for f in os.listdir(masks_dir):
        m = pat.match(f)
        if not m or m.group(1) not in stem_set:
            continue
        arr = np.array(Image.open(os.path.join(masks_dir, f)).convert("L"))
        cov.setdefault(int(m.group(2)), {})[m.group(1)] = arr.sum() / arr.size * 100 / 255
    return cov


def select_base_views(views, coverage, n_bins):
    """Bin by azimuth, pick highest-coverage per bin. Returns list of view dicts."""
    if not coverage:
        print("  ⚠️ no SAM2 coverage data, using first N_BINS views")
        return views[:n_bins]

    # Azimuth relative to scene center (mean of camera positions)
    center = np.mean([v["C"] for v in views], axis=0)
    for v in views:
        d = v["C"] - center
        v["azimuth"] = math.atan2(d[2], d[0])

    # Bin by azimuth [0, 2pi) → n_bins
    bin_size = 2 * math.pi / n_bins
    bins = [[] for _ in range(n_bins)]
    for v in views:
        az = v["azimuth"] % (2 * math.pi)
        bi = min(int(az / bin_size), n_bins - 1)
        bins[bi].append(v)

    selected = []
    for bi, bv in enumerate(bins):
        if not bv:
            continue
        # Pick highest coverage in bin
        scored = [(coverage.get(v["stem"], 0), v) for v in bv]
        scored.sort(reverse=True, key=lambda x: x[0])
        best_cov, best_v = scored[0]
        best_v["bin"] = bi
        best_v["coverage"] = best_cov
        selected.append(best_v)
        az_deg = math.degrees(best_v["azimuth"] % (2 * math.pi))
        print(f"  bin {bi}: az={az_deg:.0f}° {best_v['stem']} cov={best_cov:.1f}%")

    print(f"  selected {len(selected)} base views from {n_bins} bins")
    return selected


def compute_closeup_pose(view, face_center, target_cov, masks_dir, min_ratio):
    """沿 C→F 直线推进相机, 直到估计覆盖率 >= target。

    返回 (R, T, C, est_cov)。

    关键修正 (Fix C):
      推进后用 look-at 重算 R, 让相机严格瞄准 face_center。
      原实现直接沿用基视图的 R —— 基视图若本身没对准人脸, 推得越近偏得越多。
    """
    orig_C = view["C"].copy()
    orig_dist = np.linalg.norm(face_center - orig_C)

    # 沿 C→F 直线推进 (与视线无关, 保证人脸始终在光轴方向上)
    direction = (face_center - orig_C)
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    best_C = orig_C.copy()
    best_cov = view.get("coverage", 0)

    # 覆盖率近似: 面积 ~ 1/距离^2
    for ratio in np.arange(0.95, min_ratio - 0.05, -0.05):
        new_dist = orig_dist * ratio
        best_C = face_center - direction * new_dist
        best_cov = min(view.get("coverage", 0) * (orig_dist / new_dist) ** 2, 100)
        if best_cov >= target_cov:
            break

    # Fix C: look-at 重算朝向, up 沿用基视图以保持 roll
    up = -view["R"][1, :]
    R_new = look_at_rotmat(best_C, face_center, up)
    T_new = -R_new @ best_C
    return R_new, T_new, best_C, best_cov


def apply_pitch(view, face_center, pitch_deg):
    """绕「过人脸中心、平行于相机右向」的轴公转 pitch 度, 再 look-at 人脸。

    修正 (Fix C):
      原实现 R_new = Rx @ R —— 绕世界 X 轴、且是相机系后乘;
      而 C 是绕世界轴旋转的。位置转了、朝向没跟着转, 两者不同步,
      实测 pitch 仅 10° 却造成 25° 的视线偏差, 超过半 FOV(27°), 人脸被甩出画面。

      现改为: 只用 Rodrigues 绕相机自身右向轴公转「位置」,
      再用 look-at 重算「朝向」, 保证无论怎么转, 人脸永远在画面中心。
    """
    R = view["R"]
    C = view["C"]
    right_axis = R[0, :]   # 相机右向 (世界系) —— 公转轴
    up = -R[1, :]          # 相机上向 (世界系) —— 保持 roll

    C_new = face_center + rotate_around_axis(C - face_center, right_axis, pitch_deg)
    R_new = look_at_rotmat(C_new, face_center, up)
    T_new = -R_new @ C_new
    return R_new, T_new, C_new


def render_views(closeup_views, out_dir, alpha_dir, debug_dir=""):
    """3DGS render closeup views. closeup_views: list of {R, T, W, H, fx, fy, name}."""
    import torch
    sys.path.insert(0, GS_DIR)
    from scene import GaussianModel
    from scene.cameras import Camera
    from gaussian_renderer import render
    from argparse import Namespace

    ply_path = os.path.join(GAUSSIAN_DIR, "point_cloud", f"iteration_{ITERATION}", "point_cloud.ply")
    if not os.path.isfile(ply_path):
        sys.exit(f"PLY not found: {ply_path}")

    sh_degree = 3
    cfg_path = os.path.join(GAUSSIAN_DIR, "cfg_args")
    if os.path.isfile(cfg_path):
        import re
        with open(cfg_path) as f:
            m = re.search(r"sh_degree\s*=\s*(\d+)", f.read())
            if m:
                sh_degree = int(m.group(1))

    print(f"  loading 3DGS: {ply_path} (sh={sh_degree})")
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path)
    gaussians.active_sh_degree = sh_degree

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                     antialiasing=False, debug=False)
    bg_black = torch.zeros(3, device=DEVICE)
    bg_white = torch.ones(3, device=DEVICE)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(alpha_dir).mkdir(parents=True, exist_ok=True)
    if DEBUG_ANNOTATE and debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

    poses = []
    for i, cv in enumerate(closeup_views):
        R = cv["R"]; T = cv["T"]; W = cv["W"]; H = cv["H"]
        fx = cv["fx"]; fy = cv["fy"]
        FoVx = 2 * math.atan(W / (2 * fx))
        FoVy = 2 * math.atan(H / (2 * fy))

        dummy = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
        # 注意: gaussian-splatting 的 getWorld2View2 内部会 R.transpose(),
        # Camera 期望 camera-to-world 的 R (见 dataset_readers.py:85 的 qvec2rotmat(...).T)。
        # 我们存的 R 是 COLMAP 约定 world-to-camera, 传 R.T。
        cam = Camera(resolution=(W, H), colmap_id=0, R=R.T, T=T,
                     FoVx=FoVx, FoVy=FoVy, depth_params=None,
                     image=dummy, invdepthmap=None,
                     image_name=cv["name"], uid=i, data_device=DEVICE)

        with torch.no_grad():
            pkg_b = render(cam, gaussians, pipe, bg_black)
            pkg_w = render(cam, gaussians, pipe, bg_white)

        rgb = pkg_b["render"].clamp(0, 1)
        white = pkg_w["render"].clamp(0, 1)
        alpha = 1 - (white - rgb).mean(dim=0, keepdim=True).clamp(0, 1)

        rgb_path = os.path.join(out_dir, cv["name"])
        alpha_path = os.path.join(alpha_dir, cv["name"])
        Image.fromarray((rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(rgb_path)
        Image.fromarray((alpha[0].cpu().numpy() * 255).astype(np.uint8)).save(alpha_path)

        avg_a = float(alpha.mean())
        print(f"  [{i+1}/{len(closeup_views)}] {cv['name']} alpha={avg_a:.3f} "
              f"{'⚠️' if avg_a < 0.3 else '✅'}")

        # 优化点 3: sidecar _debug/ 标注副本 (渲染图 + alpha 色化图)
        if DEBUG_ANNOTATE and debug_dir:
            base_stem = cv.get("base_stem", "")
            lines = [
                f"{cv['name']} | {os.path.basename(debug_dir)}",
                f"base={base_stem} pitch={cv.get('pitch', 0):+.0f}deg",
                f"est_cov={cv.get('est_cov', 0):.1f}% "
                f"dist={cv.get('dist_to_face', -1):.3f} "
                f"offset={cv.get('face_offset_px', -1):.1f}px",
                f"size={W}x{H} f={fx:.0f}"
                + (f" (iso CLOSEUP_SIZE={CLOSEUP_SIZE})" if CLOSEUP_SIZE > 0 else ""),
                f"alpha_mean={avg_a:.3f}",
            ]
            rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            draw_annotations(Image.fromarray(rgb_np), lines).save(
                os.path.join(debug_dir, cv["name"]))
            alpha_rgb = np.stack([alpha[0].cpu().numpy() * 255] * 3, axis=-1).astype(np.uint8)
            draw_annotations(Image.fromarray(alpha_rgb), lines).save(
                os.path.join(debug_dir, cv["name"].replace(".png", ".alpha.png")))

        poses.append(dict(
            name=cv["name"],
            R=cv["R"].tolist(),
            T=cv["T"].tolist(),
            C=cv["C"].tolist(),
            W=W, H=H, fx=fx, fy=fy,
            cx=cv["cx"], cy=cv["cy"],
            base_view=cv.get("base_stem", ""),
            pitch=cv.get("pitch", 0),
            est_coverage=cv.get("est_cov", 0),
            face_offset_px=cv.get("face_offset_px", -1.0),
            dist_to_face=cv.get("dist_to_face", -1.0),
        ))

    del gaussians
    torch.cuda.empty_cache()
    return poses


def run_pipeline(views, face_center, coverage, tag="", prefix="", r_face=0.0):
    """单人流: 选基视角 → 推进近景 → pitch 外插 → sanity → 3DGS 渲染 → poses。

    tag: 输出目录后缀 (单人 "" / 多人 "_p00"); prefix: 日志前缀。
    r_face: 人脸 3D 半径估计 (FACE_SPREAD_K × mean(spread)), CLOSEUP_SIZE 模式必需。
    """
    # 1. Select base views (azimuth binning)
    print(f"\n{prefix}🎯 selecting {N_BINS} base views by azimuth binning...")
    base_views = select_base_views(views, coverage, N_BINS)

    # 2. For each base view: closeup + pitch extrapolation
    print(f"\n{prefix}🔭 generating closeup + pitch views...")
    closeup_views = []
    for bv in base_views:
        # Closeup: move toward face center
        R, T, C, cov = compute_closeup_pose(bv, face_center, TARGET_COV, MASKS_DIR, MIN_DIST_RATIO)
        closeup = dict(bv)
        closeup["R"] = R; closeup["T"] = T; closeup["C"] = C
        closeup["est_cov"] = cov
        closeup["base_stem"] = bv["stem"]
        closeup["pitch"] = 0

        # Pitch +10
        R_p, T_p, C_p = apply_pitch(closeup, face_center, PITCH_DEG)
        vp = dict(closeup)
        vp["R"] = R_p; vp["T"] = T_p; vp["C"] = C_p
        vp["pitch"] = +PITCH_DEG
        vp["name"] = f"closeup_{len(closeup_views)+1:04d}.png"
        closeup_views.append(vp)

        # Pitch -10
        R_m, T_m, C_m = apply_pitch(closeup, face_center, -PITCH_DEG)
        vm = dict(closeup)
        vm["R"] = R_m; vm["T"] = T_m; vm["C"] = C_m
        vm["pitch"] = -PITCH_DEG
        vm["name"] = f"closeup_{len(closeup_views)+1:04d}.png"
        closeup_views.append(vm)

    print(f"{prefix}  {len(closeup_views)} closeup views generated")

    # 2b. CLOSEUP_SIZE 模式: 覆写为方形分辨率 + 等距焦距 (人脸占 FACE_FILL 半边长)
    if CLOSEUP_SIZE > 0:
        if r_face <= 0:
            sys.exit(f"{prefix}CLOSEUP_SIZE={CLOSEUP_SIZE} 需要 FACE_CENTER JSON 的 spread "
                     f"(r_face<=0; face_center_3d 输出含 spread 字段)")
        print(f"{prefix}📐 CLOSEUP_SIZE={CLOSEUP_SIZE}: r_face={r_face:.4f}, fill={FACE_FILL}")
        for cv in closeup_views:
            d = float(np.linalg.norm(face_center - cv["C"]))
            f = FACE_FILL * (CLOSEUP_SIZE / 2) * d / r_face
            cv["W"] = cv["H"] = CLOSEUP_SIZE
            cv["cx"] = cv["cy"] = CLOSEUP_SIZE / 2
            cv["fx"] = cv["fy"] = f
            cv["iso_focal"] = round(f, 1)

    # 3. Sanity check: face_center 反投回每个近景视角
    print(f"\n{prefix}🔍 sanity check: face_center 反投到近景视角...")
    n_bad = 0
    for cv in closeup_views:
        Pc = cv["R"] @ face_center + cv["T"]
        dist = float(np.linalg.norm(face_center - cv["C"]))
        if Pc[2] <= 0:
            print(f"{prefix}  ❌ {cv['name']}: F 在相机背后 (z={Pc[2]:.3f})")
            cv["face_offset_px"] = -1.0
            n_bad += 1
            continue
        u = cv["fx"] * Pc[0] / Pc[2] + cv["cx"]
        v = cv["fy"] * Pc[1] / Pc[2] + cv["cy"]
        off_px = math.hypot(u - cv["cx"], v - cv["cy"])
        off_pct = off_px / min(cv["W"], cv["H"]) * 100
        in_frame = (0 <= u < cv["W"]) and (0 <= v < cv["H"])
        flag = "✅" if off_pct < 5 else ("⚠️" if in_frame else "❌")
        if flag != "✅":
            n_bad += 1
        cv["face_offset_px"] = round(float(off_px), 1)
        cv["dist_to_face"] = dist
        half_fov = math.degrees(math.atan(min(cv["W"], cv["H"]) / (2 * min(cv["fx"], cv["fy"]))))
        print(f"{prefix}  {flag} {cv['name']}: F→({u:6.0f},{v:6.0f}) 偏离中心 {off_px:5.1f}px "
              f"({off_pct:4.1f}%) 物距 {dist:.3f} 半FOV {half_fov:.0f}°")
    if n_bad:
        print(f"{prefix}  ⚠️ {n_bad}/{len(closeup_views)} 个视角未对准人脸")
    else:
        print(f"{prefix}  ✅ 全部 {len(closeup_views)} 个视角都对准人脸 (偏移 <5%)")

    # 4. 3DGS render
    print(f"\n{prefix}🖼️ rendering closeup views...")
    renders_dir = os.path.join(RESULTS_DIR, f"06c_closeup_renders{tag}")
    alpha_dir = os.path.join(RESULTS_DIR, f"06c_closeup_alpha{tag}")
    debug_dir = os.path.join(RESULTS_DIR, f"06c_closeup_debug{tag}")
    poses = render_views(closeup_views, renders_dir, alpha_dir, debug_dir)

    # 5. Save poses (debug 目录同步一份 annotations.json, 与图像标注同源)
    poses_path = os.path.join(RESULTS_DIR, f"06c_closeup_poses{tag}.json")
    with open(poses_path, "w") as f:
        json.dump(poses, f, indent=2)
    if DEBUG_ANNOTATE and debug_dir:
        with open(os.path.join(debug_dir, "annotations.json"), "w") as f:
            json.dump({"results_dir": RESULTS_DIR, "tag": tag,
                       "closeup_size": CLOSEUP_SIZE, "r_face": r_face,
                       "views": poses}, f, indent=2, ensure_ascii=False)
    print(f"\n{prefix}💾 poses: {poses_path}")
    print(f"{prefix}🖼️ renders: {renders_dir}")
    print(f"{prefix}✅ done: {len(poses)} closeup views rendered")


def r_face_from_spread(spread):
    """人脸 3D 半径估计 = FACE_SPREAD_K × mean(spread_xyz)。"""
    return FACE_SPREAD_K * float(np.mean(spread))


def main():
    print("🧑 [render_closeup] face closeup + pitch extrapolation")
    print(f"  📂 3DGS: {GAUSSIAN_DIR} (iter={ITERATION})")
    print(f"  📂 source: {SOURCE_DIR}")
    print(f"  📂 masks: {MASKS_DIR}")
    print(f"  📐 bins={N_BINS}, target_cov={TARGET_COV}%, min_dist={MIN_DIST_RATIO}, pitch=±{PITCH_DEG}°")
    print("")

    # 1. Load face center JSON
    with open(FACE_CENTER_JSON) as f:
        fc = json.load(f)

    # 2. Parse COLMAP cameras (一次, 多人共用)
    print("\n📷 parsing COLMAP cameras...")
    views = parse_colmap(SOURCE_DIR)
    stems = [v["stem"] for v in views]

    # 3. 多人 / 单人分派
    if "persons" in fc:
        pids_avail = sorted(int(k) for k in fc["persons"])
        sel = [s.strip() for s in os.environ.get("PERSONS", "").split(",") if s.strip()]
        if sel:
            pids = [int(s.replace("p", "")) for s in sel]
            missing = [p for p in pids if p not in pids_avail]
            if missing:
                sys.exit(f"PERSONS 里的 {missing} 不在 FACE_CENTER JSON persons 里 ({pids_avail})")
        else:
            pids = pids_avail
        print(f"\n👥 multi-person mode: {len(pids)} person(s) to run: {[f'p{p:02d}' for p in pids]}")
        cov_multi = load_coverage_multi(MASKS_DIR, stems)
        for pid in pids:
            if pid not in cov_multi:
                print(f"\n⚠️ p{pid:02d}: MASKS_DIR 里没有 p{pid:02d} 的 mask, 跳过")
                continue
            print(f"\n{'='*60}\n👤 person p{pid:02d}")
            pinfo = fc["persons"][str(pid)]
            center = np.array(pinfo["center"])
            r_face = r_face_from_spread(pinfo["spread"]) if "spread" in pinfo else 0.0
            run_pipeline(views, center, cov_multi[pid],
                         tag=f"_p{pid:02d}", prefix=f"[p{pid:02d}] ", r_face=r_face)
    else:
        face_center = np.array(fc["center"])
        print(f"  face center: {face_center}")
        print(f"\n🎭 loading SAM2 coverage...")
        coverage = load_sam2_coverage(MASKS_DIR, stems)
        print(f"  SAM2 coverage for {len(coverage)} views")
        r_face = r_face_from_spread(fc["spread"]) if "spread" in fc else 0.0
        run_pipeline(views, face_center, coverage, r_face=r_face)


if __name__ == "__main__":
    main()
