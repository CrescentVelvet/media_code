#!/usr/bin/env python3
"""三方法单图输入对比视频（FlashWorld / Gen3R / Scene Splatter），佳佳抱枕样例。

设计（2026-11-10 第三轮，经多轮实证定案）
----------------------------------------
**每方法在「自己的世界」里绕「自己的主体」转**——不做跨方法相似配准。理由（实证链）：
  1. FW PLY 在 cam0 相机系（p_pub = R0·p + t0，与 03_render_ply.py 的
     viewmat = FLIP@inv(c2w_i)@c2w_0 严格互逆，数值已验）。
  2. 统一内参不能用 FW 官方 K（fx=203.4 cx=172.2 cy=-112 @704×480 是 app
     center-crop 产物，hfov 120°）——正常焦距 fx≈0.875·W。
  3. FW「注视点」是视线交点（原点），但主体在 [0.66,-1.07,0.59]（低 1.07m），
     官方 K 的 cy=-112 正是补偿这个的——跨方法对齐到原点的方案全部 therefore 失败。
  4. Gen3R 单图重建的有效观察壳极薄：自家 49 相机只占 ~49° 方位、固定仰角、
     r∈[0.95,1.1]；壳外 10° 即多深度重影（单图深度歧义）、壳上视角即干净——
     任何合成环都必须贴各方法自己的相机剖面。
  5. G3R 世界是 **y 朝下**（cam0 反投影实证：图像 up=[0,-1,0]）；
     up 向量错 = 渲染横滚 90°（踩过）。

弧参数化（三面板同步）：
  · 每方法取自己的主体中心 S（投影中部+opacity² 加权质心）、世界 up、
    相机方位覆盖（最小包容弧）、r/h 观测范围。
  · θ=0 对齐各方法观测弧的中点；统一 span = min(170°, 各方法覆盖−6°)，下限 40°。
  · 眼位 = S + panel_r·(水平环) + panel_h·up，look_at S；panel_r 取
    clamp(2·主体半径, r_obs_min·0.98, r_obs_max·1.04)（FW 大致 1.8，G3R ≈1.0）。

产出 <OUT>/three_way_arc.mp4 + 首/中/末帧抽检 PNG。
环境变量：OUT / NFRAMES / FPS / CELLW / FW_OP_MIN / FW_PLY / G3R_PLY / G3R_CAMS /
          SS_PLY / SS_TRAJ / INPUT_IMG / ONLY
"""
import gzip
import json
import os
import pickle
import re
import subprocess
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from gsplat import rasterization

HOME = "/home/velvet"
O = HOME + "/output"

FW = O + "/flashworld_results/03_gaussians/jiajia/gaussians.ply"
FW_JSON = O + "/flashworld_results/02_input_json/jiajia.json"
INPUT_IMG = os.environ.get(
    "INPUT_IMG", O + "/flashworld_results/_input/jiajia.png")

G3R_PLY = os.environ.get("G3R_PLY", "")
G3R_CAMS = os.environ.get("G3R_CAMS", "")
G3R_COLMAP = os.environ.get("G3R_COLMAP", "")
SS_PLY = os.environ.get("SS_PLY", "")
SS_TRAJ = os.environ.get("SS_TRAJ", "")

OUT = os.environ.get("OUT", O + "/compare_4methods_video")
NFRAMES = int(os.environ.get("NFRAMES", "144"))
FPS = int(os.environ.get("FPS", "24"))
CELLW = int(os.environ.get("CELLW", "396"))
BG = os.environ.get("BG", "white")
FXN = float(os.environ.get("FXN", "0.875"))     # 统一焦距倍率（≈59° hfov）

FONT_R = "/mnt/c/Windows/Fonts/msyh.ttc"
FONT_B = "/mnt/c/Windows/Fonts/msyhbd.ttc"

CASES = [
    dict(name="FlashWorld", meta="单图→视频→逐帧反投影", color=(91, 127, 181),
         ply=FW, cams=("fwjson", FW_JSON),
         # FW PLY 在 cameras[0] 相机系：p_pub = R0·p + t0（03_render_ply.py L97-128 实证，
         # 数值 our_w2c@c2w_0 == FLIP 已验）。剪枝去单图多深度重影副本。
         prewarp="fw_c2w0",
         op_min=float(os.environ.get("FW_OP_MIN", "0.05"))),
    dict(name="Gen3R", meta="单图→点云→COLMAP→3DGS", color=(168, 131, 79),
         ply=G3R_PLY,
         # 位姿必须取 colmap_scene（COLMAP 精化后的世界 = PLY 世界）；
         # 3dgs_7000/cameras.json 在另一个世界，只有 cam0 碰巧重合（2026-11-10 实证）。
         cams=("colmap", G3R_COLMAP) if G3R_COLMAP else ("json", G3R_CAMS)),
    dict(name="Scene Splatter", meta="flash3d→ViewCrafter 迭代", color=(15, 110, 110),
         ply=SS_PLY, cams=("pkl", SS_TRAJ),
         # ViewCrafter 优化期 Adam 自由漂移 + cameras_extent=10，云带远程垃圾
         #（实测 x∈[-4.9,5.3] 而主体 r≈0.9）：轨道路径上有低透明 floater 糊镜头。
         op_min=float(os.environ.get("SS_OP_MIN", "0.1")),
         shell=float(os.environ.get("SS_SHELL", "3.5")),
         panel_r_abs=float(os.environ.get("SS_PANEL_R", "1.8"))),
]
for c in CASES:
    if not c["ply"] or not os.path.exists(c["ply"]):
        sys.exit("❌ %s 产物不存在: %s（先跑对应推理）" % (c["name"], c["ply"]))
    if not os.path.exists(c["cams"][1]):
        sys.exit("❌ %s 相机不存在: %s" % (c["name"], c["cams"][1]))
ONLY = os.environ.get("ONLY", "").strip()
if ONLY:
    keys = [k.strip().lower() for k in ONLY.split(",") if k.strip()]
    CASES = [c for c in CASES if any(k in c["name"].lower() for k in keys)]


def log(*a):
    print(*a, flush=True)


def quat_to_R_wxyz(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# ── PLY 读取（read_ply / _asvec）────────────────────────────────
def read_ply(path):
    with open(path, "rb") as f:
        head = b""
        while b"end_header" not in head:
            head += f.read(1)
        tail = f.read(1)
        while tail not in (b"\n", b""):
            head += tail
            tail = f.read(1)
        head += tail
        text = head.decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", text).group(1))
        names = re.findall(r"^property float (\w+)$", text, re.M)
        dt = np.dtype([(nm, "<f4") for nm in names])
        f.seek(len(head))
        raw = np.fromfile(f, dtype=dt, count=n)
    means = np.stack([raw["x"], raw["y"], raw["z"]], 1).astype(np.float32)
    scales = np.exp(np.stack([raw["scale_%d" % i] for i in range(3)], 1)).astype(np.float32)
    opac = (1.0 / (1.0 + np.exp(-raw["opacity"].astype(np.float64)))).astype(np.float32)
    quats = np.stack([raw["rot_%d" % i] for i in range(4)], 1).astype(np.float32)
    # SH：两家（FW 全写进 f_dc_0..26 / 官方 f_dc+f_rest）都是**系数主序**
    # （连续 3 个 float = 一个系数的 RGB）——03_render_ply 的 reshape(n,-1,3)
    # 与官方 3DGS 均如此。直接 (N,K,3) reshape，勿 transpose。
    dc = sorted([nm for nm in names if nm.startswith("f_dc_")],
                key=lambda s: int(s.rsplit("_", 1)[1]))
    rest = sorted([nm for nm in names if nm.startswith("f_rest_")],
                  key=lambda s: int(s.rsplit("_", 1)[1]))
    cols = [raw[nm] for nm in dc + rest]
    k = len(cols) // 3
    sh = np.stack(cols, 1).reshape(len(means), k, 3).astype(np.float32)
    deg = int(round(k ** 0.5)) - 1
    return dict(means=means, scales=scales, opac=opac, quats=quats, sh=sh,
                n=int(n), deg=deg)


def _asvec(v):
    if isinstance(v, str):
        return np.array(json.loads(v.replace("[3]", "").strip()), dtype=np.float64)
    return np.array(v, dtype=np.float64)


def read_cams_c2w(kind, path):
    """统一返回 (c2ws (V,4,4), K (3,3) 或 None, W, H)。

    c2ws = 各相机在世界里的位姿；up 约定：OpenCV 系（json/pkl）世界 up = −cam_y，
    OpenGL 系（fwjson）世界 up = +cam_y。"""
    if kind == "fwjson":
        j = json.load(open(path))
        c2ws = []
        for c in j["cameras"]:
            M = np.eye(4)
            M[:3, :3] = quat_to_R_wxyz(c["quaternion"])
            M[:3, 3] = np.array(c["position"], dtype=np.float64)
            c2ws.append(M)
        # 官方渲染内参（app center-crop 缩放版；仅用于主体估计）
        _, Hj, Wj = j["resolution"]
        c0 = j["cameras"][0]
        fx, fy, cx, cy = c0["fx"], c0["fy"], c0["cx"], c0["cy"]
        if os.path.exists(INPUT_IMG):
            iw, ih = Image.open(INPUT_IMG).size
            sc = max(Hj / ih, Wj / iw)
            fx, fy = fx * sc, fy * sc
            cx = (cx - (iw - int(Wj / sc)) // 2) * sc
            cy = (cy - (ih - int(Hj / sc)) // 2) * sc
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return np.stack(c2ws), K, int(Wj), int(Hj)
    if kind == "json":
        j = json.load(open(path))
        c2ws = []
        for it in j:
            M = np.eye(4)
            M[:3, :3] = _asvec(it["rotation"]).reshape(3, 3)
            M[:3, 3] = _asvec(it["position"])
            c2ws.append(M)
        c0 = j[0]
        K = np.array([[c0["fx"], 0, c0["width"] / 2.0],
                      [0, c0["fy"], c0["height"] / 2.0], [0, 0, 1]], dtype=np.float64)
        return np.stack(c2ws), K, int(c0["width"]), int(c0["height"])
    if kind == "pkl":
        with gzip.open(path, "rb") as f:
            d0 = pickle.load(f)
        e = np.asarray(d0["poses"], dtype=np.float64)      # (V,3,4) w2c OpenCV
        c2ws = []
        for i in range(len(e)):
            M = np.eye(4)
            Rw, t = e[i, :3, :3], e[i, :3, 3]
            M[:3, :3] = Rw.T
            M[:3, 3] = -Rw.T @ t
            c2ws.append(M)
        fx, fy, cx, cy = np.asarray(d0["intrinsics"], dtype=np.float64).flat[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        # SS 内部世界 = frame0 相机空间（2026-11-10 实证：flash3d 高斯由 inv_K 反投影
        # 生成、render_images 对 frame0 用 T=identity；ViewCrafter 优化同世界）。
        # pkl 位姿经 M = w2c(pose0) 搬入该世界后，cam0 = 单位位姿（输入视角 = 原点朝 +z）。
        M0 = np.linalg.inv(c2ws[0])
        c2ws = np.stack([M0 @ c for c in c2ws])
        return np.stack(c2ws), K, int(round(cx * 2)), int(round(cy * 2))
    if kind == "colmap":
        # COLMAP 文本场景（images.txt 的 qvec/tvec 是 w2c，世界系 = PLY 世界）。
        # Gen3R 实证（2026-11-10）：3dgs_7000/cameras.json 在**另一个世界**（cam0 恰在原点
        # 所以只有近 cam0 视角碰巧能看），只有 colmap_scene 的位姿与 PLY 同世界。
        w2cs, camids = {}, {}
        for line in open(os.path.join(path, "sparse/0/images.txt")):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 10:
                continue
            q = np.array([float(v) for v in p[1:5]], dtype=np.float64)
            t = np.array([float(v) for v in p[5:8]], dtype=np.float64)
            M = np.eye(4)
            M[:3, :3] = quat_to_R_wxyz(q)
            M[:3, 3] = t
            w2cs[int(p[0])] = M
            camids[int(p[0])] = int(p[8])
        kinfo = {}
        for line in open(os.path.join(path, "sparse/0/cameras.txt")):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            kinfo[int(p[0])] = (float(p[4]), float(p[5]), float(p[6]), float(p[7]),
                                int(p[2]), int(p[3]))
        ids = sorted(w2cs)
        c2ws = np.stack([np.linalg.inv(w2cs[i]) for i in ids])
        fx, fy, cx, cy, W, H = kinfo[camids[ids[0]]]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return c2ws, K, W, H
    raise SystemExit("未知相机格式 " + kind)


def look_at(eye, target, up):
    z = target - eye
    z = z / (np.linalg.norm(z) + 1e-12)
    if abs(float(z @ up)) > 0.98:
        up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
    x = np.cross(z, up)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 1)
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ eye
    return w2c


def quat_mul(a, b):
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], 1)


def warp(g, s, R, t):
    from scipy.spatial.transform import Rotation as Rot
    g["means"] = (s * (g["means"].astype(np.float64) @ R.T) + t).astype(np.float32)
    g["scales"] = (g["scales"].astype(np.float64) * s).astype(np.float32)
    qR = Rot.from_matrix(R).as_quat()
    qR = np.array([qR[3], qR[0], qR[1], qR[2]], dtype=np.float64)
    qR = np.repeat(qR[None, :], len(g["quats"]), 0)
    q = quat_mul(qR, g["quats"].astype(np.float64))
    g["quats"] = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
    return g


def subject_stats(means, opac, w2c, K, W, H):
    """主体中心 S（投影中部 + opacity² 加权）与主体半径（P90 距离）。

    返回 (S (3,), r_subj float)。w2c 用该方法自己的第一个相机 pose + 自己的 K。"""
    pc = means.astype(np.float64) @ w2c[:3, :3].T + w2c[:3, 3]
    ok = pc[:, 2] > 0.2
    uv = pc[:, :2] / pc[:, 2:3] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    central = ok & (uv[:, 0] > W * 0.3) & (uv[:, 0] < W * 0.7) \
        & (uv[:, 1] > H * 0.3) & (uv[:, 1] < H * 0.9)
    P = means[central].astype(np.float64)
    w = opac[central].astype(np.float64) ** 2
    S = (w[:, None] * P).sum(0) / w.sum()
    r_subj = float(np.percentile(np.linalg.norm(P - S, axis=1), 90))
    return S, r_subj


def arc_coverage(az):
    """方位角的最小包容弧（对聚簇正确，360−最大缺口 会高估）。返回 (start, cover)。"""
    az = np.sort(np.asarray(az, dtype=np.float64))
    n = len(az)
    best = None
    for i in range(n):
        d = (az - az[i]) % (2 * np.pi)
        span = float(d.max())
        if best is None or span < best[1]:
            best = (float(az[i]), span)
    return best[0], best[1]


def ffmpeg_exe():
    from shutil import which
    p = which("ffmpeg")
    if p:
        return p
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"
    CELLH = int(round(CELLW * 270.0 / 396.0))
    K = np.array([[FXN * CELLW, 0, (CELLW - 1) / 2.0],
                  [0, FXN * CELLW, (CELLH - 1) / 2.0], [0, 0, 1]], dtype=np.float64)
    Kt = torch.from_numpy(K).float().to(dev)[None]
    bg = torch.ones(3, device=dev) if BG == "white" else torch.zeros(3, device=dev)
    log("🖼  单格 %d×%d ｜ fx=%.0f（统一内参，hfov≈%.0f°）"
        % (CELLW, CELLH, K[0, 0], np.degrees(2 * np.arctan(0.5 / FXN))))

    # ── 阶段 1：逐方法准备（自己的世界、自己的主体、自己的相机剖面）──
    prepared = []
    for c in CASES:
        g = read_ply(c["ply"])
        c2ws, K_own, W_own, H_own = read_cams_c2w(*c["cams"])
        kind = c["cams"][0]
        # 世界 up：OpenCV 系（json/pkl）= −cam_y；OpenGL 系（fwjson）= +cam_y
        ydir = np.array([0.0, 1.0, 0.0]) if kind == "fwjson" else np.array([0.0, -1.0, 0.0])
        up = c2ws[0][:3, :3] @ ydir
        up = up / (np.linalg.norm(up) + 1e-12)
        # FW：PLY 从 cam0 相机系搬到公共系（与官方渲染严格互逆）
        if c.get("prewarp") == "fw_c2w0":
            R0, t0 = c2ws[0][:3, :3], c2ws[0][:3, 3]
            g = warp(g, 1.0, R0, t0)
        if c.get("op_min"):
            keep = g["opac"] >= c["op_min"]
            deg_keep = g["deg"]
            g = {k: v[keep] for k, v in g.items()
                 if k not in ("n", "deg") and hasattr(v, "__len__")}
            g["n"] = int(keep.sum())
            g["deg"] = deg_keep
        # 主体统计：用自己的 K + 自己 cam0 的 OpenCV pose
        R_c, C0 = c2ws[0][:3, :3], c2ws[0][:3, 3]
        w2c0 = np.eye(4)
        if kind == "fwjson":
            w2c0[:3, :3] = np.diag([1.0, -1.0, -1.0]) @ R_c.T
            w2c0[:3, 3] = -w2c0[:3, :3] @ C0
        else:
            w2c0[:3, :3] = R_c.T
            w2c0[:3, 3] = -R_c.T @ C0
        S, r_subj = subject_stats(g["means"], g["opac"], w2c0, K_own, W_own, H_own)
        if c.get("shell"):
            # 主体壳裁剪：丢掉离 S 超过 shell 的漂移垃圾（ViewCrafter 优化残留），
            # 避免轨道经过时 floater 贴脸糊镜头。
            keep = np.linalg.norm(g["means"] - S, axis=1) <= c["shell"]
            deg_keep = g["deg"]
            g = {k: v[keep] for k, v in g.items()
                 if k not in ("n", "deg") and hasattr(v, "__len__")}
            g["n"] = int(keep.sum())
            g["deg"] = deg_keep
            log("      🧹 %s 壳裁剪: %d -> %d 点" % (c["name"], len(keep), g["n"]))
        # 相机剖面：水平基、方位覆盖、r/h 观测范围
        z_hint = np.array([0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0, 0])
        xh = np.cross(up, z_hint)
        xh /= np.linalg.norm(xh) + 1e-12
        zh = np.cross(xh, up)
        rel = c2ws[:, :3, 3] - S
        rh = rel - np.outer(rel @ up, up)
        r_obs = np.linalg.norm(rh, axis=1)
        h_obs = rel @ up
        az = np.arctan2(rh @ xh, rh @ zh)
        az0, cov = arc_coverage(az)
        az_c0 = float(az[0])   # 输入视角（视频首帧）的方位，弧锚点
        panel_r = min(max(2.0 * r_subj, float(r_obs.min()) * 0.98),
                      float(r_obs.max()) * 1.04)
        if c.get("panel_r_abs"):
            # 绝对半径覆盖（SS：主体偏心 + 大高斯，弧扫近端贴脸糊，自家相机最远在 2.4m 外）
            panel_r = c["panel_r_abs"]
        panel_h = float(np.median(h_obs))
        log("📷 %-13s 主体 S=%s r_subj=%.2f ｜ 观测弧 %.1f°（cam0 方位 %.1f°）r∈[%.2f,%.2f] h=%.2f → panel r=%.2f"
            % (c["name"], np.round(S, 2), r_subj, np.degrees(cov), np.degrees(az_c0),
               r_obs.min(), r_obs.max(), panel_h, panel_r))
        prepared.append(dict(c=c, g=g, S=S, up=up, xh=xh, zh=zh, az0=az0, cov=cov,
                             az_c0=az_c0, panel_r=panel_r, panel_h=panel_h))

    span = min([170.0] + [np.degrees(p["cov"]) for p in prepared]) - 6.0
    span = max(span, 20.0)
    # 弧从「输入视角附近」扫入观测弧（cam0 常在弧端点：G3R 实测弧心在 cam0 之外 35°，
    # 曾致弧中点落在观察壳外全糊）。起点夹在观测弧内、整段不出界。
    th_start = []
    for p in prepared:
        # wrap 安全：先把 cam0 方位归一到弧内相对角，再钳制，避免 ±180° 缠绕时
        # 数值比较把弧起点推出观测弧（G3R 实测：az_c0=-177.4° 被误判在弧外）
        rel = (p["az_c0"] - p["az0"]) % (2 * np.pi)
        lo, hi = np.radians(3.0), p["cov"] - np.radians(3.0 + span)
        th_start.append(p["az0"] + min(max(rel, lo), hi if hi > lo else lo))
    log("🎬 统一 span=%.1f°（受最窄观测弧限制）｜ %d 帧 / %d fps" % (span, NFRAMES, FPS))

    # ── 阶段 2：逐方法渲染（各自世界，同步方位参数）──
    frames = []
    off = np.linspace(-span / 2, span / 2, NFRAMES)
    for pi, p in enumerate(prepared):
        g, S, up = p["g"], p["S"], p["up"]
        G = {k: torch.from_numpy(g[k]).to(dev)
             for k in ("means", "quats", "scales", "opac", "sh")}
        buf = []
        amed = 0.0
        for i in range(NFRAMES):
            az = th_start[pi] + np.radians(off[i] + span / 2.0)
            eye = (S + p["panel_r"] * (np.cos(az) * p["zh"] + np.sin(az) * p["xh"])
                   + p["panel_h"] * up)
            w2c = torch.from_numpy(look_at(eye, S, up)).float().to(dev)[None]
            with torch.no_grad():
                img, alpha, _ = rasterization(
                    G["means"], G["quats"], G["scales"], G["opac"], G["sh"],
                    w2c, Kt, CELLW, CELLH, sh_degree=g["deg"], packed=False,
                    backgrounds=bg[None], render_mode="RGB")
            buf.append(img[0].clamp(0, 1).mul(255).byte().cpu().numpy())
            amed = float(alpha.median())
            if (i + 1) % 48 == 0:
                log("      frame %3d/%d  α 中位 %.3f" % (i + 1, NFRAMES, amed))
        del G
        frames.append(buf)
        torch.cuda.empty_cache()
        log("      ✅ %s 渲完，α 中位 %.3f" % (p["c"]["name"], amed))

    # ── 阶段 3：拼接 + 编码 ──
    MARGIN, GAP, HEAD, INROW = 14, 8, 92, 132
    n = len(CASES)
    Wv = MARGIN * 2 + CELLW * n + GAP * (n - 1)
    Hv = HEAD + INROW + CELLH + 12
    Hv += Hv % 2
    canvas = Image.new("RGB", (Wv, Hv), "white")
    dr = ImageDraw.Draw(canvas)
    f_t = ImageFont.truetype(FONT_R, 17)
    f_n = ImageFont.truetype(FONT_B, 21)
    f_m = ImageFont.truetype(FONT_R, 14)
    f_i = ImageFont.truetype(FONT_R, 15)
    title = ("同一输入图 · 三方法各自重建 · 各自观测弧同步对照（span=%.0f°，各方法绕自身主体）"
             % span)
    tw = dr.textlength(title, font=f_t)
    dr.text(((Wv - tw) / 2, 9), title, font=f_t, fill="#6E6E6E")
    xs = []
    x0 = MARGIN
    for c in CASES:
        xs.append(x0)
        nw = dr.textlength(c["name"], font=f_n)
        cw = dr.textlength(c["meta"], font=f_m)
        tot = max(nw, cw)
        dr.text((x0 + (CELLW - tot) / 2, 36), c["name"], font=f_n, fill=c["color"])
        dr.text((x0 + (CELLW - cw) / 2, 64), c["meta"], font=f_m, fill="#8A8A8A")
        dr.rectangle([x0, HEAD - 8, x0 + CELLW, HEAD - 4], fill=c["color"])
        x0 += CELLW + GAP
    if os.path.exists(INPUT_IMG):
        im = Image.open(INPUT_IMG).convert("RGB")
        th = INROW - 12
        tw_ = int(im.width * th / im.height)
        im = im.resize((tw_, th))
        canvas.paste(im, (MARGIN, HEAD + 6))
        note = "输入：同一张单图（三方法各自从该图生成 3D）\n弧：各方法绕自身主体，方位参数同步"
        dr.text((MARGIN + tw_ + 14, HEAD + 30), note, font=f_i, fill="#4A4A4A")
    del dr

    mp4 = os.path.join(OUT, "three_way_arc.mp4")
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (Wv, Hv),
           "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-movflags", "+faststart", mp4]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    keep = {}
    for i in range(NFRAMES):
        frame = canvas.copy()
        for ci, x0 in enumerate(xs):
            frame.paste(Image.fromarray(frames[ci][i]), (x0, HEAD + INROW))
        proc.stdin.write(frame.tobytes())
        if i in (0, NFRAMES // 2, NFRAMES - 1):
            keep[i] = frame
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("❌ ffmpeg 编码失败")
    for i, fr in keep.items():
        fr.save(os.path.join(OUT, "three_way_arc_frame_%03d.png" % i))
    log("✅ %s（%.1f MB）" % (mp4, os.path.getsize(mp4) / 1e6))


if __name__ == "__main__":
    main()
