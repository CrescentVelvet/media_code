#!/usr/bin/env python3
"""05_render_orbit.py — Pi3X 点云渲染展示片（环绕 orbit + GT 轨迹重投影对照）。

Pi3X 产物是彩色点云（非高斯），不能用 gsplat；这里用 Sim(3) 对齐到 GT 系后做
z-buffer 点 splat（远→近顺序写 3x3 块近似遮挡），纯 numpy，不用 OpenGL。

Env vars:
  SCENE     场景名（默认 jiajia_baozhen）
  TAG       预测标签（默认 nocond）
  RESULTS_DIR  结果根（默认 ~/output/pi3x_results）
  CONF      置信度阈值（默认 0.5，sigmoid 后）
  ORBIT_N   环绕帧数（默认 180，30fps → 6s）
  ORBIT_W/H 环绕分辨率（默认 768x1024，3:4 竖屏与采集画幅一致）
  ELEV      环绕仰角（默认 12°）
  MAXPTS    环绕抽稀上限（默认 400 万；重投影不抽稀）
输出（<scene>/ 下）:
  orbit_<tag>.mp4 / poster_orbit.png / reproject.mp4 / poster_reproject.png
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

SCENE = os.environ.get("SCENE", "jiajia_baozhen")
TAG = os.environ.get("TAG", "nocond")
RESULTS = os.environ.get("RESULTS_DIR", os.path.expanduser("~/output/pi3x_results"))
CONF = float(os.environ.get("CONF", "0.5"))
ORBIT_N = int(os.environ.get("ORBIT_N", "180"))
OW, OH = int(os.environ.get("ORBIT_W", "768")), int(os.environ.get("ORBIT_H", "1024"))
ELEV = float(os.environ.get("ELEV", "5"))
MAXPTS = int(os.environ.get("MAXPTS", "4000000"))
FFMPEG = os.environ.get("FFMPEG", os.path.expanduser("~/miniconda3/envs/colmap/bin/ffmpeg"))

root = os.path.join(RESULTS, SCENE)
pred = np.load(os.path.join(root, TAG, "predictions.npz"), allow_pickle=True)
gt = np.load(os.path.join(root, "conditions.npz"))
meta = json.load(open(os.path.join(root, "gt_meta.json")))

pts = pred["points"].reshape(-1, 3).astype(np.float32)
rgb = pred["images"].reshape(-1, 3)
conf = 1.0 / (1.0 + np.exp(-pred["conf"].reshape(-1).astype(np.float32)))
keep = conf > CONF
pts, rgb = pts[keep], rgb[keep]
print(f"🚀 points: {len(pts):,} (conf>{CONF})")


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    c = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ c @ U.T
    s = float((sig * np.diag(c)).sum() / ((S ** 2).sum() / len(src)))
    return s, R, mu_d - s * R @ mu_s


# Sim(3) 对齐 pred→GT：旋转必须用 Wahba（直接对齐相机旋转矩阵），尺度取
# 中心 Umeyama 的 s，平移在固定 R/s 后由相机中心均值求出。
# 不可直接用中心 Umeyama 的 R/t：本包轨迹近平面，中心拟合把 27.81° 的
# 面外旋转残差塞进 R，旧版因此把人物投到左下、帧0相机中心偏 0.372m。
# 2026-11-20 实测：Wahba 口径 ATE 0.379m → 0.0107m。
def wahba(Rp, Rg):
    M = np.einsum("nij,nkj->ik", Rp, Rg)  # sum_i Rp_i @ Rg_i^T
    U, _, Vt = np.linalg.svd(M)
    corr = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ corr @ U.T


P = pred["camera_poses"][:, :3, 3].astype(np.float64)
G = gt["poses"][:, :3, 3].astype(np.float64)
Rp = pred["camera_poses"][:, :3, :3].astype(np.float64)
Rg = gt["poses"][:, :3, :3].astype(np.float64)
s, _, _ = umeyama(P, G)
R = wahba(Rp, Rg)
t = (G - s * (P @ R.T)).mean(0)
pts_m = (s * (pts.astype(np.float64) @ R.T) + t).astype(np.float32)
ate = np.linalg.norm(s * (P @ R.T) + t - G, axis=1)
del pts, P
print(f"📐 Sim(3) scale={s:.4f}; Wahba+meanT ATE RMSE={np.sqrt(np.mean(ate**2)):.4f} m")

# 注视点：主体附近点云的稳健质心。不要直接用轨道弧心（主体可能偏出画面），
# 也不要让轨道半径决定取景距离（本包 median radius=1.13m，会把首帧头部裁掉）。
# 轨道仍绕主体旋转；取景距离独立设置（本包 median radius=1.13m，会裁首帧头部），
center = G.mean(0)
radius = float(np.median(np.linalg.norm(G - center, axis=1)))
_near = np.linalg.norm(pts_m - center, axis=1) < 1.2 * radius
if _near.sum() > 10000:
    center = np.median(pts_m[_near], axis=0)
orbit_dist = float(os.environ.get("ORBIT_DIST", "2.2"))  # m；GT0 距主体 1.61m 会裁头，2.2m 留全身余量
# 世界朝向不假设 Y-up：从 GT 相机取「图像向下」轴的均值当世界重力方向
# （c2w 第 2 列 = 相机 +y 在世界系的指向，正常拍摄 ≈ 世界 down）。
# 2026-11-20 教训：先按 ARKit Y-up 假设（up=(0,1,0)）环绕画面侧翻 90°。
down_hint = gt["poses"][:, :3, 1].astype(np.float64).mean(0)
down_hint /= np.linalg.norm(down_hint)
up_vec = -down_hint
# 水平面内取一对正交基作环绕方向
u0 = np.cross(up_vec, [1.0, 0.0, 0.0])
if np.linalg.norm(u0) < 1e-6:
    u0 = np.cross(up_vec, [0.0, 0.0, 1.0])
u0 /= np.linalg.norm(u0)
v0 = np.cross(up_vec, u0)
# 环绕起点对齐 GT 第 0 帧方位（首帧=真实采集视角，构图与内容对照得上）
d0 = G[0] - center
d0_h = d0 - up_vec * np.dot(d0, up_vec)
th0 = float(np.arctan2(np.dot(d0_h, v0), np.dot(d0_h, u0)))
print(f"🎯 orbit center={center.round(2)}, radius={radius:.2f} m, dist={orbit_dist:.2f} m, world_up={up_vec.round(2)}, th0={np.degrees(th0):.1f}°")

# 内参：GT 第 0 帧，按渲染分辨率缩放
K0 = gt["intrinsics"][0].astype(np.float64)
cw, ch = meta["crop_xywh"][2], meta["crop_xywh"][3]


def K_at(w, h):
    sc = w / cw
    K = K0.copy()
    K[0, 0] *= sc; K[0, 2] = w / 2.0
    K[1, 1] = K0[1, 1] * (h / ch); K[1, 2] = h / 2.0
    return K


def splat_render(pts_w, rgb, c2w, K, w, h):
    """远→近顺序写 3x3 splat，返回 u8 图。"""
    w2c = np.linalg.inv(c2w)
    pc = (w2c[:3, :3] @ pts_w.T).T + w2c[:3, 3]
    z = pc[:, 2]
    uvw = (K @ pc.T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-9, None)
    m = (z > 0.1) & (uv[:, 0] >= -1) & (uv[:, 0] < w + 1) & (uv[:, 1] >= -1) & (uv[:, 1] < h + 1)
    uvi = np.round(uv[m]).astype(np.int64)
    zm, cm = z[m], rgb[m]
    order = np.argsort(-zm)  # 远的先画，近的覆盖
    canvas = np.full((h, w, 3), 255, np.uint8)  # 白底（与 figkit/前作一致）
    uo, vo = uvi[order, 0], uvi[order, 1]
    co = cm[order]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            xx = np.clip(uo + dx, 0, w - 1)
            yy = np.clip(vo + dy, 0, h - 1)
            canvas[yy, xx] = co
    return canvas


def encode_mp4(frames, path, fps):
    """frames: list of HxWx3 u8，pipe 进 ffmpeg。"""
    h, w = frames[0].shape[:2]
    cmd = [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "20", "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for fr in frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"✅ saved: {path} ({len(frames)} 帧)")


# ---------- 1) 环绕 orbit ----------
DO_ORBIT = os.environ.get("DO_ORBIT", "1") == "1"
DO_REPROJECT = os.environ.get("DO_REPROJECT", "1") == "1"
K_orb = K_at(OW, OH)
orbit_pts = pts_m
if len(orbit_pts) > MAXPTS:
    idx = np.arange(0, len(orbit_pts), len(orbit_pts) // MAXPTS)
    orbit_pts, orbit_rgb = orbit_pts[idx], rgb[idx]
else:
    orbit_rgb = rgb
frames = []
for i in range(ORBIT_N if DO_ORBIT else 0):
    th = th0 + 2 * np.pi * i / ORBIT_N
    el = np.radians(ELEV)
    eye = center + orbit_dist * (np.cos(th) * np.cos(el) * u0 + np.sin(th) * np.cos(el) * v0 + np.sin(el) * up_vec)
    # look-at（OpenCV 系：+z 看向目标，+y=图像向下，须对齐世界重力方向）：
    # right=cross(down,fwd)、down=cross(fwd,right)；顺序写反画面会颠倒/侧翻
    fwd = center - eye; fwd /= np.linalg.norm(fwd)
    down = down_hint - fwd * np.dot(down_hint, fwd)  # 重力方向投影到成像面
    down /= np.linalg.norm(down)
    right = np.cross(down, fwd); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([right, down, fwd], axis=1)
    c2w[:3, 3] = eye
    frames.append(splat_render(orbit_pts, orbit_rgb, c2w, K_orb, OW, OH))
    if (i + 1) % 60 == 0:
        print(f"  orbit {i+1}/{ORBIT_N}")
if DO_ORBIT:
    Image.fromarray(frames[0]).save(os.path.join(root, "poster_orbit.png"))
    encode_mp4(frames, os.path.join(root, f"orbit_{TAG}.mp4"), 30)
del frames

# ---------- 2) GT 轨迹重投影对照（左 GT 实拍 | 右点云投影） ----------
RW, RH = 720, 960
K_r = K_at(RW, RH)
img_dir = os.path.join(root, "images")
frames = []
n = len(gt["poses"]) if DO_REPROJECT else 0
for i in range(n):
    proj = splat_render(pts_m, rgb, gt["poses"][i].astype(np.float64), K_r, RW, RH)
    real = Image.open(os.path.join(img_dir, f"{i:05d}.png")).convert("RGB").resize((RW, RH), Image.LANCZOS)
    frames.append(np.concatenate([np.asarray(real), proj], axis=1))
    if (i + 1) % 30 == 0:
        print(f"  reproject {i+1}/{n}")
if DO_REPROJECT:
    Image.fromarray(frames[0]).save(os.path.join(root, "poster_reproject.png"))
    encode_mp4(frames, os.path.join(root, "reproject.mp4"), 15)
print("🎉 Done.")
