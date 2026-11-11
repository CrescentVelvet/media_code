#!/usr/bin/env python3
"""99p_ply_orbit_mp4.py — 3DGS PLY 水平 360° 环绕渲染 H.264 MP4（gsplat 离屏）。

与 99m（按 COLMAP 轨迹渲染）互补：99p 不依赖任何相机文件，**只从 PLY + 可选的
gs_camera_params.json 构造水平匀速圆环**，适合给单个重建产物快速出环绕展示片。

轨道（全部可用 env 覆盖）:
- 注视点：CAM_JSON 的 scene.center；没有 json 就退回高斯质心（opacity>0.05 的中位），
  此时建议核对抽帧目检。
- 半径：scene.radius × RADIUS_K（默认 1.0）——radius 来自 99k 的相机球拟合，
  正好让画面构图接近采集视角。
- 仰角：json setting.pitch_init（pitch = 相机仰角，Theta = 90 − pitch，与 99b:386-415
  同语义）；没有 json 默认 15°。
- 方位角：满 360° 匀速（N 帧闭环）；内参缩放到 OUTW，主点放画面正中。

PLY 解析沿用 99m 约定：rot 磁盘 (w,x,y,z) → gsplat (x,y,z,w)（ROT_ORDER=xyzw 关掉）；
f_rest 默认通道主序（SH_LAYOUT=coef 切官方 3DGS 系数主序，SH_LAYOUT=auto 仅参考）。

环境：conda env infinisplat_v2（gsplat 1.5.2 + torch 2.6 cu124），需要 NVIDIA GPU。

用法（WSL）:
    PLY=/mnt/d/x/point_cloud.ply python tools_99/99p_ply_orbit_mp4.py
    PLY=... CAM_JSON=... OUT=/tmp/orbit.mp4 ELEV=20 python tools_99/99p_ply_orbit_mp4.py

Env vars:
  PLY       输入高斯 PLY（必填）
  CAM_JSON  gs_camera_params.json（默认 <PLY 同目录>/gs_camera_params.json，不存在则拟合）
  OUT       输出 mp4（默认 <PLY 目录>/orbit.mp4）
  轨道     N（默认 240）, FPS（默认 30）, OUTW（默认 540）, RADIUS_K（默认 1.0）,
            ELEV（度，覆盖 pitch_init）, TARGET（"x,y,z" 手调注视点）
  渲染     BG（white 默认 | black）, ROT_ORDER（wxyz 默认 | xyzw）,
            SH_LAYOUT（channel 默认 | coef | auto）, DRY_RUN（=1 只算几何不渲染）
  编码     CRF（默认 18）
"""
import math
import os
import re
import subprocess
import sys

import numpy as np
import torch

from gsplat import rasterization

PLY = os.environ["PLY"]
CAM_JSON = os.environ.get("CAM_JSON", os.path.join(os.path.dirname(PLY), "gs_camera_params.json"))
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(PLY), "orbit.mp4"))
N = int(os.environ.get("N", "240"))
FPS = int(os.environ.get("FPS", "30"))
OUTW = int(os.environ.get("OUTW", "540"))
RADIUS_K = float(os.environ.get("RADIUS_K", "1.0"))
ELEV = os.environ.get("ELEV", "").strip()
TARGET = os.environ.get("TARGET", "").strip()
BG = os.environ.get("BG", "white")
ROT_ORDER = os.environ.get("ROT_ORDER", "wxyz")
SH_LAYOUT = os.environ.get("SH_LAYOUT", "channel")
DRY_RUN = os.environ.get("DRY_RUN", "") == "1"
CRF = os.environ.get("CRF", "18")
dev = "cuda"


def log(*a):
    print(*a, flush=True)


# ── PLY（与 99m/audit_22 同构：header 按 end_header 真实偏移） ──────────────
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
        offset = len(head)
        text = head.decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", text).group(1))
        names = re.findall(r"^property float (\w+)$", text, re.M)
        dt = np.dtype([(nm, "<f4") for nm in names])
        f.seek(offset)
        raw = np.fromfile(f, dtype=dt, count=n, offset=0)
    if raw.shape[0] != n:
        raise SystemExit("❌ %s: 读到 %d 顶点，头里写 %d" % (path, raw.shape[0], n))
    means = np.stack([raw["x"], raw["y"], raw["z"]], 1).astype(np.float32)
    scales = np.exp(np.stack([raw["scale_%d" % i] for i in range(3)], 1)).astype(np.float32)
    opac = (1.0 / (1.0 + np.exp(-raw["opacity"].astype(np.float64)))).astype(np.float32)
    quats = np.stack([raw["rot_%d" % i] for i in range(4)], 1).astype(np.float32)
    if ROT_ORDER == "wxyz":                       # 官方 3DGS 磁盘序 (w,x,y,z) → gsplat 要 xyzw
        quats = np.roll(quats, -1, axis=1)
    f_dc = np.stack([raw["f_dc_%d" % i] for i in range(3)], 1).astype(np.float32)
    rest = [nm for nm in names if nm.startswith("f_rest_")]
    k = len(rest) // 3
    if k:
        fr = np.stack([raw["f_rest_%d" % i] for i in range(len(rest))], 1)
        if SH_LAYOUT == "channel":                # 本仓实际资产全是通道主序（99m 默认）
            fr = fr.reshape(len(means), 3, k).transpose(0, 2, 1)
        else:                                     # coef=官方系数主序；auto 按 99m 规则判
            fr = fr.reshape(len(means), k, 3) if SH_LAYOUT == "coef" else _sh_auto(fr, n, k)
        sh = np.concatenate([f_dc[:, None, :], fr], 1)
    else:
        sh = f_dc[:, None, :]
    return dict(means=means, scales=scales, opac=opac, quats=quats, sh=sh, n=int(n),
                deg=int(round((1 + k) ** 0.5)) - 1)


def _sh_auto(fr, n, k):
    """auto：系数主序下相邻列（同基函数 RGB）应比通道主序更相关——重尾数据上
    不可靠（99m 文档 2026-10-31 两次误判），仅作参考。"""
    a = fr.reshape(n, k, 3)
    b = fr.reshape(n, 3, k).transpose(0, 2, 1)
    def score(m):
        d = np.abs(np.diff(m, axis=1)).mean()
        return d
    log("   ⚠️  SH_LAYOUT=auto：coef 平均列差 %.4f / channel %.4f"
        % (score(a), score(b)))
    return b if score(b) <= score(a) else a


# ── 相机 ─────────────────────────────────────────────────────────────────────
def look_at(eye, target, up):
    """world→camera，看向 +z（OpenCV/COLMAP 惯例，gsplat 同约定）。

    x = cross(forward, up)（右），y = cross(forward, x)（下，注意是 −up）——
    第二个基向量用 down 而不是 up，否则画面上下颠倒（99l 文档 2026-10-31 踩过）。
    """
    z = target - eye
    z = z / (np.linalg.norm(z) + 1e-12)
    if abs(float(z @ up)) > 0.98:
        up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
    x = np.cross(z, up)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    Rcw = np.stack([x, y, z], 1)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = Rcw.T
    w2c[:3, 3] = -Rcw.T @ eye
    return w2c


def load_cam_json(path):
    import json
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    s, sc, intr = d["setting"], d["scene"], d["intrinsics"]
    return dict(center=np.array(sc["center"], dtype=np.float64),
                radius=float(sc["radius"]),
                elev=float(s.get("pitch_init", 15.0)),
                yaw0=float(s.get("yaw_init", 0.0)),
                K=np.array([[intr["fx"], 0, intr["cx"]],
                            [0, intr["fy"], intr["cy"]],
                            [0, 0, 1]], dtype=np.float64),
                W0=int(intr["width"]), H0=int(intr["height"]))


def ffmpeg_exe():
    from shutil import which
    p = which("ffmpeg")
    if p:
        return p
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)

    g = read_ply(PLY)
    log("📦 PLY：%s ｜ %d 顶点 ｜ SH degree %d ｜ %.0f MB"
        % (os.path.basename(PLY), g["n"], g["deg"], os.path.getsize(PLY) / 1e6))
    P = g["means"]
    hi = g["opac"] > 0.05
    med = np.median(P[hi], axis=0) if hi.any() else np.median(P, axis=0)
    log("   包围盒 边长 %.3f × %.3f × %.3f m" % tuple(P.max(0) - P.min(0)))

    if os.path.isfile(CAM_JSON):
        cam = load_cam_json(CAM_JSON)
        mu, Rb, elev, yaw0 = cam["center"], cam["radius"] * RADIUS_K, cam["elev"], cam["yaw0"]
        K0, W0, H0 = cam["K"], cam["W0"], cam["H0"]
        log("📷 相机参数：%s（scene.center / radius × %.2f / pitch_init）"
            % (os.path.basename(CAM_JSON), RADIUS_K))
    else:
        log("⚠️  没有 %s —— 从点云拟合轨道（构图可能偏，务必看抽帧）" % CAM_JSON)
        mu = med
        horiz = np.linalg.norm((P[hi] if hi.any() else P)[:, [0, 2]] - mu[[0, 2]], axis=1)
        Rb = float(np.percentile(horiz, 95)) * 1.4 * RADIUS_K
        elev, yaw0 = 15.0, 0.0
        W0, H0 = 1080, 1920                        # 竖构图缺省，只影响内参比例
        f = 0.8 * W0
        K0 = np.array([[f, 0, (W0 - 1) / 2.0], [0, f, (H0 - 1) / 2.0], [0, 0, 1.0]])
    if TARGET:
        mu = np.array([float(x) for x in TARGET.split(",")], dtype=np.float64)
        log("   ⚙️  TARGET 覆盖 → 注视点 (%.3f, %.3f, %.3f)" % (mu[0], mu[1], mu[2]))
    if ELEV:
        elev = float(ELEV)
    log("   注视点 (%.3f, %.3f, %.3f) ｜ 半径 %.3f m ｜ 仰角 %.1f° ｜ 起点方位角 %.1f°"
        % (mu[0], mu[1], mu[2], Rb, elev, yaw0))
    log("   点云质心到注视点 %.3f m" % np.linalg.norm(P.mean(0) - mu))

    kw = OUTW / float(W0)
    OUTH = int(round(H0 * kw))
    OUTH += OUTH % 2
    K = K0.copy()
    K[0, 0] *= OUTW / float(W0)
    K[1, 1] *= OUTH / float(H0)
    K[0, 2] = (OUTW - 1) / 2.0
    K[1, 2] = (OUTH - 1) / 2.0
    log("🖼  输出 %d×%d ｜ fx %.1f fy %.1f（原 %d×%d, fx %.1f）"
        % (OUTW, OUTH, K[0, 0], K[1, 1], W0, H0, K0[0, 0]))

    if DRY_RUN:
        log("🧪 DRY_RUN：几何已算完，跳过渲染与编码")
        return

    G = {k: torch.from_numpy(g[k]).to(dev) for k in ("means", "quats", "scales", "opac", "sh")}
    bg = torch.ones(3, device=dev) if BG == "white" else torch.zeros(3, device=dev)

    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (OUTW, OUTH),
           "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", CRF,
           "-movflags", "+faststart", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    up = np.array([0.0, 1.0, 0.0])
    el = math.radians(elev)
    th0 = math.radians(yaw0)
    keeps = []
    try:
        for i in range(N):
            th = th0 + 2 * math.pi * i / N       # 水平匀速圆环，闭环
            eye = mu + Rb * np.array([math.cos(el) * math.cos(th),
                                      math.sin(el),
                                      math.cos(el) * math.sin(th)])
            w2c = look_at(eye, mu, up)
            with torch.no_grad():
                img, alpha, _ = rasterization(
                    G["means"], G["quats"], G["scales"], G["opac"], G["sh"],
                    torch.from_numpy(w2c).float().to(dev)[None],
                    torch.from_numpy(K).float().to(dev)[None],
                    OUTW, OUTH, sh_degree=g["deg"], packed=False,
                    backgrounds=bg[None], render_mode="RGB")
            frame = img[0].clamp(0, 1).mul(255).byte().cpu().numpy()
            proc.stdin.write(frame.tobytes())
            if i in (0, N // 2):
                keeps.append((i, frame.copy()))
            if (i + 1) % 60 == 0 or i == 0:
                log("   frame %3d/%d  α 中位 %.3f" % (i + 1, N, float(alpha.median())))
    except BrokenPipeError:
        proc.wait()
        raise SystemExit("❌ ffmpeg 编码中断（exit %d）" % proc.returncode)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("❌ ffmpeg 编码失败（exit %d）" % proc.returncode)

    from PIL import Image
    out_dir = os.path.dirname(os.path.abspath(OUT))
    stem = os.path.splitext(os.path.basename(OUT))[0]
    for i, fr in keeps:
        png = os.path.join(out_dir, "%s_frame_%03d.png" % (stem, i))
        Image.fromarray(fr).save(png)
        log("  🖼  目检帧: %s" % png)
    log("🎉 Done. %s（%d 帧 / %d fps / %.1f 秒 / %.1f MB）"
        % (OUT, N, FPS, N / float(FPS), os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()
