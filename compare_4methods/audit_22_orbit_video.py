#!/usr/bin/env python3
"""把 DA3-GS 的 PLY 绕一圈渲染成 MP4（用于与 supersplat 手动查看做观感对照）。

设计要点
--------
1. **读的是交付的 PLY 本身**（`official_export/i12_fullpts_sh/0000.ply`，610 万点 / 953 MB），
   不是内存里的高斯 —— supersplat 打开的就是这个文件，这样比才有意义。
   读法沿用 E8/E9 的约定：header 按 `end_header` 真实偏移、f_rest 是 channel-major。
2. **轨道从真实相机环拟合**，不凭空假设球心在原点：32 个训练相机中心做 PCA，
   最小方差方向 = 环平面法向（世界 up），环平面内两轴 e1/e2 张成方位角。
   半径取相机中心到质心的 90 分位 × RADIUS_K，仰角默认取原始环的平均仰角 ——
   这样轨迹落在这批相机"同一圈"上，画面构图与训练视角可比。
3. 内参按等比例放大（默认 540×720，3:4 同原帧 378×504），sh_degree 取 PLY 实际阶数。
4. 输出：`orbit_video/orbit.mp4` + 抽三帧 `frame_000/120/240.png` 便于并排核看。

环境变量：SC / NPZ（必填）、PLY / OUT / NFRAMES / FPS / OUTW / RADIUS_K / ELEV / BG
"""
import os
import re
import subprocess

import numpy as np
import torch

from gsplat import rasterization

SC = os.environ["SC"]
NPZ = os.environ["NPZ"]
PLY = os.environ.get("PLY", os.path.join(SC, "official_export/i12_fullpts_sh/0000.ply"))
OUT = os.environ.get("OUT", os.path.join(SC, "orbit_video"))
NFRAMES = int(os.environ.get("NFRAMES", "300"))
FPS = int(os.environ.get("FPS", "30"))
OUTW = int(os.environ.get("OUTW", "540"))
RADIUS_K = float(os.environ.get("RADIUS_K", "1.06"))
ELEV = os.environ.get("ELEV", "").strip()
TARGET = os.environ.get("TARGET", "").strip()
BG = os.environ.get("BG", "white")
DRYRUN = os.environ.get("DRYRUN", "") == "1"
# ARC=1：只在「有观测的弧」上走，跳过采集缺口 —— 环视展示用。
# 默认（ARC=0）走满 360°，缺口处会露出拖影，那是 PLY 本身没数据，supersplat 里也一样。
ARC = os.environ.get("ARC", "") == "1"
OUT_NAME = os.environ.get("OUT_NAME", "orbit")
dev = "cuda"


def log(*a):
    print(*a, flush=True)


# ── PLY ──────────────────────────────────────────────────────────────────────
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
    f_dc = np.stack([raw["f_dc_%d" % i] for i in range(3)], 1).astype(np.float32)
    rest = [nm for nm in names if nm.startswith("f_rest_")]
    k = len(rest) // 3
    if k:                                    # E8：channel-major → (N,K,3)
        fr = np.stack([raw["f_rest_%d" % i] for i in range(len(rest))], 1)
        fr = fr.reshape(len(means), 3, k).transpose(0, 2, 1)
        sh = np.concatenate([f_dc[:, None, :], fr], 1)
    else:
        sh = f_dc[:, None, :]
    return dict(means=means, scales=scales, opac=opac, quats=quats, sh=sh, n=int(n),
                deg=int(round((1 + k) ** 0.5)) - 1)


# ── 轨道拟合 ─────────────────────────────────────────────────────────────────
def fit_orbit(extr):
    """extr: (V,3,4) world→camera。

    相机中心 C = −R_wcᵀ t_wc；环平面基由 C 的 PCA 给出。
    **注视点不能取 C 的均值** —— 采集轨迹常带系统性偏置（首跑钻进书架就是这个错），
    改用 32 条视线的**最小二乘交点**：min_x Σ‖(I − d dᵀ)(x − C)‖²，
    解 (ΣP) x = ΣP C。这才是「这批相机共同盯着哪儿」。
    """
    Rwc, twc = extr[:, :3, :3], extr[:, :3, 3]
    C = -np.transpose(Rwc, (0, 2, 1)) @ twc[..., None]
    C = C[..., 0]                                  # (V,3) 相机中心（世界）
    # 视线方向：DA3 / COLMAP 惯例是看向 **+z**（实测：首版按 −z 算，与真实视线
    # 夹角中位 178°，整条轨道朝向全反 → 第一版视频钻进书架）。这里不假定时序，
    # 用「指向交点」与「相机轴」的夹角自动判向，判完再断言一次。
    d = np.transpose(Rwc, (0, 2, 1)) @ np.array([0.0, 0.0, 1.0])
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for Ci, di in zip(C, d):
        P = np.eye(3) - np.outer(di, di)
        A += P
        b += P @ Ci
    target = np.linalg.solve(A, b)

    # 不变量：从相机指向 target 的方向必须与真实视线重合；不重合就是符号/坐标系搞错了。
    v = target[None, :] - C
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    ang = np.degrees(np.arccos(np.clip(np.sum(v * d, 1), -1, 1)))
    if np.median(ang) > 90:          # 相机轴反向（OpenGL 惯例），翻一次
        d = -d
        ang = 180.0 - ang
    if np.median(ang) > 10:
        raise SystemExit("❌ 注视点与真实视线夹角中位 %.1f° —— 坐标系假设错了，别往下渲"
                         % np.median(ang))

    # 环平面基与"共面度"要从相机环的几何中心算 —— 若用 target 当基准，
    # target 在法向上的偏移会伪装成"不共面"（实测奇异值比 0.03 → 0.35）。
    Cc = C.mean(0)
    _, sv, vt = np.linalg.svd(C - Cc, full_matrices=False)
    axis = vt[2]                                   # 环平面法向（方向待定）
    # 法向有两个解，PCA 给哪个都合法。必须用相机自身的"上"定符号 ——
    # 否则 look_at 里的 x = up×forward 会整体反号，构造出的相机相对世界转 180°
    # （首版就是这样：渲染出来人在背后，画面全是书柜）。
    up_cam = -(np.transpose(Rwc, (0, 2, 1)) @ np.array([0.0, 1.0, 0.0]))  # 相机 y 向下
    up_cam /= np.maximum(np.linalg.norm(up_cam, axis=1, keepdims=True), 1e-12)
    up_w = up_cam.mean(0)
    up_w /= np.linalg.norm(up_w) + 1e-12
    if float(axis @ up_w) < 0:
        axis = -axis
    up_ang = float(np.degrees(np.arccos(np.clip(abs(axis @ up_w), -1, 1))))
    r = np.linalg.norm(C - target, axis=1)         # 轨道半径：相机到注视点
    el = np.degrees(np.arcsin(np.clip(((C - target) @ axis) /
                                     np.maximum(r, 1e-9), -1, 1)))
    # 起始方位角对齐到 0 号相机（它是参考视角，画面等同 view 0）
    th_i = np.arctan2((C - Cc) @ vt[1], (C - Cc) @ vt[0])
    order = np.argsort(th_i)
    ring = dict(th=th_i[order], C=C[order], th0=float(th_i[0]))
    return C, target, axis, vt[0], vt[1], r, el, sv, d, float(np.median(ang)), ring, up_ang


def ffmpeg_exe():
    """WSL 里通常没有系统 ffmpeg，退回 imageio_ffmpeg 自带的那份。"""
    from shutil import which
    p = which("ffmpeg")
    if p:
        return p
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def look_at(eye, target, up):
    """构造 world→camera，**相机看向 +z**（COLMAP/OpenCV 惯例，gsplat 用的就是这套）。

    首版按 OpenGL（看向 −z）写，喂给 gsplat 后整条轨道反了 180° ——
    画面里全是书柜和窗帘，人在相机背后。自证办法：拿 0 号相机的真实外参对一遍
    （main 里断言旋转差 < 5°）。
    """
    z = target - eye
    z = z / (np.linalg.norm(z) + 1e-12)              # 相机 +z = 视线方向
    if abs(float(z @ up)) > 0.98:
        up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
    x = np.cross(z, up)                              # 右 = forward × up（手性！）
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)                               # 下 = forward × 右
    Rcw = np.stack([x, y, z], 1)                     # 列 = 相机三轴在世界系
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = Rcw.T
    w2c[:3, 3] = -Rcw.T @ eye
    return w2c


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(NPZ)
    extr = d["extrinsics"].astype(np.float64)
    if extr.ndim == 4:
        extr = extr[:, :3, :]
    K0 = d["intrinsics"].astype(np.float64)[0]
    V, H0, W0 = d["depth"].shape[0], d["depth"].shape[1], d["depth"].shape[2]

    C, mu, axis, e1, e2, r, el, sv, dirs, ang_med, ring, up_ang = fit_orbit(extr)
    Rb = float(np.percentile(r, 90) * RADIUS_K)
    elev = float(ELEV) if ELEV else float(np.median(el))
    log("📷 相机环拟合")
    log("   注视点（视线最小二乘交点）(%.3f, %.3f, %.3f) m" % (mu[0], mu[1], mu[2]))
    log("   ✅ 不变量：指向注视点 vs 真实视线 夹角中位 %.2f°（阈值 10°）" % ang_med)
    log("   相机中心均值 (%.3f, %.3f, %.3f) m ｜ 差值 %.3f m"
        % (C.mean(0)[0], C.mean(0)[1], C.mean(0)[2], np.linalg.norm(C.mean(0) - mu)))
    log("   半径 90 分位 %.3f m（相机到注视点）" % np.percentile(r, 90))
    log("   环平面法向 (%.3f, %.3f, %.3f) ｜ 奇异值比 %.4f（越小越共面）"
        % (axis[0], axis[1], axis[2], sv[2] / max(sv[0], 1e-12)))
    log("   ✅ 法向已按相机 up 定向：与相机 up 夹角 %.2f°（残差来自相机俯仰，反向会是 ~180°−该值）"
        % up_ang)
    if up_ang > 60:
        raise SystemExit("❌ 环平面法向与相机 up 夹角 %.1f° —— 定符号失败了" % up_ang)
    log("   原始仰角 中位 %.1f°  范围 %.1f° ~ %.1f°"
        % (np.median(el), el.min(), el.max()))
    log("   轨迹 = 沿真实相机位置的方位角插值（不另造圆；首帧对齐 0 号相机）")

    # 自证：用 0 号相机（参考视角）的真实外参校验 look_at 的构造约定。
    w_chk = look_at(C[0], mu, axis)
    R_err = float(np.degrees(np.arccos(np.clip(
        (np.trace(w_chk[:3, :3].T @ extr[0, :3, :3]) - 1) / 2, -1, 1))))
    if R_err > 5:
        raise SystemExit("❌ look_at(C[0]) 与真实外参旋转差 %.1f° —— 相机约定又错了" % R_err)
    log("   ✅ 自证：look_at(C[0]) vs 真实外参 0 号，旋转差 %.2f°（阈值 5°）" % R_err)

    g = read_ply(PLY)
    log("📦 PLY：%s ｜ %d 顶点 ｜ SH degree %d ｜ %.0f MB"
        % (os.path.basename(PLY), g["n"], g["deg"], os.path.getsize(PLY) / 1e6))
    P = g["means"]
    hi = g["opac"] > 0.05
    med = np.median(P[hi], axis=0) if hi.any() else np.median(P, axis=0)
    log("   点云包围盒 边长 %.3f × %.3f × %.3f m ｜ 质心 (%.3f, %.3f, %.3f)"
        % (*(P.max(0) - P.min(0)), *P.mean(0)))
    log("   点云中位中心 (%.3f, %.3f, %.3f) ｜ 到注视点 %.3f m"
        % (med[0], med[1], med[2], np.linalg.norm(med - mu)))
    if TARGET:                                  # 手调逃生口：TARGET=x,y,z
        mu = np.array([float(x) for x in TARGET.split(",")], dtype=np.float64)
        log("   ⚙️  TARGET 覆盖 → 注视点 (%.3f, %.3f, %.3f)" % (mu[0], mu[1], mu[2]))
    log("   点云质心到注视点 %.3f m" % np.linalg.norm(P.mean(0) - mu))

    G = {k: torch.from_numpy(g[k]).to(dev) for k in ("means", "quats", "scales", "opac", "sh")}
    deg = g["deg"]

    kw = OUTW / float(W0)
    OUTH = int(round(H0 * kw))
    OUTH += OUTH % 2
    K = K0.copy()
    K[0, 0] *= OUTW / float(W0)
    K[1, 1] *= OUTH / float(H0)
    K[0, 2] = (OUTW - 1) / 2.0          # 主点放到画面正中：轨道视图没有原始 c2w 的偏置
    K[1, 2] = (OUTH - 1) / 2.0
    log("🖼  输出 %d×%d ｜ fx %.1f fy %.1f（原帧 %d×%d, fx %.1f）"
        % (OUTW, OUTH, K[0, 0], K[1, 1], W0, H0, K0[0, 0]))

    bg = torch.ones(3, device=dev) if BG == "white" else torch.zeros(3, device=dev)
    if DRYRUN:
        log("🧪 DRYRUN：几何已算完，跳过渲染与编码")
        return
    mp4 = os.path.join(OUT, OUT_NAME + ".mp4")
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (OUTW, OUTH),
           "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-movflags", "+faststart", mp4]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    ths = ring["th"]
    Cs = ring["C"]
    two_pi = 2 * np.pi
    ths_c = np.concatenate([ths, ths[:1] + two_pi])          # 闭环
    Cs_c = np.vstack([Cs, Cs[:1]])
    gaps = np.diff(ths_c)
    if gaps.min() <= 1e-6:
        raise SystemExit("❌ 方位角有重复/退化，无法插值")
    th_start = float(ring["th0"])                            # 0 号相机所在方位角 → 首帧 ≈ view 0

    base = float(ths[0])                                     # 插值表的起点
    g_all = np.diff(np.concatenate([ths, ths[:1] + two_pi]))
    k_max = int(np.argmax(g_all))
    log("   方位角分布：中位间隔 %.1f°　最大缺口 %.1f°（%.0f° → %.0f° 之间没有相机）"
        % (np.degrees(np.median(g_all)), np.degrees(g_all[k_max]),
           np.degrees(ths[k_max]), np.degrees(ths[(k_max + 1) % len(ths)])))
    if np.degrees(g_all[k_max]) > 30:
        log("   ⚠️ 采集不完整环绕：有 %.0f° 没有相机观测 —— 走进这段必然出现拖影/空洞，"
            "PLY 里本来就没有那块数据" % np.degrees(g_all[k_max]))

    if ARC:                                                  # 只走有观测的弧
        th_seq = np.degrees(g_all[k_max])
        start = ths[(k_max + 1) % len(ths)]                  # 缺口结束端 = 观测弧起点
        span = two_pi - g_all[k_max]
        log("   🎞  ARC 模式：观测弧跨度 %.1f°，从 %.1f° 起走" % (np.degrees(span), np.degrees(start)))
        th_list = [start + span * i / max(NFRAMES - 1, 1) for i in range(NFRAMES)]
    else:
        log("   🎞  360° 模式：满一圈，缺口段会露出拖影（诚实展示数据边界）")
        th_list = [th_start + two_pi * i / NFRAMES for i in range(NFRAMES)]

    keep = []
    for i, th in enumerate(th_list):
        t = base + (th - base) % two_pi                      # 折回插值表区间
        k = int(np.clip(np.searchsorted(ths_c, t) - 1, 0, len(ths_c) - 2))
        w = (t - ths_c[k]) / gaps[k]
        eye = Cs_c[k] * (1 - w) + Cs_c[k + 1] * w          # 落在真实相机之间
        w2c = look_at(eye, mu, axis)
        with torch.no_grad():
            img, alpha, _ = rasterization(
                G["means"], G["quats"], G["scales"], G["opac"], G["sh"],
                torch.from_numpy(w2c).float().to(dev)[None],
                torch.from_numpy(K).float().to(dev)[None],
                OUTW, OUTH, sh_degree=deg, packed=False,
                backgrounds=bg[None], render_mode="RGB")
        frame = (img[0].clamp(0, 1).mul(255).byte().cpu().numpy())
        proc.stdin.write(frame.tobytes())
        if i in (0, NFRAMES // 2):
            keep.append((i, frame.copy()))
        if (i + 1) % 30 == 0 or i == 0:
            log("   frame %3d/%d  α 中位 %.3f" % (i + 1, NFRAMES, float(alpha.median())))

    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("❌ ffmpeg 编码失败")

    from PIL import Image
    for i, fr in keep:
        Image.fromarray(fr).save(os.path.join(OUT, "%s_frame_%03d.png" % (OUT_NAME, i)))
    log("✅ %s（%d 帧 / %d fps / %.1f 秒 / %.1f MB）"
        % (mp4, NFRAMES, FPS, NFRAMES / FPS, os.path.getsize(mp4) / 1e6))


if __name__ == "__main__":
    main()
