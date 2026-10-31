#!/usr/bin/env python3
"""99l_gen_colmap_traj.py — 参数化运镜轨迹生成器：输出 COLMAP 文本三件套
（cameras.txt / images.txt / points3D.txt），供 99m 渲染、或喂给 99g / COLMAP GUI 目检。

环境准备
    纯标准库，任意 python3 即可（无需 conda）；仅 PLY 自动定心需要 numpy+plyfile。

与 99m 成对，轨迹格式约定（改格式必须两边同步）：
  * 帧序 = IMAGE_ID 升序（本脚本按 1..N 顺序写）；
  * images.txt 每条记录**占两行**（位姿行 + POINTS2D 行，后者可为空）——
    漏第二行是 COLMAP 解析错位的第一大坑；
  * qvec = (qw,qx,qy,qz)，tvec 为世界系→相机系（x_c = R x_w + t）；
  * 只写 PINHOLE 相机；dolly_zoom 逐帧一个 CAMERA_ID（焦距随距离等比缩放，
    主体在画面中的高度恒定），其余运镜共用 CAMERA_ID=1。

用法:
    python tools_99/99l_gen_colmap_traj.py                 # 用 main() 里的预设跑
    TYPE=dolly_zoom R0=1.0 R1=0.4 python ...               # env 覆盖单项参数
    PLY="" CENTER="0,0,0" RADIUS=2 python ...              # 关自动定心、手给中心/半径

Env vars（不设则用下方 main() 里的默认值）:
  通用        TYPE, OUT_DIR, PLY, CENTER, RADIUS, LOOK_AT, UP, N, W, H, FOV_Y, EASE
  orbit/arc   EL, AZ0, AZ_SWEEP          # arc 默认 EL=5 / AZ_SWEEP=120（低角度过肩）
  helix       EL0, EL1, AZ_SWEEP
  push_in/    R0, R1, R_AS_MULT（1=R0/R1 是 RADIUS 的倍数；0=直接米数）, EL, AZ0,
  dolly_zoom  DZ_SENSOR_H（只影响 dolly_zoom 焦距绝对值，不影响画面效果）
"""
import math
import os
import sys
from pathlib import Path


def die(msg):
    sys.exit(f"❌ {msg}")


def parse_vec(s, name, default=None):
    if s is None or not str(s).strip():
        if default is not None:
            return default
        die(f"{name} 未设置")
    try:
        v = [float(x) for x in str(s).split(",")]
    except ValueError:
        die(f"{name} 格式应为 x,y,z，实际: {s!r}")
    if len(v) != 3:
        die(f"{name} 需要 3 个数，实际 {len(v)} 个: {s!r}")
    return v


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def ease(t, mode):
    return smoothstep(t) if mode == "smoothstep" else t


def norm(v):
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-12:
        die("零向量无法归一化（UP 与视线方向平行？）")
    return [x / n for x in v]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def look_at_to_w2c(cam, target, up):
    """OpenCV 相机约定（z 前、x 右、y 向下）的 look-at → (R_w2c, t_w2c)。

    w2c 旋转的行 = 相机三个基向量在世界系的坐标：[right, down, fwd]。
    ⚠️ 两个易错点（2026-10-31 实测踩过）：
    * 行基向量组成的矩阵**就是** w2c 旋转，不能再转置；
    * OpenCV 的 y 轴向下，第二个基向量是 -up（down = cross(fwd, right)），
      用 +up 画面会上下颠倒且仍能通过「相机中心回算」检查。
    """
    fwd = norm([target[i] - cam[i] for i in range(3)])
    right = norm(cross(fwd, up))
    down = cross(fwd, right)                    # = -up 方向
    R_w2c = [right, down, fwd]
    t = [-sum(R_w2c[i][j] * cam[j] for j in range(3)) for i in range(3)]
    return R_w2c, t


def rotmat_to_qvec_wxyz(R):
    """R(3x3 行主序) → 单位四元数 (w,x,y,z)。Shepperd 法，R 接近对称时数值稳定。"""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return norm([w, x, y, z])


def sphere_pos(center, radius, az_deg, el_deg):
    """方位角 az（绕 +Y，度）、仰角 el（高于水平面，度）→ 世界坐标。"""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    r_xy = radius * math.cos(el)
    return [
        center[0] + r_xy * math.sin(az),
        center[1] + radius * math.sin(el),
        center[2] + r_xy * math.cos(az),
    ]


def auto_center_radius(ply_path):
    """PLY 自动定心：中位数中心 + q90 距离半径。需要 plyfile+numpy。

    不用包围盒：实测 flashworld 场景拉离散远点，bbox 半径 20m、主体在 1m 级
    （中位数中心与彩色点质心距离 <0.3m）；中位数 + 分位数对离散点鲁棒。
    q 用 AUTO_Q 覆盖（默认 0.90 = 「主体取景」口径；要含环境用 0.98）。
    """
    try:
        import numpy as np
        from plyfile import PlyData
    except ImportError:
        die("PLY 自动定心需要 plyfile+numpy（99m 的 conda env 有）；"
            "或在普通解释器里显式设 CENTER/RADIUS 并去掉 PLY")
    q = float(os.environ.get("AUTO_Q", "0.90"))
    data = PlyData.read(str(ply_path))["vertex"]
    xyz = np.stack([np.asarray(data[c]) for c in ("x", "y", "z")], axis=1)
    center = np.median(xyz, axis=0)
    radius = float(np.quantile(np.linalg.norm(xyz - center, axis=1), q))
    return center.tolist(), radius


def gen_frames(cfg):
    """按 cfg["type"] 生成 [(pos, focal_scale)]。focal_scale 只有 dolly_zoom 非 1。"""
    t_type = cfg["type"]
    n = cfg["n"]
    if n < 2:
        die("N 至少 2")

    center, radius = cfg["center"], cfg["radius"]
    if cfg.get("ply"):
        c_auto, r_auto = auto_center_radius(Path(cfg["ply"]))
        print(f"🎯 PLY 自动定心: center=({c_auto[0]:.3f},{c_auto[1]:.3f},{c_auto[2]:.3f}) "
              f"radius={r_auto:.3f}")
        center, radius = c_auto, r_auto

    look_at = cfg["look_at"]
    up = cfg["up"]
    ease_mode = cfg["ease"]
    dist = (lambda v: v * radius) if cfg["r_as_mult"] else (lambda v: v)

    az0 = cfg["az0"]
    frames = []
    f_scale = [1.0] * n

    if t_type in ("orbit", "arc"):
        el = cfg["el"]
        sweep = cfg["az_sweep"]
        for i in range(n):
            t = ease(i / (n - 1), ease_mode)
            frames.append(sphere_pos(center, dist(1.0), az0 + sweep * t, el))
    elif t_type == "helix":
        el0, el1 = cfg["el0"], cfg["el1"]
        sweep = cfg["az_sweep"]
        for i in range(n):
            t = ease(i / (n - 1), ease_mode)
            frames.append(sphere_pos(center, dist(1.0), az0 + sweep * t, el0 + (el1 - el0) * t))
    elif t_type in ("push_in", "dolly_zoom"):
        el = cfg["el"]
        r0, r1 = cfg["r0"], cfg["r1"]
        cam_dir = norm([sphere_pos(center, 1.0, az0, el)[i] - center[i] for i in range(3)])
        for i in range(n):
            t = ease(i / (n - 1), ease_mode)
            d = dist(r0) + (dist(r1) - dist(r0)) * t
            frames.append([center[j] + cam_dir[j] * d for j in range(3)])
            if t_type == "dolly_zoom":
                # 主体画面高度 ∝ 焦距/距离：距离变焦距同比变，主体占比恒定
                f_scale[i] = d / dist(r0)
    else:
        die(f"未知 TYPE: {t_type}")

    return frames, f_scale, look_at, up


def main():
    # Windows 控制台默认 cp936，直接打印 emoji 会 UnicodeEncodeError；
    # WSL/Linux 下这个调用无害
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ===== 预设路径与参数，改模型/批次时只改这几行 =====
    # 源：待渲染的 3DGS PLY（本机 WSL 示例路径；服务器上改成服务器路径）。
    # 给了就自动定心；不想自动定心设 PLY="" 并显式给 CENTER / RADIUS
    PLY = Path(
        "/mnt/d/output/flashworld_results/03_gaussians/"
        "01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕_frame000/gaussians.ply"
    )
    TYPE = "orbit"                                  # orbit | helix | dolly_zoom | push_in | arc
    # 输出：COLMAP 三件套目录（相对 cwd=media_code/ 解析，与 99a~99f 同一约定）
    OUT_DIR = Path("../../output/traj/demo_orbit")
    # 常用运镜参数（其余走 env 覆盖，见文件头 Env vars 清单）
    N = 120                 # 帧数（= 时长 × FPS，99m 侧默认 30fps）
    AZ_SWEEP = 360          # 方位角扫过范围（度）
    EL = 15                 # 相机仰角（度，高于目标）；arc 建议 5
    # ======================================================

    # env 覆盖（不设则用上面的预设）
    ply_s = os.environ.get("PLY", str(PLY)).strip()
    t_type = os.environ.get("TYPE", TYPE).strip()
    look_env = os.environ.get("LOOK_AT")
    cfg = {
        "type": t_type,
        "n": int(os.environ.get("N", str(N))),
        "ply": None if not ply_s else ply_s,
        "center": parse_vec(os.environ.get("CENTER"), "CENTER", [0.0, 0.0, 0.0]),
        "radius": float(os.environ.get("RADIUS", "1.0")),
        # LOOK_AT 不设 = 跟随 CENTER（自动定心时 = 自动中心），不能走 parse_vec 的必填分支
        "look_at": parse_vec(look_env, "LOOK_AT") if look_env else None,
        "up": parse_vec(os.environ.get("UP"), "UP", [0.0, 1.0, 0.0]),
        "ease": os.environ.get("EASE", "smoothstep"),
        "r_as_mult": os.environ.get("R_AS_MULT", "1") == "1",
        "az0": float(os.environ.get("AZ0", "0")),
        "el": float(os.environ.get("EL", str(5 if t_type == "arc" else EL))),
        "az_sweep": float(os.environ.get("AZ_SWEEP", str(120 if t_type == "arc" else AZ_SWEEP))),
        "el0": float(os.environ.get("EL0", "5")),
        "el1": float(os.environ.get("EL1", "40")),
        "r0": float(os.environ.get("R0", "1.0")),
        "r1": float(os.environ.get("R1", "0.4")),
    }
    if cfg["look_at"] is None:
        cfg["look_at"] = cfg["center"]      # LOOK_AT 默认 = CENTER（自动定心时 = 自动中心）

    out_dir = Path(os.environ.get("OUT_DIR", str(OUT_DIR)))
    out_dir.mkdir(parents=True, exist_ok=True)

    w = int(os.environ.get("W", "1920"))
    h = int(os.environ.get("H", "1080"))
    fov_y = float(os.environ.get("FOV_Y", "45"))
    # DZ_SENSOR_H 只影响 dolly_zoom 写入的焦距绝对值（作比值基准），不影响画面效果
    fy0 = h / (2.0 * math.tan(math.radians(fov_y) / 2.0))
    fx0 = fy0 * w / h  # 方形像素
    cx, cy = w / 2.0, h / 2.0

    frames, f_scale, look_at, up = gen_frames(cfg)
    n = len(frames)
    dz = any(abs(s - 1.0) > 1e-9 for s in f_scale)

    cam_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
    ]
    if dz:
        # dolly_zoom：每帧一台相机，焦距随距离等比缩放
        for i in range(n):
            cam_lines.append(
                f"{i + 1} PINHOLE {w} {h} {fx0 * f_scale[i]:.6f} {fy0 * f_scale[i]:.6f} {cx:.6f} {cy:.6f}")
    else:
        cam_lines.append(f"1 PINHOLE {w} {h} {fx0:.6f} {fy0:.6f} {cx:.6f} {cy:.6f}")

    img_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] (空行即可，但行必须存在)",
    ]
    for i, pos in enumerate(frames):
        R_w2c, t = look_at_to_w2c(pos, look_at, up)
        qw, qx, qy, qz = rotmat_to_qvec_wxyz(R_w2c)
        cid = i + 1 if dz else 1
        img_lines.append(
            f"{i + 1} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
            f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {cid} frame_{i:06d}.jpg")
        img_lines.append("")  # POINTS2D 行：空，但必须有

    (out_dir / "cameras.txt").write_text("\n".join(cam_lines) + "\n", encoding="utf-8")
    (out_dir / "images.txt").write_text("\n".join(img_lines) + "\n", encoding="utf-8")
    (out_dir / "points3D.txt").write_text(
        "# 3D point list (trajectory file: no points)\n", encoding="utf-8")

    first, last = frames[0], frames[-1]
    print(f"🚀 [99l] TYPE={t_type} N={n} 分辨率 {w}x{h} fov_y={fov_y}° EASE={cfg['ease']}")
    print(f"  📷 相机首帧 ({first[0]:.3f},{first[1]:.3f},{first[2]:.3f}) "
          f"末帧 ({last[0]:.3f},{last[1]:.3f},{last[2]:.3f})")
    if dz:
        print(f"  🔭 dolly_zoom: 焦距缩放 {f_scale[0]:.3f} → {f_scale[-1]:.3f}（每帧一台相机）")
    print(f"✅ 写出 {out_dir}/cameras.txt images.txt points3D.txt")


if __name__ == "__main__":
    main()
