#!/usr/bin/env python3
"""99l — 参数化运镜轨迹生成器：输出 COLMAP 文本三件套（cameras/images/points3D）。

与 99g/99k 同为「纯标准库」实现（PLY 自动定心除外，那一步需要 plyfile+numpy，
没有就显式给 CENTER/RADIUS）。输出目录可直接被 99m 渲染、也可喂给 99g/ COLMAP GUI 目检。

轨迹格式约定（99m 按此消费，改格式必须两边同步）：
  * 帧序 = IMAGE_ID 升序（生成器按 1..N 顺序写）；
  * images.txt 每条记录**占两行**（位姿行 + POINTS2D 行，后者可为空）——
    漏第二行是 COLMAP 解析错位的第一大坑；
  * qvec = (qw,qx,qy,qz)，tvec 为世界系→相机系（x_c = R x_w + t）；
  * 只写 PINHOLE 相机；dolly_zoom 逐帧一个 CAMERA_ID（焦距随距离等比缩放，
    主体在画面中的高度恒定），其余运镜共用 CAMERA_ID=1。

Env vars:
  TYPE        orbit | helix | dolly_zoom | push_in | arc（必填）
  OUT_DIR     输出目录（必填）
  PLY         给了就自动算 CENTER（中位数中心）与 RADIUS（q90 距离，AUTO_Q 可调）
  CENTER      "x,y,z"，默认 "0,0,0"（PLY 自动值优先于它）
  RADIUS      相机半径基准（米），默认 1.0（PLY 自动值优先于它）
  N           帧数，默认 120
  W, H        写入 cameras.txt 的分辨率，默认 1920x1080
  FOV_Y       垂直视场角（度），默认 45
  EASE        smoothstep（默认）| linear——决定「电影感」的缓动
  UP          "x,y,z"，默认 "0,1,0"
  LOOK_AT     "x,y,z"，默认 = CENTER（希区柯克变焦的目标点）
  -- orbit / arc --
  EL          相机仰角（度，高于目标），默认 15；arc 默认 5（低角度过肩感）
  AZ0         起始方位角（度），默认 0
  AZ_SWEEP    方位角扫过范围（度），orbit 默认 360，arc 默认 120
  -- helix --
  EL0 / EL1   起/止仰角（度），默认 5 / 40
  AZ_SWEEP    同上，默认 360
  -- push_in / dolly_zoom --
  R0 / R1     起/止相机到目标距离（= RADIUS 的倍数或直接米数，见 R_AS_MULT）
  EL / AZ0    同上（push_in 默认 EL=15；dolly_zoom 默认 EL=0 正对主体）
  R_AS_MULT   1（默认）= R0/R1 是 RADIUS 的倍数；0 = 直接米数
  DZ_SENSOR_H 等效传感器高度（mm 仅作比值基准），默认 24——只影响 dolly_zoom 的
              焦距绝对值，不影响画面效果（主体占比恒定的性质与具体值无关）
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


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


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


def gen_frames():
    """按 TYPE 生成 [(pos, az_unused, extra)]，返回 (frames, per_frame_focal_scale)。

    per_frame_focal_scale：每帧焦距缩放（dolly_zoom 用，其余全 1.0）。
    """
    t_type = os.environ.get("TYPE", "").strip()
    if not t_type:
        die("TYPE 未设置（orbit | helix | dolly_zoom | push_in | arc）")
    n = int(os.environ.get("N", "120"))
    if n < 2:
        die("N 至少 2")

    center = parse_vec(os.environ.get("CENTER"), "CENTER", [0.0, 0.0, 0.0])
    radius = float(os.environ.get("RADIUS", "1.0"))
    ply = os.environ.get("PLY", "").strip()
    if ply:
        c_auto, r_auto = auto_center_radius(Path(ply))
        print(f"🎯 PLY 自动定心: center=({c_auto[0]:.3f},{c_auto[1]:.3f},{c_auto[2]:.3f}) "
              f"radius={r_auto:.3f}")
        center, radius = c_auto, r_auto

    look_at = parse_vec(os.environ.get("LOOK_AT"), "LOOK_AT", center)
    up = parse_vec(os.environ.get("UP"), "UP", [0.0, 1.0, 0.0])
    ease_mode = os.environ.get("EASE", "smoothstep")
    r_as_mult = os.environ.get("R_AS_MULT", "1") == "1"

    def dist(v):
        return v * radius if r_as_mult else v

    az0 = float(os.environ.get("AZ0", "0"))
    frames = []          # (pos, focal_scale)
    f_scale = [1.0] * n

    if t_type in ("orbit", "arc"):
        el = float(os.environ.get("EL", "15" if t_type == "orbit" else "5"))
        sweep = float(os.environ.get("AZ_SWEEP", "360" if t_type == "orbit" else "120"))
        for i in range(n):
            t = ease(i / (n - 1), ease_mode)
            frames.append(sphere_pos(center, dist(1.0), az0 + sweep * t, el))
    elif t_type == "helix":
        el0 = float(os.environ.get("EL0", "5"))
        el1 = float(os.environ.get("EL1", "40"))
        sweep = float(os.environ.get("AZ_SWEEP", "360"))
        for i in range(n):
            t = ease(i / (n - 1), ease_mode)
            frames.append(sphere_pos(center, dist(1.0), az0 + sweep * t, el0 + (el1 - el0) * t))
    elif t_type in ("push_in", "dolly_zoom"):
        el = float(os.environ.get("EL", "15" if t_type == "push_in" else "0"))
        r0, r1 = float(os.environ.get("R0", "1.0")), float(os.environ.get("R1", "0.4"))
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

    return frames, f_scale, look_at, up, n


def main():
    out_dir = Path(os.environ.get("OUT_DIR", "").strip())
    if not out_dir:
        die("OUT_DIR 未设置")
    out_dir.mkdir(parents=True, exist_ok=True)

    w = int(os.environ.get("W", "1920"))
    h = int(os.environ.get("H", "1080"))
    fov_y = float(os.environ.get("FOV_Y", "45"))
    sensor_h = float(os.environ.get("DZ_SENSOR_H", "24"))
    fy0 = h / (2.0 * math.tan(math.radians(fov_y) / 2.0))
    fx0 = fy0 * w / h  # 方形像素
    cx, cy = w / 2.0, h / 2.0

    frames, f_scale, look_at, up, n = gen_frames()
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
    print(f"🚀 [99l] TYPE={os.environ.get('TYPE')} N={n} 分辨率 {w}x{h} fov_y={fov_y}° "
          f"EASE={os.environ.get('EASE', 'smoothstep')}")
    print(f"  📷 相机首帧 ({first[0]:.3f},{first[1]:.3f},{first[2]:.3f}) "
          f"末帧 ({last[0]:.3f},{last[1]:.3f},{last[2]:.3f})")
    if dz:
        print(f"  🔭 dolly_zoom: 焦距缩放 {f_scale[0]:.3f} → {f_scale[-1]:.3f}（每帧一台相机）")
    print(f"✅ 写出 {out_dir}/cameras.txt images.txt points3D.txt")


if __name__ == "__main__":
    main()
