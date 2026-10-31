#!/usr/bin/env python3
"""99l_gen_colmap_traj.py — 读取 3DGS 重建的 COLMAP 相机位姿，生成电影感运镜轨迹。

输入：场景目录（内含 refined_camera/，或直接传 refined_camera/ 目录）。
      refined_camera 含 cameras.txt（PINHOLE / SIMPLE_PINHOLE，支持每帧独立内参）
      与 images.txt（图像名必须匹配 ^(01|02).+\\.(jpe?g)$）。
输出：COLMAP 文本三件套（cameras.txt / images.txt + 拷贝的 points3D），
      目录 <场景根>/traj_camera/<效果名>/，可直接被 99m 渲染。

单条与批量都在本脚本内（不要 bash 批量）：EFFECTS 预设列表默认全部 6 种，
env EFFECT=orbit_left 可只跑单条。

环境准备：Python 3 + numpy（渲染 env 有；不再依赖 plyfile）。

坐标约定（严格遵守，改动必须与 99m 同步验证）：
- 世界系右手系：y 向上、z 向左、x 向后；世界向上轴 UP=(0,1,0)。
- COLMAP 相机系（OpenCV）：x 图像右、y 图像下、z 视线。images.txt 存 world→cam
  的四元数 (QW,QX,QY,QZ) 与平移；相机中心 C=-R^T t，视线 = R 第三行 R[2,:]。
- 四元数↔旋转矩阵用标准 COLMAP 公式（Shepperd，返回归一化四元数）。
- look_at：fwd=normalize(target-pos)；right=normalize(cross(fwd,UP))；
  down=cross(fwd,right)；R=按行堆叠[right;down;fwd]；t=-R@pos。
  ⚠️ 第二个基向量是 down（-up）：OpenCV y 轴向下，用 +up 画面上下颠倒
  （2026-10-31 实测踩过）。

用法:
    python tools_99/99l_gen_colmap_traj.py                    # 预设批量全部 6 种
    EFFECT=orbit_left python ...                              # 单条
    CENTER=forward python ...                                 # 中心=沿初始相机视线
    CENTER=yaxis PITCH=10 python ...                          # 中心=俯仰看向 y 轴
    INIT_INDEX=-1 GROUP=02 python ...                         # 指定初始相机
    OUT=/tmp/x EFFECT=orbit_left python ...                   # 单条时 --out 即效果目录

Env vars（不设则用下方 main() 预设；多值用逗号分隔）:
  输入   SCENE（场景根或 refined_camera 目录）, GROUP（01|02）, INIT_INDEX（整数，负数=倒数）
  中心   CENTER（"x,y,z" | forward/view | yaxis/y_axis/y轴）, PITCH（yaxis 用，度）
  轨迹   EFFECTS（效果列表）, EFFECT（单条快捷方式）, N, ANGLE（orbit_* 角度°）,
         PARAM（dolly_out/hitchcock 默认 1.5，grammy 默认 2.0）
  输出   OUT（单条=效果目录；多条=父目录）, PREFIX（图像名前缀，默认 03）,
         NO_POINTS3D（=1 不拷贝 points3D.txt/.ply）
"""
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

IMG_NAME_RE = re.compile(r"^(01|02).+\.(jpe?g)$", re.IGNORECASE)
EFFECTS_ALL = ("orbit_left", "orbit_right", "orbit_up",
               "dolly_out", "hitchcock", "grammy")
UP = np.array([0.0, 1.0, 0.0])
DOLLY_DEFAULT_M = 1.5
GRAMMY_DEFAULT_M = 2.0


def die(msg):
    sys.exit(f"❌ {msg}")


def warn(msg):
    print(f"  ⚠️ WARN: {msg}")


# ============ 旋转 / 位姿数学（与 99m 的 qvec2rotmat 同式） ============

def qvec2rotmat(q):
    """COLMAP (w,x,y,z) → 3x3 旋转矩阵（与 99b:313 / 99m 同式）。"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def rotmat2qvec(R):
    """3x3 旋转矩阵 → COLMAP 归一化四元数 (w,x,y,z)（Shepperd 最大迹分支）。"""
    t = float(np.trace(R))
    if t > 0.0:
        s = math.sqrt(1.0 + t) * 2.0
        q = np.array([s / 4.0, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, s / 4.0,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      s / 4.0, (R[1, 2] + R[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, s / 4.0])
    return q / np.linalg.norm(q)


def axis_angle_rotmat(axis, theta):
    """Rodrigues：绕轴 axis 转 theta（弧度）。axis 内部归一化。"""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = math.cos(theta), math.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def look_at(pos, target):
    """OpenCV 约定 look-at → world→cam (R, t)。pos/target 为世界系 3 向量。"""
    pos = np.asarray(pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    fwd = target - pos
    d = float(np.linalg.norm(fwd))
    if d < 1e-12:
        die("相机位置与目标重合，无法确定视线方向")
    fwd /= d
    if abs(float(fwd @ UP)) > 1.0 - 1e-9:
        die("look-at 退化：视线与 UP 平行")
    right = np.cross(fwd, UP)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    return R, -R @ pos


# ============ COLMAP 文本解析 ============

def parse_cameras(path):
    """cameras.txt → {id: (w, h, fx, fy, cx, cy)}。只认 PINHOLE / SIMPLE_PINHOLE。"""
    cams = {}
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 4:
            die(f"cameras.txt 第 {ln} 行字段不足: {line!r}")
        cid, model = f[0], f[1]
        w, h = int(f[2]), int(f[3])
        try:
            params = [float(x) for x in f[4:]]
        except ValueError:
            die(f"cameras.txt 第 {ln} 行内参不是数字: {line!r}")
        if model == "PINHOLE" and len(params) == 4:
            fx, fy, cx, cy = params
        elif model == "SIMPLE_PINHOLE" and len(params) == 3:
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            die(f"cameras.txt 第 {ln} 行: 只支持 PINHOLE/SIMPLE_PINHOLE，"
                f"收到 {model}（{len(params)} 参数）")
        cams[cid] = (w, h, fx, fy, cx, cy)
    if not cams:
        die(f"cameras.txt 没有任何相机: {path}")
    return cams


def parse_images(path):
    """images.txt → [{'id', 'qvec', 'tvec', 'cam', 'name'}]，按文件序。

    每条记录两行（位姿行 + POINTS2D 行，后者可为空）。图像名校验
    ^(01|02).+\\.(jpe?g)$（不区分大小写），不匹配即报错。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    imgs = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or not line.strip():
            i += 1
            continue
        f = line.split()
        if len(f) < 10:
            die(f"images.txt 第 {i + 1} 行不是合法位姿行（需 ≥10 字段）: {line!r}")
        try:
            entry = {
                "id": int(f[0]),
                "qvec": [float(x) for x in f[1:5]],
                "tvec": [float(x) for x in f[5:8]],
                "cam": f[8],
                "name": f[9],
            }
        except ValueError:
            die(f"images.txt 第 {i + 1} 行数字解析失败: {line!r}")
        if not IMG_NAME_RE.match(entry["name"]):
            die(f"图像名 {entry['name']!r} 不匹配 ^(01|02).+\\.(jpe?g)$")
        imgs.append(entry)
        i += 2  # 跳过 POINTS2D 行（可为空，下一轮会被 blank 跳过）
    if not imgs:
        die(f"images.txt 没有任何图像: {path}")
    return imgs


def pick_initial(imgs, group, init_index):
    """按名字前 2 字符分组、文件名排序，选初始相机。默认 01 组第 len//2 帧
    （偶数帧数取后半段首帧）；init_index 支持负数（倒数），越界报错。"""
    glist = sorted((im for im in imgs if im["name"][:2] == group),
                   key=lambda im: im["name"])
    if not glist:
        die(f"{group} 组没有任何图像")
    n = len(glist)
    idx = n // 2 if init_index is None else (
        n + init_index if init_index < 0 else init_index)
    if not (0 <= idx < n):
        die(f"init_index={init_index} 越界（{group} 组共 {n} 帧，有效范围 "
            f"[{-n}, {n - 1}]）")
    print(f"  📷 初始相机: {group} 组第 {idx}/{n} 帧 → {glist[idx]['name']}")
    return glist[idx]


# ============ 中心点解析（forward/yaxis 在选定初始相机后才解析） ============

def resolve_center(mode, C0, R0, pitch):
    """mode: "x,y,z" | forward/view | yaxis/y_axis/y轴。"""
    if mode in ("forward", "view"):
        n = float(np.linalg.norm(C0))
        if n < 1e-12:
            die("相机在原点，forward 模式无法定中心（前移距离 = ||C0|| = 0）")
        fwd0 = R0[2, :]
        return C0 + fwd0 / np.linalg.norm(fwd0) * n
    if mode in ("yaxis", "y_axis", "y轴"):
        h = math.hypot(C0[0], C0[2])
        if h < 1e-12:
            die("相机在 y 轴上，yaxis 模式无法定中心（h = hypot(x,z) = 0）")
        if not (-90.0 < pitch < 90.0):
            die(f"pitch={pitch} 越界，须在 (-90, 90) 内")
        return np.array([0.0, C0[1] + h * math.tan(math.radians(pitch)), 0.0])
    try:
        parts = [float(x) for x in str(mode).split(",")]
    except ValueError:
        die(f"CENTER 无法解析为 x,y,z: {mode!r}")
    if len(parts) != 3:
        die(f"CENTER 需要 3 个数（x,y,z）: {mode!r}")
    return np.array(parts)


# ============ 六种运镜（输入 C0/center/参数/N，返回 [(pos, fscale), ...]） ============

def gen_traj(effect, C0, center, n, angle_deg, param):
    d = np.asarray(C0, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    d0 = float(np.linalg.norm(d))
    if d0 < 1e-12:
        die("初始相机与中心点重合")
    dhat = d / d0
    forward0 = -dhat
    u = np.linspace(0.0, 1.0, n)

    if effect in ("orbit_left", "orbit_right"):
        # 绕 +y 正角度旋转时相机向自身右侧移动，故向左取负号（Rodrigues 已验证）
        sign = -1.0 if effect == "orbit_left" else 1.0
        total = sign * math.radians(angle_deg)
        return [(center + axis_angle_rotmat(UP, total * ui) @ d, 1.0) for ui in u]

    if effect == "orbit_up":
        axis = np.cross(UP, forward0)
        na = float(np.linalg.norm(axis))
        if na < 1e-12:
            die("orbit_up 退化：初始视线与 UP 平行")
        axis /= na
        e0 = math.asin(min(1.0, max(-1.0, float(dhat @ UP))))
        lim = math.radians(89.0)
        if e0 >= lim - 1e-6:
            die(f"初始仰角 {math.degrees(e0):.1f}° 已过 89°，无法 orbit_up")
        total = math.radians(angle_deg)
        lo, hi = -(lim - e0), lim - e0
        if total > hi or total < lo:
            clipped = min(max(total, lo), hi)
            warn(f"orbit_up 角度 {angle_deg}° 超出可行范围 "
                 f"({math.degrees(lo):.1f}, {math.degrees(hi):.1f})，截断为 "
                 f"{math.degrees(clipped):.1f}°")
            total = clipped
        return [(center + axis_angle_rotmat(axis, total * ui) @ d, 1.0) for ui in u]

    if effect in ("dolly_out", "hitchcock"):
        m = DOLLY_DEFAULT_M if param is None else param
        if m <= 0:
            die(f"{effect}: param 必须 > 0，收到 {m}")
        if m < 1:
            warn(f"{effect}: m={m} < 1，拉远变成推近")
        elif abs(m - 1.0) < 1e-9:
            warn(f"{effect}: m≈1，轨迹静止")
        fs = 1.0 + (m - 1.0) * u
        pos = [center + dhat * (d0 * f) for f in fs]
        if effect == "dolly_out":
            return [(p, 1.0) for p in pos]
        return [(p, float(f)) for p, f in zip(pos, fs)]  # hitchcock: 焦距同比增大

    if effect == "grammy":
        m = GRAMMY_DEFAULT_M if param is None else param
        if m <= 0:
            die(f"grammy: param 必须 > 0，收到 {m}")
        if m < 1:
            warn(f"grammy: m={m} < 1，推近变成拉远")
        elif abs(m - 1.0) < 1e-9:
            warn(f"grammy: m≈1，轨迹静止")
        ss = u * u * (3.0 - 2.0 * u)                    # smoothstep
        ss = 1.0 + (1.0 / m - 1.0) * ss
        return [(center + dhat * (d0 * s), 1.0) for s in ss]

    die(f"未知效果 {effect!r}，可选: {'/'.join(EFFECTS_ALL)}")


# ============ 输出（COLMAP 文本格式，浮点一律 %.12g） ============

CAMERAS_HEADER = (
    "# Camera list with one line of data per camera:\n"
    "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
    "# Number of cameras: {n}\n"
)
IMAGES_HEADER = (
    "# Image list with two lines of data per image:\n"
    "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
    "# Number of images: {n}, mean observations per image: 0.0\n"
)


def fmt(x):
    return f"{x:.12g}"


def write_effect(out_dir, traj, init_cam, center, prefix, points3d_files):
    """写一条轨迹的 COLMAP 三件套。traj = [(pos, fscale)]；init_cam 提供 W/H/内参。"""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        warn(f"{out_dir} 已存在非空，覆盖写入")
    out_dir.mkdir(parents=True, exist_ok=True)

    w, h, fx0, fy0, cx, cy = init_cam
    cam_lines, img_lines = [], []
    for i, (pos, fscale) in enumerate(traj, 1):
        R, t = look_at(pos, center)          # 姿态一律重新 look-at 中心
        q = rotmat2qvec(R)
        name = f"{prefix}{i:06d}.jpg"
        cam_lines.append(f"{i} PINHOLE {w} {h} {fmt(fx0 * fscale)} {fmt(fy0 * fscale)} "
                         f"{fmt(cx)} {fmt(cy)}")
        img_lines.append(" ".join([str(i)] + [fmt(v) for v in q] +
                                  [fmt(v) for v in t] + [str(i), name]))
        img_lines.append("")                 # 空 POINTS2D 行（必须存在）

    n = len(traj)
    (out_dir / "cameras.txt").write_text(
        CAMERAS_HEADER.format(n=n) + "\n".join(cam_lines) + "\n",
        encoding="utf-8", newline="\n")
    (out_dir / "images.txt").write_text(
        IMAGES_HEADER.format(n=n) + "\n".join(img_lines) + "\n",
        encoding="utf-8", newline="\n")
    for src in points3d_files:
        shutil.copy2(src, out_dir / src.name)


# ============ 主流程 ============

def resolve_refined_dir(scene):
    """--scene 传场景根（内含 refined_camera/）或 refined_camera/ 本身。"""
    scene = Path(scene)
    if (scene / "cameras.txt").exists() and (scene / "images.txt").exists():
        ref, root = scene, scene.parent
    elif (scene / "refined_camera").is_dir():
        ref, root = scene / "refined_camera", scene
    else:
        die(f"{scene} 既不是 refined_camera 目录，也不内含 refined_camera/")
    return ref, root


def run(scene, effects, n, center_mode, pitch, angle, param, prefix,
        group, init_index, out, no_points3d):
    ref, root = resolve_refined_dir(scene)
    cams = parse_cameras(ref / "cameras.txt")
    imgs = parse_images(ref / "images.txt")

    init = pick_initial(imgs, group, init_index)
    R0 = qvec2rotmat(init["qvec"])
    C0 = -R0.T @ np.asarray(init["tvec"], dtype=np.float64)
    if init["cam"] not in cams:
        die(f"初始相机引用的 CAMERA_ID {init['cam']} 不在 cameras.txt 里")
    init_cam = cams[init["cam"]]
    center = resolve_center(center_mode, C0, R0, pitch)
    print(f"  🎯 中心点({center_mode}): ({fmt(center[0])}, {fmt(center[1])}, "
          f"{fmt(center[2])})，初始距离 {float(np.linalg.norm(C0 - center)):.3f}")

    points3d_files = [] if no_points3d else [
        ref / nm for nm in ("points3D.txt", "points3D.ply") if (ref / nm).exists()]

    if out:
        base = Path(out)
        # 单效果：--out 即效果目录；多效果：--out 视为父目录
        dirs = {eff: (base if len(effects) == 1 else base / eff)
                for eff in effects}
    else:
        dirs = {eff: root / "traj_camera" / eff for eff in effects}

    for eff in effects:
        traj = gen_traj(eff, C0, center, n, angle, param)
        write_effect(dirs[eff], traj, init_cam, center, prefix, points3d_files)
        p0, p1 = traj[0][0], traj[-1][0]
        print(f"  ✅ {eff}: {n} 帧 → {dirs[eff]}")
        print(f"     首帧 ({fmt(p0[0])},{fmt(p0[1])},{fmt(p0[2])}) "
              f"末帧 ({fmt(p1[0])},{fmt(p1[1])},{fmt(p1[2])})")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ===== 预设路径与参数，改场景/批次时只改这几行 =====
    SCENE = Path(
        "../../output/2025.11.27-HMSD-Lisidani_out/"
        "action_01/result_gs/2025-11-28--00-27-09---4khq_scene"
    )
    EFFECTS = list(EFFECTS_ALL)   # 批量默认全部 6 种；env EFFECT= 可只跑单条
    N = 120                       # 每条轨迹帧数
    CENTER = "0,0,0"              # "x,y,z" | forward/view | yaxis/y_axis/y轴
    PITCH = 10.0                  # CENTER=yaxis 时的俯仰角（度，(-90,90)）
    ANGLE = 90.0                  # orbit_left/right/up 的角度（度）
    PARAM = None                  # None=各效果默认（dolly_out/hitchcock 1.5，grammy 2.0）
    PREFIX = "03"                 # 输出图像名前缀：03000001.jpg
    GROUP = "01"                  # 初始相机组
    INIT_INDEX = None             # 初始相机序号（None=组内中间帧；负数=倒数）
    OUT = None                    # None=<场景根>/traj_camera；单条=效果目录，多条=父目录
    NO_POINTS3D = False           # True 不拷贝 points3D.txt/.ply
    # ======================================================

    def env(k, d):
        v = os.environ.get(k, "").strip()
        return d if v == "" else v

    effects = env("EFFECT", "")
    if effects:
        effects = [effects]
    else:
        effects = [e.strip() for e in env("EFFECTS", ",".join(EFFECTS)).split(",")
                   if e.strip()]
    for e in effects:
        if e not in EFFECTS_ALL:
            die(f"未知效果 {e!r}，可选: {'/'.join(EFFECTS_ALL)}")

    init_index = env("INIT_INDEX", INIT_INDEX)
    param = env("PARAM", PARAM)
    out = env("OUT", OUT)
    run(
        scene=Path(env("SCENE", str(SCENE))),
        effects=effects,
        n=int(env("N", str(N))),
        center_mode=env("CENTER", str(CENTER)),
        pitch=float(env("PITCH", str(PITCH))),
        angle=float(env("ANGLE", str(ANGLE))),
        param=None if param is None else float(param),
        prefix=env("PREFIX", str(PREFIX)),
        group=env("GROUP", str(GROUP)),
        init_index=None if init_index is None else int(init_index),
        out=None if out is None else Path(out),
        no_points3d=env("NO_POINTS3D", "1" if NO_POINTS3D else "0") == "1",
    )


if __name__ == "__main__":
    main()
