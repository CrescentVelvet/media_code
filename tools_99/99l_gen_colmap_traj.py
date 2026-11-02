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
    python tools_99/99l_gen_colmap_traj.py                    # 预设批量全部 7 种
    EFFECT=orbit_left python ...                              # 单条
    EFFECT=bullet_time ANGLE=120 ELEV=12 python ...            # 子弹时间蒙太奇式运镜
    ANCHOR=cam python ...                                     # 老行为：从初始相机出发
    CENTER=forward python ...                                 # 中心=沿初始相机视线
    CENTER=yaxis PITCH=10 python ...                          # 中心=俯仰看向 y 轴
    INIT_INDEX=-1 GROUP=02 python ...                         # 指定初始相机
    OUT=/tmp/x EFFECT=orbit_left python ...                   # 单条时 --out 即效果目录

七种效果：orbit_left / orbit_right / orbit_up（纯环绕）、dolly_out（拉远）、
hitchcock（拉远 + 焦距同比 = dolly zoom）、grammy（缓起缓收推近）、
bullet_time（绕人环绕 + 推近 + 机位下降，默认快起收尾的变速，参考 VOGUE Ball
子弹时间蒙太奇；EASE/ORBIT_DIR/ELEV 可调）。

两种锚点（ANCHOR）：
- coverage（默认）：先统计拍摄相机的球坐标（方位/仰角/半径），取方位覆盖主段
  （gap-split 后帧数最多那段）的中点作基准方向、主段内仰角与半径的中位数作基准；
  轨迹以它为中心向两侧各展开 ANGLE/2，仰角与半径也从这里出发。ANGLE 超过覆盖
  可用范围时自动截断并 WARN。整圈数据（跨度≈360°）会自动退回初始相机方向。
- cam（老行为）：轨迹首帧 = 初始相机位姿。若初始相机在拍摄覆盖边缘，转 ANGLE
  度会有一半跑到没拍到的方向 → 画面糊/伪影，这就是 coverage 要解决的问题。

Env vars（不设则用下方 main() 预设；多值用逗号分隔）:
  输入   SCENE（场景根或 refined_camera 目录）, GROUP（01|02）, INIT_INDEX（整数，负数=倒数）
  中心   CENTER（"x,y,z" | forward/view | yaxis/y_axis/y轴）, PITCH（yaxis 用，度）
  锚点   ANCHOR（coverage=对准覆盖中心，默认 | cam=从初始相机出发）,
         COVERAGE_GROUP（覆盖统计用哪组，空=跟 GROUP，all=01+02）,
         COVERAGE_PAD（覆盖两侧余量，度，默认 3）
  轨迹   EFFECTS（效果列表）, EFFECT（单条快捷方式）, N, ANGLE（orbit_*/bullet_time 角度°）,
         PARAM（dolly_out/hitchcock 默认 1.5，grammy/bullet_time 默认 2.0）
  变速   EASE（linear|smooth|accel|decel；bullet_time 默认 decel，orbit_* 默认 linear）,
         ORBIT_DIR（bullet_time 环绕方向 right|left）, ELEV（bullet_time 机位下降角，度）
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
               "dolly_out", "hitchcock", "grammy", "bullet_time")
UP = np.array([0.0, 1.0, 0.0])
DOLLY_DEFAULT_M = 1.5
GRAMMY_DEFAULT_M = 2.0
BULLET_DEFAULT_M = 2.0        # bullet_time：末态距离 = 初始 ÷ 2（与 grammy 同口径）


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


# ============ 球坐标（以 center 为心，方位绕 UP 量） ============

def unit_h(v):
    """取向量的水平分量并归一化；水平分量退化（正上/正下）时返回 None。"""
    v = np.asarray(v, dtype=np.float64)
    h = v - float(v @ UP) * UP
    n = float(np.linalg.norm(h))
    return None if n < 1e-9 else h / n


def wrap180(a):
    """角度归一到 (-180, 180]，并把 -180 显示成 +180（同一方向，只是好读）。"""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a <= -180.0 + 1e-9 else a


def az_relative(h, ref_h):
    """h 相对 ref_h 的方位角（度，绕 UP 右手为正）。"""
    return math.degrees(math.atan2(float(np.cross(ref_h, h) @ UP), float(ref_h @ h)))


def dir_from(az_deg, el_deg, ref_h):
    """(方位°, 仰角°) → 单位方向向量（center→cam）。方位相对 ref_h 绕 UP 量。"""
    h = axis_angle_rotmat(UP, math.radians(az_deg)) @ ref_h
    e = math.radians(el_deg)
    return math.cos(e) * h + math.sin(e) * UP


def main_arc(azs, gap_min=30.0):
    """环形 gap-split：反复切掉 > gap_min 的洞，返回**帧数最多**那段 (lo, hi, span)。

    azs 升序（度）。hi 可能 >180（已解缠，保持连续）。为什么不是只切一次：方位是
    环形的，两组数据（如 01/02 各一段弧）之间会有**两个**洞，只切最大洞后剩下的
    序列里横跨另一个洞，中点就落到没拍过的方向上了。整圈数据（最大洞 < gap_min）
    直接返回 span=360。
    """
    n = len(azs)
    if n == 1:
        return azs[0], azs[0], 0.0
    gaps = [((azs[(i + 1) % n] - azs[i]) % 360.0) for i in range(n)]
    k = int(np.argmax(gaps))
    # 整圈判据用「最大洞 vs 中位帧间距」的相对倍数，不能只看绝对阈值：
    # 12 台均匀整圈的洞正好 30°，用 gap_min 卡边界会误判成两段。
    if gaps[k] < max(gap_min, 3.0 * float(np.median(gaps))):
        return azs[0], azs[0] + 360.0, 360.0      # 整圈
    seq, prev = [], None
    for i in range(n):
        a = azs[(k + 1 + i) % n]                  # 从最大洞之后开始绕一圈
        if prev is not None and a < prev - 1e-9:
            a += 360.0                            # 跨 0° 时解缠，保证单调
        seq.append(a)
        prev = a
    segs, cur = [], [seq[0]]
    for a in seq[1:]:
        if a - cur[-1] > gap_min:
            segs.append(cur)
            cur = [a]
        else:
            cur.append(a)
    segs.append(cur)
    best = max(segs, key=len)                     # 帧数最多的一段 = 覆盖最齐的位置
    return best[0], best[-1], best[-1] - best[0]


def on_arc(a, lo, hi):
    """a（度）是否落在已解缠的区间 [lo, hi] 内（含 ±360 的等价写法）。"""
    return any(lo - 1e-6 <= a + d <= hi + 1e-6 for d in (0.0, 360.0, -360.0))


def collect_spherical(imgs, center):
    """每台拍摄相机 → {name, h(水平单位向量), el(度), r(米)}；退化的跳过。"""
    rows = []
    for im in imgs:
        R = qvec2rotmat(im["qvec"])
        C = -R.T @ np.asarray(im["tvec"], dtype=np.float64)
        d = C - center
        r = float(np.linalg.norm(d))
        if r < 1e-9:
            continue
        h = unit_h(d)
        if h is None:
            continue                              # 正上方/正下方：没有方位
        el = math.degrees(math.asin(max(-1.0, min(1.0, float(d @ UP) / r))))
        rows.append({"name": im["name"], "h": h, "el": el, "r": r})
    return rows


def resolve_anchor(rows, ref_h):
    """coverage 锚点：方位覆盖主段中点 + 主段内仰角/半径中位数。

    中位数而非均值：采集里常有单帧离群（手持抖动、跟踪丢帧），均值会被拽走。
    """
    azs = sorted(az_relative(r["h"], ref_h) for r in rows)
    lo, hi, span = main_arc(azs)
    kept = [r for r in rows if on_arc(az_relative(r["h"], ref_h), lo, hi)]
    if not kept:                                  # 理论上不会（lo/hi 就来自这些点）
        kept = rows
    els = np.array([r["el"] for r in kept], dtype=np.float64)
    rs = np.array([r["r"] for r in kept], dtype=np.float64)
    # 整圈时「中点」没有定义（哪个方向都拍到了），退回初始相机方位（相对量 = 0）。
    az_mid = 0.0 if span >= 350.0 else wrap180((lo + hi) / 2.0)
    return {
        "az": az_mid,
        "el": float(np.median(els)),
        "r": float(np.median(rs)),
        "span": span, "az_lo": lo, "az_hi": hi,
        "el_lo": float(els.min()), "el_hi": float(els.max()),
        "r_lo": float(rs.min()), "r_hi": float(rs.max()),
        "n_main": len(kept), "n_all": len(rows),
    }


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


# ============ 七种运镜（输入 C0/center/参数/N，返回 [(pos, fscale), ...]） ============

def ease_progress(u, mode):
    """线性进度 u∈[0,1] → 变速曲线（「子弹时间」那种慢-快-慢靠这个）。

    linear 匀速 | smooth 缓起缓收 | accel 慢起加速 | decel 快起减速（bullet_time 默认）。
    曲线只改时间分布、不改起止位置，所以首末帧与匀速版完全重合。
    """
    if mode in ("linear", "none"):
        return u
    if mode in ("smooth", "smoothstep", "ease_in_out"):
        return u * u * (3.0 - 2.0 * u)
    if mode in ("accel", "ease_in"):
        return u * u
    if mode in ("decel", "ease_out"):
        return 1.0 - (1.0 - u) ** 2
    die(f"未知 EASE {mode!r}，可选: linear/smooth/accel/decel")


def gen_traj(effect, base, ref_h, center, n, angle_deg, param, ease=None,
             orbit_dir="right", elev_deg=0.0, symmetric=False):
    """从锚点 base={az,el,r} 生成一条轨迹，返回 [(pos, fscale), ...]。

    symmetric=True（ANCHOR=coverage）：方位/仰角的扫掠以 base 为中点、向两侧各展开
    angle/2，整段都落在拍摄覆盖内；False（ANCHOR=cam）沿用老行为，从 base 单向走。
    """
    az0 = float(base["az"])
    el0 = float(base["el"])
    r0 = float(base["r"])
    if r0 < 1e-9:
        die("基准半径 ≈ 0（相机与中心点重合）")
    u = np.linspace(0.0, 1.0, n)

    def P(az, el, r):
        """(方位°, 仰角°, 半径) → 世界坐标。"""
        return center + dir_from(az, el, ref_h) * r

    if effect in ("orbit_left", "orbit_right"):
        # 绕 +y 正角度旋转时相机向自身右侧移动，故向左取负号（Rodrigues 已验证）
        sign = -1.0 if effect == "orbit_left" else 1.0
        total = sign * angle_deg
        start = az0 - total / 2.0 if symmetric else az0
        s = ease_progress(u, ease or "linear")
        return [(P(start + total * si, el0, r0), 1.0) for si in s]

    if effect == "orbit_up":
        lim = 89.0
        start = el0 - angle_deg / 2.0 if symmetric else el0
        stop = start + angle_deg
        if start < -lim or stop > lim:
            c0, c1 = max(-lim, start), min(lim, stop)
            warn(f"orbit_up 角度 {angle_deg}° 超出可行仰角范围（±{lim:.0f}°），"
                 f"截断为 {c1 - c0:.1f}°（{c0:.1f}° → {c1:.1f}°）")
            start, stop = c0, c1
        s = ease_progress(u, ease or "linear")
        return [(P(az0, start + (stop - start) * si, r0), 1.0) for si in s]

    if effect in ("dolly_out", "hitchcock"):
        m = DOLLY_DEFAULT_M if param is None else param
        if m <= 0:
            die(f"{effect}: param 必须 > 0，收到 {m}")
        if m < 1:
            warn(f"{effect}: m={m} < 1，拉远变成推近")
        elif abs(m - 1.0) < 1e-9:
            warn(f"{effect}: m≈1，轨迹静止")
        fs = 1.0 + (m - 1.0) * u
        pos = [P(az0, el0, r0 * f) for f in fs]
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
        return [(P(az0, el0, r0 * s), 1.0) for s in ss]

    if effect == "bullet_time":
        # 参考片（VOGUE Ball 子弹时间蒙太奇）实测运镜：同一镜头内 ①绕人水平转动
        # ②半径推近 ③机位下降（26.0s 俯看平台 → 27.25s 平视略仰），速度中段快、
        # 末段收住。三者必须共用同一条进度曲线，否则会变成「先转完再靠近」的
        # 两段动作拼接；曲线分开走也做不出「甩过去再钉住」的观感。
        m = BULLET_DEFAULT_M if param is None else param      # 末态距离 = r0 / m
        if m <= 0:
            die(f"bullet_time: param 必须 > 0（末态距离 = 基准 ÷ param），收到 {m}")
        if m < 1:
            warn(f"bullet_time: m={m} < 1，推近变成拉远")
        elif abs(m - 1.0) < 1e-9:
            warn(f"bullet_time: m≈1，半径不变（退化为纯环绕）")
        sign = -1.0 if orbit_dir == "left" else 1.0
        start = az0 - sign * angle_deg / 2.0 if symmetric else az0
        s = ease_progress(u, ease or "decel")
        out = []
        for si in s:
            el = max(-89.0, min(89.0, el0 - elev_deg * si))   # ELEV 度 = 整段下降量
            r = r0 * (1.0 + (1.0 / m - 1.0) * si)
            out.append((P(start + sign * angle_deg * si, el, r), 1.0))
        return out

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
        group, init_index, out, no_points3d, ease=None, orbit_dir="right",
        elev_deg=0.0, anchor_mode="cam", coverage_group="", coverage_pad=3.0):
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

    # 方位基准取初始相机的水平方向：打印出来的角度天然就是「相对初始相机」的偏移。
    d0v = C0 - center
    r_init = float(np.linalg.norm(d0v))
    if r_init < 1e-9:
        die("初始相机与中心点重合")
    ref_h = unit_h(d0v)
    if ref_h is None:
        die("初始相机在中心点正上/正下方，无法确定方位基准（改 CENTER 或 INIT_INDEX）")
    el_init = math.degrees(math.asin(max(-1.0, min(1.0, float(d0v @ UP) / r_init))))
    print(f"  🎯 中心点({center_mode}): ({fmt(center[0])}, {fmt(center[1])}, "
          f"{fmt(center[2])})，初始距离 {r_init:.3f}")

    anchor = {"az": 0.0, "el": el_init, "r": r_init}      # ANCHOR=cam：老行为
    symmetric = anchor_mode == "coverage"
    st = None
    if symmetric:
        cov_group = (coverage_group or "").strip() or group
        if cov_group.lower() in ("all", "*"):
            cov_imgs, cov_label = imgs, "01+02 全部"
        else:
            cov_imgs = [im for im in imgs if im["name"][:2] == cov_group]
            cov_label = f"{cov_group} 组"
            if not cov_imgs:
                die(f"COVERAGE_GROUP={cov_group} 没有任何相机")
        rows = collect_spherical(cov_imgs, center)
        if len(rows) < 2:
            die(f"覆盖统计只有 {len(rows)} 台可用相机（在中心正上/正下方的被跳过），"
                f"无法定中心；改用 ANCHOR=cam")
        st = resolve_anchor(rows, ref_h)
        anchor = {"az": st["az"], "el": st["el"], "r": st["r"]}
        full = st["span"] >= 350.0
        arc = "整圈覆盖 360°（对准哪都行，方位退回初始相机）" if full else \
              (f"方位 {st['az_lo']:.1f}° ~ {st['az_hi']:.1f}°"
               f"（跨度 {st['span']:.1f}°，主段 {st['n_main']} 台）")
        print(f"  📐 拍摄覆盖（{cov_label}，可用 {st['n_all']} 台）：{arc}")
        print(f"     仰角 {st['el_lo']:.1f}° ~ {st['el_hi']:.1f}°，半径 "
              f"{st['r_lo']:.3f} ~ {st['r_hi']:.3f}")
        print(f"  ⚓ 锚点 coverage：方位 {st['az']:+.1f}°（相对初始相机）、"
              f"仰角 {st['el']:.1f}°、半径 {st['r']:.3f}")
        if full:
            warn("覆盖接近整圈，对齐收益有限（但不会变差）")
        if st["n_main"] < st["n_all"]:
            hint = "" if cov_group.lower() in ("all", "*") else \
                   "；两组都该用的话设 COVERAGE_GROUP=all"
            warn(f"有 {st['n_all'] - st['n_main']} 台相机在覆盖主段之外，"
                 f"已按 gap-split 丢弃{hint}")

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
        ang = angle
        if symmetric:
            # 覆盖对齐只放行「覆盖得住的」幅度：方位看 span，仰角看 el 跨度，
            # 各留 COVERAGE_PAD 余量。截断而非放任，否则等于没解决穿帮。
            what, cap = None, None
            if eff in ("orbit_left", "orbit_right", "bullet_time"):
                what, cap = "方位", st["span"] - 2.0 * coverage_pad
            elif eff == "orbit_up":
                what, cap = "仰角", st["el_hi"] - st["el_lo"] - 2.0 * coverage_pad
            if cap is not None:
                cap = max(0.0, cap)
                if ang > cap + 1e-9:
                    extra = ("；截断为 0° 后轨迹全程静止，"
                             "要强行升降请改 ANCHOR=cam") if cap < 1e-6 else ""
                    warn(f"{eff}: ANGLE={angle:g}° 超出{what}覆盖可用 {cap:.1f}°"
                         f"（覆盖跨度 − 2×COVERAGE_PAD({coverage_pad:g})）{extra}")
                    ang = cap
            if eff == "bullet_time" and elev_deg > 0 and \
                    anchor["el"] - elev_deg < st["el_lo"] - 1e-6:
                warn(f"bullet_time: 末态仰角 {anchor['el'] - elev_deg:.1f}° 低于覆盖下限 "
                     f"{st['el_lo']:.1f}°，画面可能露伪影")
        traj = gen_traj(eff, anchor, ref_h, center, n, ang, param, ease,
                        orbit_dir, elev_deg, symmetric)
        write_effect(dirs[eff], traj, init_cam, center, prefix, points3d_files)
        p0, p1 = traj[0][0], traj[-1][0]
        print(f"  ✅ {eff}: {n} 帧 → {dirs[eff]}")
        if eff == "bullet_time":
            m = BULLET_DEFAULT_M if param is None else param
            print(f"     环绕 {ang:g}°（向{orbit_dir}）+ 推近 {m:.2f}× + 机位下降 "
                  f"{elev_deg:.0f}°，变速 {ease or 'decel'}")
        print(f"     首帧 ({fmt(p0[0])},{fmt(p0[1])},{fmt(p0[2])}) "
              f"末帧 ({fmt(p1[0])},{fmt(p1[1])},{fmt(p1[2])})")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ===== 预设路径与参数，改场景/批次时只改这几行 =====
    SCENE = Path(
        "../../output/2025.11.27-HMSD-Lisidani_out/"
        "action_01/result_gs/2025-11-28--00-27-09---4khq_human"
    )
    EFFECTS = list(EFFECTS_ALL)   # 批量默认全部 7 种；env EFFECT= 可只跑单条
    N = 120                       # 每条轨迹帧数
    CENTER = "0,0,0"              # "x,y,z" | forward/view | yaxis/y_axis/y轴
    PITCH = 10.0                  # CENTER=yaxis 时的俯仰角（度，(-90,90)）
    ANGLE = 90.0                  # orbit_left/right/up/bullet_time 的环绕角度（度）
    PARAM = None                  # None=各效果默认（dolly_out/hitchcock 1.5，grammy/bullet_time 2.0）
    EASE = None                   # 变速曲线：None=bullet_time 用 decel、orbit_* 用 linear
    ORBIT_DIR = "right"           # bullet_time 环绕方向：right | left
    ELEV = 0.0                    # bullet_time 机位下降角（度，0=不升降；参考片同一镜头约 10~15）
    PREFIX = "03"                 # 输出图像名前缀：03000001.jpg
    GROUP = "01"                  # 初始相机组
    INIT_INDEX = None             # 初始相机序号（None=组内中间帧；负数=倒数）
    ANCHOR = "coverage"           # coverage=对准拍摄覆盖中心（默认）| cam=从初始相机出发（老行为）
    COVERAGE_GROUP = ""           # 覆盖统计用哪组：空=跟 GROUP；all=01+02；也可写 01/02
    COVERAGE_PAD = 3.0            # 覆盖两侧余量（度），ANCHOR=coverage 时用于截断角度
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
    ease = env("EASE", EASE)
    orbit_dir = env("ORBIT_DIR", ORBIT_DIR)
    if orbit_dir not in ("left", "right"):
        die(f"ORBIT_DIR 只能是 left | right，收到 {orbit_dir!r}")
    anchor_mode = env("ANCHOR", ANCHOR).strip().lower()
    if anchor_mode not in ("cam", "coverage"):
        die(f"ANCHOR 只能是 cam | coverage，收到 {anchor_mode!r}")
    coverage_pad = float(env("COVERAGE_PAD", str(COVERAGE_PAD)))
    if coverage_pad < 0:
        die(f"COVERAGE_PAD 不能为负，收到 {coverage_pad}")
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
        ease=ease,
        orbit_dir=orbit_dir,
        elev_deg=float(env("ELEV", str(ELEV))),
        anchor_mode=anchor_mode,
        coverage_group=env("COVERAGE_GROUP", COVERAGE_GROUP),
        coverage_pad=coverage_pad,
    )


if __name__ == "__main__":
    main()
