#!/usr/bin/env python3
"""99k_make_gs_params.py — 采集包「主体居中 + 视角区间」→ gs_camera_params_final.json。

背景（为什么需要它）
    Remy / HarmonyOS 采集包（transforms.json + image/*.heic + pcd.ply）的世界原点
    是手机开机点附近，和主体没有关系：实测主体（视线汇聚点）离原点 1.3–2.9 m。
    而 UWA 侧 view_limits 的 target 被写死成 [0,0,0]（见 99b_pack_ply_to_mp4.py:435），
    所以不居中的话，手机端环绕中心会整个偏离主体。
    本脚本做三件事：
      1) 求主体中心（球心），把相机位姿与点云整体平移过去 —— 主体落到原点；
      2) 以主体中心为球心逐帧算 方位角 az / 仰角 el / 距离 r，取实测 min/max 各加余量；
      3) 落盘 gs_camera_params_final.json（99b 的 COLMAP 模式消费它）+ 一份「已居中采集包」。

主体中心取哪个点（VL_TARGET）
    sight  （默认）最小二乘视线汇聚中心：最小化各帧视线到该点的正交距离平方和，
                    即 99g 的 sight_center()，纯标准库 3×3 小系统；
    anchor          transforms.json 自带的 anchor_point（采集包标称锚点）。
    实测 10 个样本（手部微动样例）：sight 的视线垂距 RMS 0.05–0.16 m、夹角均值 2.1–5.7°，
    anchor 0.16–0.58 m、6.3–16.3°，**10/10 都是 sight 更贴合**，故默认 sight（与 99g 一致）。

区间口径（与 99g compute_view_limit 同源，env 名也一致，便于两处对账）
    az 自 +Z 轴起、朝 +X 为正；el 自水平面（+Y 为高度轴）起；r = 相机到球心的 3D 距离。
    解缠：帧序增量归一到 (−180,180] 后累加（慢速小步长、跨 ±180°、绕两圈都成立）。
    整圈：az 跨度 ≥ VL_FULL_AZ（默认 300°）时方位角不设限，写成 [az0, az0+360]。
    余量：az / el 各 ±VL_AZ_PAD / VL_EL_PAD 度，r 乘 (1±VL_R_PAD)。
    换算成 UWA：Phi = −az、Theta = 90 − el、Radius = r（与 99b/99c 的
    pos_to_pitch_yaw 互为逆变换，见 99c_repack_view_limits.py:165-186）。

输入
    <采集包>/transforms.json          必需（逐帧 w/h/cx/cy/fl_x/fl_y/k1..k3/p1/p2
                                      + transform_matrix，c2w / OpenGL 相机系）
    <采集包>/pcd.ply                  可选（落居中包时平移它的 x/y/z）
    <采集包>/image/*.heic             可选（落居中包时挂链，供 01_convert_capture 直接用）

输出（<OUT_ROOT>/<task>/）
    gs_camera_params_final.json   99b 直接消费的相机参数（schema 见下）
    transforms.json               居中后位姿（同名 drop-in，可直接当新采集包用）
    pcd.ply                       居中后点云（binary_little_endian，按属性偏移平移 xyz）
    image/                        指向源图的硬链接（跨卷失败退符号链接，再失败给手工提示）
    center_report.json            球心对照 / 拟合质量 / 区间明细 / 自检结果（诊断用）

gs_camera_params_final.json 的 schema（99b_pack_ply_to_mp4.py:386-415 读的就是这些键）
    {"setting": {"pitch_min", "pitch_max", "yaw_min", "yaw_max", "pitch_init", "yaw_init"},
     "scene":   {"radius"}}
    99b 的用法：maxPhi = yaw_max、minPhi = yaw_min、maxTheta = 90 − pitch_min、
    minTheta = 90 − pitch_max、minRadius = 0.9×radius、maxRadius = 1.2×radius。
    所以 radius 取「让 0.9r/1.2r 恰好罩住实测半径区间」的值（RADIUS_MODE=mid，默认）；
    罩不住（实测半径跨度 > 1.333 倍）时打印警告，可换 max/min/mean/median。

用法
    python tools_99/99k_make_gs_params.py                     # 跑 main() 里的预设批次
    SRC=D:/dataset/手部微动样例/01e954...-佳佳抱枕 python tools_99/99k_make_gs_params.py
    DRY_RUN=1 python tools_99/99k_make_gs_params.py           # 只算不落盘
    ONLY=01e954...,10e3ec... python ...                       # 批次里只跑白名单

Env vars
    SRC            单个采集包目录（或直接给 transforms.json 的路径）；设了就忽略 SRC_ROOT
    SRC_ROOT       批次根目录（其下每个子目录是一个采集包）
    RESULTS_ROOT   统一结果根（默认 ../../output/recon_human_results）
    OUT_ROOT       输出根，默认 <RESULTS_ROOT>/<批次名>/centered
    VL_TARGET      球心口径 sight(默认) | anchor
    VL_AZ_PAD / VL_EL_PAD / VL_R_PAD / VL_FULL_AZ   同 99g（默认 3.0 / 3.0 / 0.08 / 300.0）
    RADIUS_MODE    scene.radius 取法：mid(默认) | mean | median | max | min
                   mid = 让 99b 的 [0.9r,1.2r] 恰好罩住实测半径区间
    SORT_BY_TIME   1(默认) 先按 file_path（时间戳）排序再解缠与取 init 帧；0=保持 json 原序
    PHI_NORMALIZE  1(默认) 把 Phi 区间整体平移 k×360，使区间中心落进 (−180,180]
    OUT_CENTERED / OUT_PLY / OUT_IMAGES   1(默认) 是否落居中位姿 / 居中点云 / 图链
    ONLY / FORCE / DRY_RUN   含义同 99b/99c（ONLY=逗号分隔 task 白名单）

⚠️ 兼容 Python 3.10（服务器 env），别用 3.12+ 才合法的语法；提交前跑
   `python tools_99/99h_check_py_syntax.py` 自查。纯标准库，不需要 numpy。
"""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import sys
from pathlib import Path

# PLY 属性类型 → 字节宽度 / struct 格式（只处理定长标量属性）
PLY_SIZES = {
    "char": 1, "int8": 1, "uchar": 1, "uint8": 1,
    "short": 2, "int16": 2, "ushort": 2, "uint16": 2,
    "int": 4, "int32": 4, "uint": 4, "uint32": 4,
    "float": 4, "float32": 4, "double": 8, "float64": 8,
}
PLY_FMT = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}

# --------------------------------------------------------------------------
# 几何：最小二乘视线汇聚（移植 99g，纯标准库，免 numpy）
# --------------------------------------------------------------------------
def solve3(A, b):
    """3×3 线性方程组，带部分选主元的高斯消元。"""
    M = [list(A[i]) + [b[i]] for i in range(3)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            return None
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        for r in range(3):
            if r == c:
                continue
            f = M[r][c] / pv
            for k in range(c, 4):
                M[r][k] -= f * M[c][k]
    return [M[i][3] / M[i][i] for i in range(3)]


def eig3_sym(M):
    """3×3 实对称矩阵特征值（降序）。循环 Jacobi 旋转，只用来算条件数。"""
    a = [list(row) for row in M]
    for _ in range(60):
        off = abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2])
        if off < 1e-15:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if abs(a[p][q]) < 1e-18:
                continue
            th = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
            c, s = math.cos(th), math.sin(th)
            for k in range(3):
                akp, akq = a[k][p], a[k][q]
                a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
            for k in range(3):
                apk, aqk = a[p][k], a[q][k]
                a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
    return sorted((a[i][i] for i in range(3)), reverse=True)


def sight_normal(pos, look):
    """视线正交距离最小二乘的正规方程 (A, b)。

    A = Σ(I − dᵢdᵢᵀ) 是「视线方向散布张量」：某方向特征值越小 = 视线越少横向指向它。
    全部视线近乎平行时沿视线方向特征值 → 0，A 奇异、深度不可解（此时退回相机质心）。
    """
    A = [[0.0] * 3 for _ in range(3)]
    b = [0.0] * 3
    for c, d in zip(pos, look):
        P = [[(1.0 if i == j else 0.0) - d[i] * d[j] for j in range(3)] for i in range(3)]
        for i in range(3):
            for j in range(3):
                A[i][j] += P[i][j]
            b[i] += sum(P[i][j] * c[j] for j in range(3))
    return A, b


def sight_center(pos, look):
    """最小二乘视线汇聚中心（到各帧视线直线正交距离平方和最小）。

    注意：正交距离对视线方向的**符号不敏感**，所以前向取 +Z 还是 −Z 对解毫无影响；
    真正要小心的是 c2w / w2c —— Remy 的 transform_matrix 已是 c2w，前向 = −R[:,2]。
    误当 w2c 又转置一次不报错、不崩，只是悄悄把残差从 0.20 m 劣化到 0.78 m。
    """
    A, b = sight_normal(pos, look)
    m = solve3(A, b)
    if m is None:   # 视线几乎共线（矩阵奇异）→ 退回相机质心，总比崩掉好
        n = len(pos) or 1
        return tuple(sum(p[i] for p in pos) / n for i in range(3))
    return (m[0], m[1], m[2])


def sight_fit(pos, look, t):
    """候选球心 t 的拟合质量：垂距 RMS / 视线夹角（均值、最大）/ 正规矩阵条件数。"""
    if t is None:
        return None
    acc = 0.0
    angs = []
    for c, d in zip(pos, look):
        e = [t[i] - c[i] for i in range(3)]
        dot = sum(e[i] * d[i] for i in range(3))
        acc += sum((e[i] - dot * d[i]) ** 2 for i in range(3))
        L = math.sqrt(sum(v * v for v in e)) or 1e-9
        angs.append(math.degrees(math.acos(max(-1.0, min(1.0, dot / L)))))
    A, _ = sight_normal(pos, look)
    ev = eig3_sym(A)
    return {
        "resid": math.sqrt(acc / max(1, len(pos))),
        "ang_mean": sum(angs) / len(angs),
        "ang_max": max(angs),
        "cond": (ev[0] / ev[2]) if ev[2] > 1e-12 else float("inf"),
        "lam_min": ev[2],
    }


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------
def load_transforms(path: Path):
    """读采集包 transforms.json，返回 (data, frames)。"""
    if not path.is_file():
        print(f"  ❌ 缺少 {path}")
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        print(f"  ❌ 解析失败 {path}: {exc}")
        return None, None
    frames = data.get("frames") or []
    if not frames:
        print(f"  ❌ {path} 里 frames 为空")
        return None, None
    return data, frames


def frames_to_cloud(frames, sort_by_time: bool = True):
    """逐帧取光心与视线方向。

    transform_matrix 是 c2w / OpenGL：前 3 列 = 相机三轴在世界系的单位向量，
    第 4 列 = 光心；相机前向 = −第 3 列（OpenGL 的 −Z）。
    排序后返回 (pos, look, frames_sorted, sorted_flag)。
    """
    order = sorted(frames, key=lambda f: str(f.get("file_path", ""))) \
        if sort_by_time else list(frames)
    sorted_flag = [str(f.get("file_path", "")) for f in order] != \
        [str(f.get("file_path", "")) for f in frames]
    pos, look = [], []
    for f in order:
        m = f["transform_matrix"]
        pos.append((float(m[0][3]), float(m[1][3]), float(m[2][3])))
        v = (-float(m[0][2]), -float(m[1][2]), -float(m[2][2]))
        L = math.sqrt(sum(c * c for c in v)) or 1.0
        look.append(tuple(c / L for c in v))
    return pos, look, order, sorted_flag


# --------------------------------------------------------------------------
# 视角区间（主体居中后的球壳扇块；算法同 99g compute_view_limit）
# --------------------------------------------------------------------------
def normalize_phi(lo, hi, enabled: bool = True):
    """把 Phi 区间整体平移 k×360，使区间中心落进 (−180,180]。返回 (lo, hi, shift)。

    方位角是周期量：区间 [-204.2, 36.6] 与 [155.8, 396.6] 覆盖的是同一段弧。
    解缠会随轨迹漂移出 (−180,180]，这里做一次规范化，让落盘值可读、也避免播放器
    按字面值解释时跨分支。平移量记进报告，便于回查。
    """
    if not enabled:
        return lo, hi, 0.0
    c = (lo + hi) / 2.0
    shift = 0.0
    while c - shift * 360.0 <= -180.0:
        shift += 1.0
    while c - shift * 360.0 > 180.0:
        shift -= 1.0
    return lo - shift * 360.0, hi - shift * 360.0, shift


def compute_view_limit(pos, look, anchor, cfg: dict):
    """主体中心 + 三组区间。返回 dict（含报告与自检需要的全部中间量）。

    target_mode：sight（默认，最小二乘视线汇聚中心）| anchor（采集包标称锚点）。
    两个候选都算：选中的进 target，另一个进 target_alt 供报告对照。
    """
    mode = cfg["target_mode"]
    sc = sight_center(pos, look)
    if anchor is None:
        target, src, alt = sc, "sight_center", None
        if mode == "anchor":
            print("  ⚠️ 采集包没有 anchor_point，VL_TARGET=anchor 无法生效，改用 sight_center")
    elif mode == "anchor":
        target, src, alt = anchor, "anchor_point", sc
    else:
        target, src, alt = sc, "sight_center", anchor

    az, el, rr = [], [], []
    for p in pos:
        dx, dy, dz = p[0] - target[0], p[1] - target[1], p[2] - target[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz) or 1e-9
        rr.append(d)
        el.append(math.degrees(math.asin(max(-1.0, min(1.0, dy / d)))))
        az.append(math.degrees(math.atan2(dx, dz)))

    # 帧序解缠：每步增量归一到 (−180,180] 后累加（比「只在 |az|>0.8π 时补偿」稳）
    azu = [az[0]]
    for a in az[1:]:
        azu.append(azu[-1] + ((a - azu[-1] + 180.0) % 360.0 - 180.0))

    az_pad, el_pad, r_pad = cfg["az_pad"], cfg["el_pad"], cfg["r_pad"]
    span = max(azu) - min(azu)
    full_az = span >= cfg["full_az"]
    if full_az:   # 绕了一整圈 → 方位角不设限
        az_lo, az_hi = azu[0], azu[0] + 360.0
    else:
        az_lo, az_hi = min(azu) - az_pad, max(azu) + az_pad
    el_lo, el_hi = min(el) - el_pad, max(el) + el_pad
    r_lo = max(0.0, min(rr) * (1.0 - r_pad))
    r_hi = max(rr) * (1.0 + r_pad)

    delta = math.dist(target, alt) if alt else 0.0
    alt_src = ("anchor_point" if src == "sight_center" else "sight_center") if alt else None
    return {
        "target": target, "target_src": src, "target_alt": alt, "target_alt_src": alt_src,
        "delta": delta,
        "fit": sight_fit(pos, look, target), "fit_alt": sight_fit(pos, look, alt),
        "az_lo": az_lo, "az_hi": az_hi, "full_az": full_az, "az_span": span,
        "el_lo": el_lo, "el_hi": el_hi, "r_lo": r_lo, "r_hi": r_hi,
        "raw": {"az": (min(azu), max(azu)), "el": (min(el), max(el)),
                "r": (min(rr), max(rr)), "r_mean": sum(rr) / len(rr)},
        "r_all": rr,
        # 未加余量的 UWA 对照值（Phi = −az、Theta = 90 − el、Radius = r）
        "uwa": {"phi": (-max(azu), -min(azu)),
                "theta": (90.0 - max(el), 90.0 - min(el)),
                "radius": (min(rr), max(rr))},
        "init": {"az": azu[0], "el": el[0], "r": rr[0]},
        "n": len(pos),
    }


def pick_radius(vl: dict, mode: str):
    """scene.radius 的取法。

    99b 由 radius 硬编码展开成 [0.9r, 1.2r]（99b_pack_ply_to_mp4.py:432），
    所以半径这一维真正的约束是「这个带子要罩住实测半径区间」：
      mid    让 [0.9r,1.2r] 恰好罩住实测区间（默认）：r = √[(r_max/1.2)·(r_min/0.9)]
      mean   平均观看距离（实测上端常略超出 1.2r，靠 99c 再收窄）
      median 中位数（抗离群，当前与 mean 同源）
      max / min  取实测最大 / 最小半径
    当 r_max/r_min > 1.2/0.9（≈1.333）时无解，mid 退化为两界中点并在报告里记
    radius_band_covers=False，提示这批数据超出 99b 固定系数的表达能力。
    """
    r_min, r_max = vl["raw"]["r"]
    r_mean = vl["raw"]["r_mean"]
    if mode == "max":
        return r_max, mode
    if mode == "min":
        return r_min, mode
    if mode == "mid":
        lo, hi = r_max / 1.2, r_min / 0.9
        return (math.sqrt(max(lo, 1e-9) * max(hi, 1e-9)) if hi > lo else (lo + hi) / 2.0), mode
    if mode == "median":
        return r_mean, mode
    return r_mean, "mean"


def build_gs_params(vl: dict, cfg: dict):
    """组装 gs_camera_params_final.json 的对象。

    换算（与 99b/99c 一致的权威口径）：
      pitch = el（仰角）      → 99b: maxTheta = 90 − pitch_min、minTheta = 90 − pitch_max
      yaw   = −az = Phi       → 99b: maxPhi = yaw_max、minPhi = yaw_min
      init 取首帧（时间序第一帧），pitch_init/yaw_init 用同一换算。
    """
    phi_lo = -vl["az_hi"]
    phi_hi = -vl["az_lo"]
    phi_lo, phi_hi, shift = normalize_phi(phi_lo, phi_hi, cfg["phi_normalize"])
    # init 的方位角必须跟着同一次平移，否则会落在规范化后的区间之外
    yaw_init = -vl["init"]["az"] - shift * 360.0
    radius, r_mode = pick_radius(vl, cfg["radius_mode"])
    return {
        "setting": {
            "pitch_min": vl["el_lo"], "pitch_max": vl["el_hi"],
            "yaw_min": phi_lo, "yaw_max": phi_hi,
            "pitch_init": vl["init"]["el"],
            "yaw_init": yaw_init,
        },
        "scene": {"radius": radius},
    }, {"phi_shift_deg": shift, "radius_mode": r_mode,
        "phi": (phi_lo, phi_hi), "theta": (90.0 - vl["el_hi"], 90.0 - vl["el_lo"]),
        "radius_band": (0.9 * radius, 1.2 * radius), "yaw_init": yaw_init}


# --------------------------------------------------------------------------
# 落盘：居中位姿 / 居中点云 / 图链
# --------------------------------------------------------------------------
def write_json(path: Path, obj) -> bool:
    """UTF-8 落盘（不转义非 ASCII，task 名常带中文）。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        return True
    except OSError as exc:
        print(f"  ❌ 写文件失败 {path}: {exc}")
        return False


def recenter_transforms(data: dict, frames, target):
    """把采集包的位姿整体平移 −target（世界原点 = 主体中心）。

    只改平移列与 anchor_point，旋转保持不动：刚体平移不改朝向，
    pitch/yaw 因此保持不变，只有半径与角度都变成「相对主体」的。
    """
    tx, ty, tz = target
    out = {k: v for k, v in data.items() if k != "frames"}
    new_frames = []
    for f in frames:
        nf = dict(f)
        m = [list(row) for row in f["transform_matrix"]]
        m[0][3] = float(m[0][3]) - tx
        m[1][3] = float(m[1][3]) - ty
        m[2][3] = float(m[2][3]) - tz
        nf["transform_matrix"] = m
        new_frames.append(nf)
    out["frames"] = new_frames
    if data.get("anchor_point"):
        out["anchor_point"] = [float(v) - t for v, t in zip(data["anchor_point"], target)]
    return out


def recenter_ply(src: Path, target: Path, dst: Path, dry_run: bool = False,
                 chunk_verts: int = 1 << 16):
    """平移 binary_little_endian PLY 的 x/y/z 三列，其余属性逐字节原样保留。

    返回 (status, note)：ok / skip / failed。只处理「第一个 element 是 vertex 且属性全是
    定长标量」的布局（Remy 的 pcd.ply 就是 float xyz + uchar rgb）；ASCII 或含 list 属性的
    布局不猜，直接 skip 并说明，绝不写出半成品。
    """
    try:
        fh = src.open("rb")
    except OSError as exc:
        return "failed", f"打不开 {src}: {exc}"
    with fh:
        header = b""
        while b"end_header" not in header:
            line = fh.readline()
            if not line:
                return "failed", "PLY 头部未以 end_header 结束"
            header += line
            if len(header) > 1 << 20:
                return "failed", "PLY 头部异常大（>1 MiB）"

        elems = []          # [ [name, count, [prop...]], ... ]
        fmt_name = ""
        for raw in header.decode("ascii", "replace").splitlines():
            p = raw.split()
            if not p:
                continue
            if p[0] == "format":
                fmt_name = p[1] if len(p) > 1 else ""
            elif p[0] == "element" and len(p) >= 3:
                elems.append([p[1], int(p[2]), []])
            elif p[0] == "property" and elems:
                if len(p) > 1 and p[1] == "list":
                    elems[-1][2].append(("list",) + tuple(p[2:5]))
                elif len(p) >= 3:
                    elems[-1][2].append(("scalar", p[1], p[2]))

        if fmt_name != "binary_little_endian":
            return "skip", f"只处理 binary_little_endian，实际 {fmt_name or '未知'}"
        if not elems or elems[0][0] != "vertex":
            return "skip", "第一个 element 不是 vertex，布局不认识"

        total, props = elems[0][1], elems[0][2]
        offsets, off = {}, 0
        for pr in props:
            if pr[0] == "list":
                return "skip", "vertex 含 list 属性（变长），不做平移"
            _, ptype, pname = pr
            if ptype not in PLY_SIZES:
                return "skip", f"不认识属性类型 {ptype}"
            offsets[pname] = (off, ptype)
            off += PLY_SIZES[ptype]
        stride = off
        if not {"x", "y", "z"} <= set(offsets):
            return "skip", "vertex 里没有 x/y/z"

        if dry_run:
            return "ok", f"{total} 顶点 × {stride} B（DRY_RUN 未写）"

        nbytes = total * stride
        buf = bytearray()
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("wb") as out:
                out.write(header)
                left = nbytes
                while left > 0:
                    take = min(left, chunk_verts * stride)
                    chunk = fh.read(take)
                    if len(chunk) != take:
                        return "failed", f"顶点数据被截断（还差 {take - len(chunk)} B）"
                    buf[:] = chunk
                    vcount = take // stride
                    for i in range(vcount):
                        base = i * stride
                        for axis, tval in (("x", target[0]), ("y", target[1]), ("z", target[2])):
                            o, ptype = offsets[axis]
                            cur = struct.unpack_from("<" + PLY_FMT[ptype], buf, base + o)[0]
                            struct.pack_into("<" + PLY_FMT[ptype], buf, base + o,
                                             cur - tval)
                    out.write(buf)
                    left -= take
                shutil.copyfileobj(fh, out)   # 其余 element（若有）原样搬运
        except OSError as exc:
            return "failed", f"写 {dst} 失败: {exc}"
    return "ok", f"{total} 顶点的 xyz 已平移"


def link_images(src_dir: Path, dst_dir: Path, dry_run: bool = False):
    """把源 image/ 以硬链接挂进居中包（同卷零拷贝）；跨卷退符号链接；都失败给手工提示。

    ⚠️ 血的教训（2026-10-15，本脚本第一版）：目标已存在**软链接**时，清场循环
    `for p in dst.iterdir(): p.unlink()` 会**穿透链接删掉源目录里的真图**
    （实测把 01e954…佳佳抱枕 的 82 张 HEIC 送进了回收站，已恢复）。
    因此规则是：目标若是链接 → 只 `unlink` 链接本身；若是目录 → 非空就跳过、
    空则 `rmdir`；硬链接阶段只回滚**自己创建过**的文件，绝不按目录内容清理。
    """
    files = sorted(p for p in src_dir.iterdir() if p.is_file()) if src_dir.is_dir() else []
    if not files:
        return "none", "源目录没有图片"
    if dry_run:
        return "ok", f"{len(files)} 张（DRY_RUN 未建链）"

    # ---- 目标已存在：只处理「链接本身」或「空目录」，绝不进目录删内容 ----
    if dst_dir.is_symlink():
        try:
            os.unlink(dst_dir)
        except OSError as exc:
            return "failed", f"目标已是链接且摘不掉: {exc}"
    elif dst_dir.exists():
        if not dst_dir.is_dir():
            return "failed", "目标已存在且不是目录"
        try:
            entries = list(dst_dir.iterdir())
        except OSError as exc:
            return "failed", f"读目标目录失败: {exc}"
        if entries:
            return "skip", f"目标已有 {len(entries)} 个条目，不动"
        try:
            dst_dir.rmdir()
        except OSError as exc:
            return "failed", f"空目录删不掉: {exc}"

    # ---- 1) 硬链接（同卷秒建、零拷贝）----
    try:
        dst_dir.mkdir(parents=True)
    except OSError as exc:
        return "failed", f"建目录失败: {exc}"
    created, link_err = [], ""
    for p in files:
        try:
            os.link(p, dst_dir / p.name)
            created.append(dst_dir / p.name)
        except OSError as exc:
            link_err = str(exc)
            break
    if len(created) == len(files):
        return "hardlink", f"{len(created)} 张硬链接"
    for p in created:   # 只回滚自己建的，不按目录内容清理
        try:
            p.unlink()
        except OSError:
            pass
    try:
        dst_dir.rmdir()
    except OSError:
        pass

    # ---- 2) 符号链接 ----
    try:
        os.symlink(src_dir, dst_dir, target_is_directory=True)
        return "symlink", f"image/ → {src_dir}"
    except OSError as exc:
        print(f"  ⚠️ 图链失败（硬链接: {link_err or '未知'}；软链接: {exc}）")
        print(f'  ℹ️ 手工挂上即可（Windows 免管理员）：mklink /J "{dst_dir}" "{src_dir}"')
        return "none", f"未建 image/ 链（{len(files)} 张源图未挂）"


# --------------------------------------------------------------------------
# 单包处理
# --------------------------------------------------------------------------
def process_one(task_dir: Path, out_dir: Path, cfg: dict) -> str:
    """处理一个采集包，返回 "ok" / "failed"。"""
    print(f"\n================ {task_dir.name} ================")
    data, frames = load_transforms(task_dir / cfg["transforms_name"])
    if data is None:
        return "failed"

    pos, look, ordered, was_sorted = frames_to_cloud(frames, cfg["sort_by_time"])
    if was_sorted:
        print("  ℹ️  frames 原序非时间序，已按 file_path（时间戳）重排后再解缠与取 init 帧")
    anchor = data.get("anchor_point")
    anchor = tuple(float(v) for v in anchor) if anchor else None

    vl = compute_view_limit(pos, look, anchor, cfg)
    gs, info = build_gs_params(vl, cfg)

    # ---------- 居中位姿（先算出来，自检直接验它，而不是验中间量）----------
    centered = recenter_transforms(data, ordered, vl["target"]) if cfg["out_centered"] else None
    if centered is not None:
        d_after = []
        for f in centered["frames"]:
            m = f["transform_matrix"]
            d_after.append(math.sqrt(sum(float(m[k][3]) ** 2 for k in range(3))))
    else:
        d_after = vl["r_all"]
    dist_err = max(abs(a - b) for a, b in zip(d_after, vl["r_all"])) if d_after else 0.0

    # ---------- 自检 ----------
    phi_lo, phi_hi = info["phi"]
    theta_lo, theta_hi = info["theta"]
    yaw_init = info["yaw_init"]
    init_in = (phi_lo - 1e-6 <= yaw_init <= phi_hi + 1e-6
               and vl["el_lo"] - 1e-6 <= vl["init"]["el"] <= vl["el_hi"] + 1e-6)
    band_lo, band_hi = info["radius_band"]
    r_min, r_max = vl["raw"]["r"]
    band_cover = band_lo <= r_min + 1e-9 and band_hi >= r_max - 1e-9

    alt_note = f"  与 {vl['target_alt_src']} 相距 {vl['delta']:.3f} m" if vl["target_alt"] else ""
    print(f"  🎯 球心[{vl['target_src']}] {tuple(round(v, 3) for v in vl['target'])}{alt_note}")
    f, fa = vl["fit"], vl["fit_alt"]
    print(f"  📏 贴合度: 垂距RMS {f['resid']:.3f} m · 视线夹角 均值 {f['ang_mean']:.2f}° "
          f"最大 {f['ang_max']:.2f}° · cond {f['cond']:.2f}"
          + (f"   ｜ 对照: {fa['resid']:.3f} m / {fa['ang_mean']:.2f}°" if fa else ""))
    print(f"  📐 实测: az [{vl['raw']['az'][0]:.1f},{vl['raw']['az'][1]:.1f}] "
          f"span {vl['az_span']:.1f}°{'(整圈)' if vl['full_az'] else ''} ｜ "
          f"el [{vl['raw']['el'][0]:.1f},{vl['raw']['el'][1]:.1f}] ｜ "
          f"r [{r_min:.3f},{r_max:.3f}] mean {vl['raw']['r_mean']:.3f}")
    print(f"  ✂️ 加余量(az±{cfg['az_pad']:g}°/el±{cfg['el_pad']:g}°/r×(1±{cfg['r_pad']:g})): "
          f"Phi [{phi_lo:.2f},{phi_hi:.2f}] ｜ Theta [{theta_lo:.2f},{theta_hi:.2f}] ｜ "
          f"R [{vl['r_lo']:.3f},{vl['r_hi']:.3f}]")
    if band_cover:
        band_note = "  ✅ 罩住实测半径"
    elif cfg["radius_mode"] != "mid":
        band_note = "  ⚠️ 未罩住实测半径，建议 RADIUS_MODE=mid"
    else:
        band_note = (f"  ⚠️ 罩不住：实测半径跨度 {r_max / max(r_min, 1e-9):.2f} 倍 > 1.333，"
                     "99b 的 0.9/1.2 固定系数无法同时兜住两端（已取折中值）")
    print(f"  🔵 scene.radius = {gs['scene']['radius']:.4f} ({info['radius_mode']}) "
          f"→ 99b 展开 [0.9r,1.2r] = [{band_lo:.3f},{band_hi:.3f}]{band_note}")
    if info["phi_shift_deg"]:
        print(f"  ℹ️ Phi 区间已整体平移 {info['phi_shift_deg']:+.0f}×360° 使中心落进 (−180,180]")
    print(f"  📍 init(首帧): pitch {gs['setting']['pitch_init']:.2f}° "
          f"yaw {yaw_init:.2f}° r {vl['init']['r']:.3f}"
          f"{'  ✅ 在区间内' if init_in else '  ⚠️ 落在区间外（请检查数据）'}")
    if dist_err > 1e-9:
        print(f"  ⚠️ 居中后距离校验偏差 {dist_err:.3e} m（应为 0，请检查实现）")

    if cfg["dry_run"]:
        print(f"  ⏭️  DRY_RUN=1，不落盘（目标目录 {out_dir}）")
        return "ok"

    # ---------- 落盘 ----------
    ok = True
    fp_gs = out_dir / cfg["gs_name"]
    ok &= write_json(fp_gs, gs)

    if cfg["out_centered"] and centered is not None:
        ok &= write_json(out_dir / cfg["transforms_name"], centered)
    if cfg["out_ply"]:
        src_ply = task_dir / cfg["ply_name"]
        if src_ply.is_file():
            st, note = recenter_ply(src_ply, vl["target"], out_dir / cfg["ply_name"])
            print(f"  {'✅' if st == 'ok' else ('⏭️' if st == 'skip' else '❌')} 居中点云: {note}")
            ok &= st != "failed"
        else:
            print(f"  ⏭️  源目录没有 {cfg['ply_name']}，跳过点云")
    if cfg["out_images"]:
        st, note = link_images(task_dir / "image", out_dir / "image")
        print(f"  {'✅' if st in ('hardlink', 'symlink') else '⏭️'} 图链: {note}")

    report = {
        "task": task_dir.name, "n_frames": vl["n"],
        "sort_by_time": cfg["sort_by_time"], "input_reordered": was_sorted,
        "target_mode": vl["target_src"], "target": list(vl["target"]),
        "target_alt": list(vl["target_alt"]) if vl["target_alt"] else None,
        "target_alt_src": vl["target_alt_src"],
        "delta_two_candidates": vl["delta"],
        "anchor_point": list(anchor) if anchor else None,
        "fit": vl["fit"], "fit_alt": vl["fit_alt"],
        "measured": {"az": list(vl["raw"]["az"]), "az_span": vl["az_span"],
                     "full_az": vl["full_az"], "el": list(vl["raw"]["el"]),
                     "r": list(vl["raw"]["r"]), "r_mean": vl["raw"]["r_mean"]},
        "emitted": {"minPhi": phi_lo, "maxPhi": phi_hi,
                    "minTheta": theta_lo, "maxTheta": theta_hi,
                    "minRadius": vl["r_lo"], "maxRadius": vl["r_hi"],
                    "radius": gs["scene"]["radius"], "radius_mode": info["radius_mode"],
                    "radius_band_0.9_1.2": list(info["radius_band"]),
                    "radius_span_ratio": r_max / max(r_min, 1e-9),
                    "phi_shift_deg": info["phi_shift_deg"]},
        "init": {"pitch": gs["setting"]["pitch_init"], "yaw": yaw_init,
                 "r": vl["init"]["r"], "inside_limits": init_in},
        "checks": {"center_dist_err_m": dist_err, "radius_band_covers": band_cover},
        "params": {"VL_TARGET": cfg["target_mode"], "VL_AZ_PAD": cfg["az_pad"],
                   "VL_EL_PAD": cfg["el_pad"], "VL_R_PAD": cfg["r_pad"],
                   "VL_FULL_AZ": cfg["full_az"], "RADIUS_MODE": cfg["radius_mode"],
                   "PHI_NORMALIZE": cfg["phi_normalize"]},
    }
    ok &= write_json(out_dir / "center_report.json", report)
    print(f"  {'✅' if ok else '❌'} 输出: {out_dir}")
    return "ok" if ok else "failed"


# --------------------------------------------------------------------------
# 批量入口
# --------------------------------------------------------------------------
def run_batch(src_root: Path, out_root: Path, cfg: dict, only: list) -> None:
    if not src_root.is_dir():
        sys.exit(f"❌ 批次根目录不存在: {src_root}")
    tasks = sorted([d for d in src_root.iterdir()
                    if d.is_dir() and (d / cfg["transforms_name"]).is_file()],
                   key=lambda p: p.name)
    if only:
        tasks = [d for d in tasks if d.name in only]
    if not tasks:
        print(f"⚠️ {src_root} 下没有含 {cfg['transforms_name']} 的子目录")
        return

    print(f"🔍 批次根:   {src_root}")
    print(f"📁 输出根:   {out_root}")
    print(f"🎯 球心口径: {cfg['target_mode']}（sight=最小二乘视线汇聚中心）"
          f" ｜ 余量 az±{cfg['az_pad']:g}° el±{cfg['el_pad']:g}° r×(1±{cfg['r_pad']:g})")
    if cfg["dry_run"]:
        print("⏭️  DRY_RUN=1（只算不落盘）")

    ok = skip = fail = 0
    failed = []
    for task_dir in tasks:
        out_dir = out_root / task_dir.name
        if (out_dir / cfg["gs_name"]).is_file() and not cfg["force"] and not cfg["dry_run"]:
            print(f"\n================ {task_dir.name} ================")
            print(f"⏭️  skip（已存在 {cfg['gs_name']}，FORCE=1 可覆盖）")
            skip += 1
            continue
        status = process_one(task_dir, out_dir, cfg)
        if status == "ok":
            ok += 1
        else:
            fail += 1
            failed.append(task_dir.name)

    print()
    print(f"🎉 Done.  ✅ {ok}  ⏭️ {skip}  ❌ {fail}")
    if failed:
        print(f"❌ 失败列表: {', '.join(failed)}")
    if not cfg["dry_run"]:
        print(f"📁 结果: {out_root}")


def main():
    # ===== 在这里直接改路径（或用环境变量覆盖）=====
    RESULTS_ROOT = Path(os.environ.get(
        "RESULTS_ROOT", "../../output/recon_human_results"))
    # 采集包批次根（本机样例集；服务器上换成实际采集目录）
    SRC_ROOT = Path(os.environ.get("SRC_ROOT", "D:/dataset/手部微动样例"))
    SRC = os.environ.get("SRC", "").strip()      # 单个采集包（或其 transforms.json）
    batch = Path(SRC).parent.name if SRC else SRC_ROOT.resolve().name
    OUT_ROOT = Path(os.environ.get("OUT_ROOT", "")) if os.environ.get("OUT_ROOT") else \
        (RESULTS_ROOT / batch / "centered")
    # ==============================================

    cfg = {
        "transforms_name": os.environ.get("TRANSFORMS_NAME", "transforms.json"),
        "ply_name": os.environ.get("PLY_NAME", "pcd.ply"),
        "gs_name": os.environ.get("GS_NAME", "gs_camera_params_final.json"),
        "target_mode": os.environ.get("VL_TARGET", "sight").strip().lower(),
        "az_pad": float(os.environ.get("VL_AZ_PAD", "3.0")),
        "el_pad": float(os.environ.get("VL_EL_PAD", "3.0")),
        "r_pad": float(os.environ.get("VL_R_PAD", "0.08")),
        "full_az": float(os.environ.get("VL_FULL_AZ", "300.0")),
        "radius_mode": os.environ.get("RADIUS_MODE", "mid").strip().lower(),
        "sort_by_time": os.environ.get("SORT_BY_TIME", "1") == "1",
        "phi_normalize": os.environ.get("PHI_NORMALIZE", "1") == "1",
        "out_centered": os.environ.get("OUT_CENTERED", "1") == "1",
        "out_ply": os.environ.get("OUT_PLY", "1") == "1",
        "out_images": os.environ.get("OUT_IMAGES", "1") == "1",
        "force": os.environ.get("FORCE", "0") == "1",
        "dry_run": os.environ.get("DRY_RUN", "0") == "1",
    }
    only = [s for s in os.environ.get("ONLY", "").split(",") if s]

    # --- 前置检查 ---
    if cfg["target_mode"] not in ("sight", "anchor"):
        sys.exit(f"❌ VL_TARGET 只能是 sight / anchor，实际 {cfg['target_mode']}")
    if cfg["radius_mode"] not in ("mean", "median", "mid", "max", "min"):
        sys.exit(f"❌ RADIUS_MODE 只能是 mean/median/mid/max/min，实际 {cfg['radius_mode']}")
    for k in ("az_pad", "el_pad", "r_pad"):
        if cfg[k] < 0:
            sys.exit(f"❌ 余量参数 {k}={cfg[k]} 不能为负")
    if not 0.0 <= cfg["r_pad"] < 1.0:
        sys.exit(f"❌ VL_R_PAD 需在 [0,1)，实际 {cfg['r_pad']}")

    if SRC:
        p = Path(SRC)
        task_dir = p if p.is_dir() else p.parent
        if not (task_dir / cfg["transforms_name"]).is_file():
            sys.exit(f"❌ {task_dir} 下没有 {cfg['transforms_name']}")
        out_dir = OUT_ROOT / task_dir.name
        print(f"🔍 单个采集包: {task_dir}")
        print(f"📁 输出:       {out_dir}")
        status = process_one(task_dir, out_dir, cfg)
        print(f"\n🎉 Done. {'✅ ok' if status == 'ok' else '❌ failed'}")
        sys.exit(0 if status == "ok" else 1)

    run_batch(SRC_ROOT, OUT_ROOT, cfg, only)


if __name__ == "__main__":
    main()
