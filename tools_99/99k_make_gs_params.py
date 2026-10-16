#!/usr/bin/env python3
"""99k_make_gs_params.py — 采集包「主体居中 + 视角区间」→ gs_camera_params_final.json。

背景（为什么需要它）
    Remy / HarmonyOS 采集包（transforms.json + image/*.heic + pcd.ply）的世界原点
    是手机开机点附近，和主体没有关系：实测主体（视线汇聚点）离原点 1.3–2.9 m。
    而 UWA 侧 view_limits 的 target 被写死成 [0,0,0]（见 99b_pack_ply_to_mp4.py:435），
    所以不居中的话，手机端环绕中心会整个偏离主体。
    本脚本做四件事：
      1) 求主体中心（球心），把相机位姿与点云整体平移过去 —— 主体落到原点；
      2) 以主体中心为球心逐帧算 方位角 az / 仰角 el / 距离 r，取实测 min/max 各加余量；
      3) 落盘 gs_camera_params_final.json（99b 的 COLMAP 模式消费它）+ 一份「已居中采集包」；
      4) 出图：子进程调同目录的 99g_plot_capture_trajectory.py，把居中后的轨迹与视角约束
         画成自包含 HTML（默认画居中包；PLOT_SRC=origin/both 可加画原始采集包做对照）。

与 99g 的关系（2026-10-15 定：调脚本，不复制代码、不 import）
    99k 的产物「居中包」= transforms.json + pcd.ply，**不碰源图：image/ 既不复制也不挂链**
    （2026-10-16 起），把它当 SRC_ROOT 直接喂给 99g 即可，零适配。出图时 99k 会透传
    `PHOTO_ROOT=<原批次根>`：99g 在本包找不到 image/ 就按**同名 ID 目录**回源包读实拍首帧，
    所以 quad 左栏那张贴图照旧有（源图只读、不动；见 99g 的 PHOTO_ROOT）。之所以不把
    99g 的绘图代码搬进 99k：
      · 99g 是已定稿的汇报图面（13 种风格、1000+ 行绘图与图例口径），复制一份必然分叉，
        以后改图要改两处、还得保证两处同步；
      · import 会把 99k 钉死在 99g 的内部 API（build_ctx / process_one）上，同样脆；
      · 数值口径本就一致，不需要把 99k 的中间量喂给绘图端：
        「居中包上 99g 重算的球心 = 原点」（平移等变性；实测 |球心| ≈ 4.6e-16 m），
        重算出的 Phi/Theta 区间与 99k 写进 json 的**逐样本一致**（Δ ≈ 1e-14°）。
      代价只有「多一个进程」：10 个包 quad 风格实测 4.2 s，可忽略。
    唯一的口径差异（不是错）：99g 图上画的半径带是「实测 r ±VL_R_PAD」（即目标带），
    而 gs json 里的 minRadius/maxRadius 是同一套 padded 值、99b 实际用的却是
    [0.9×radius, 1.2×radius]（见 99b:432）。看 R 环对不上时先想到这一层。
    另：quad 图内的固定文案「左列＝原始采集轨迹（参考点 anchor_point）」里的「原始」
    指**未加约束**，不是说「未居中的包」——画居中包时这句同样出现，别看岔。

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

仰角「空洞」保护（PITCH_MODE=peak，2026-10-15 新增）
    场景：绕物体拍两圈（一圈俯视、一圈平视）时 el 呈**双峰**，两峰之间是没拍到的空档。
    此时 min/max ± pad 必然横跨空档 → 用户转到中间视角就穿帮。
    peak = 对 el 做 gap split（按仰角排序后找最大间隙切段），只保留「一圈」。区间长度由
    数据段的自然边界给出，**不需要长度系数**（按跨度收固定比例够不着洞：洞常占跨度 40%+）：
        thr = max(EL_GAP_MIN, EL_GAP_K × 中位帧间距)   # 相对倍率自适应帧密度 + 绝对下限防误切
        切段 → 丢掉帧数不足的小段 → EL_PICK 选一圈 → 外侧 +VL_EL_PAD、洞侧 +EL_PAD_GAP
    单圈数据不含 ≥thr 的间隙 → 只有一段 → 两侧都走外侧分支 → 结果与 full **逐位相同**，
    所以 peak 是「只对双峰生效」的安全开关（默认 full，历史产物零变化）。
    附带收益：单帧离群（手抖拍到 −60°）会自成一段并被丢掉，而 min/max 对它零抵抗。
    只作用于 el —— 两圈都绕整圈时 az 走 full_az 本来就不设限，无需处理（az 是环形量，
    真要做间隙还得处理解缠跨 ±180，风险大收益小）。
    首帧（开场视角）若落在被丢掉的圈里，pitch_init 会**夹进保留区间**并在报告里记
    init_clamped（不夹的话用户一开场就落在约束外 —— 99g trim_view_limit 记过这条坑）。

输入
    <采集包>/transforms.json          必需（逐帧 w/h/cx/cy/fl_x/fl_y/k1..k3/p1/p2
                                      + transform_matrix，c2w / OpenGL 相机系）
    <采集包>/pcd.ply                  可选（落居中包时平移它的 x/y/z）
    <采集包>/image/*.heic             本脚本不读、不动（居中包只含位姿与点云；出图时
                                      99g 按同名 ID 回源包读实拍首帧，见 PHOTO_ROOT）

输出（<OUT_ROOT>/<task>/）
    gs_camera_params_final.json   99b 直接消费的相机参数（schema 见下）
    transforms.json               居中后位姿（同名 drop-in，可直接当新采集包用）
    pcd.ply                       居中后点云（binary_little_endian，按属性偏移平移 xyz）
    center_report.json            球心对照 / 拟合质量 / 区间明细 / 自检结果（诊断用）

输出（<PLOT_ROOT>/<模式>/，默认 <OUT_ROOT>/../plots）
    <task>.html + .png/.svg       99g 出的自包含图（默认 quad 四联：左列轨迹、右列视角约束）
    index.html                    整批总览页
    模式 = centered（默认，画居中后的包）| origin（原始采集包，做对照）| both
    关掉出图：OUT_PLOTS=0。绘图失败不影响参数产物，只打印告警。

gs_camera_params_final.json 的 schema（4 空格缩进、带换行，便于人工核对）
    {
      "setting": {"init_fov", "yaw_init", "yaw_min", "yaw_max",
                  "pitch_init", "pitch_min", "pitch_max", "roll_init"},
      "camera_type": "pinhole",
      "intrinsics": {"width", "height", "fx", "fy", "cx", "cy"},
      "scene": {"center": [0,0,0], "radius"}
    }
    其中 99b_pack_ply_to_mp4.py:386-415 只消费 setting.pitch_min/pitch_max/yaw_min/yaw_max
    （+ pitch_init 做 1° 一致性告警）与 scene.radius，其余字段是按 3DGS 侧相机文件的既有
    格式补齐的，供人工核对与下游直接消费。99b 的用法：maxPhi = yaw_max、minPhi = yaw_min、
    maxTheta = 90 − pitch_min、minTheta = 90 − pitch_max、minRadius = 0.9×radius、
    maxRadius = 1.2×radius —— 所以 radius 取「让 0.9r/1.2r 恰好罩住实测半径区间」的值
    （RADIUS_MODE=mid，默认）；罩不住（跨度 > 1.333 倍）时打印警告，可换 max/min/mean/median。
    字段取值来源：init_fov = 2·atan(h/(2·fx))·INIT_FOV_SCALE（同 99b:466，默认系数 0.8）；
    intrinsics 取时间序首帧的 w/h/fl_x/fl_y/cx/cy；camera_type 恒 "pinhole"（采集包
    camera_model=OPENCV 的畸变系数不进这个文件，99b/99c 侧也按 PINHOLE 处理）；
    roll_init 恒 0（本链路无滚转）；scene.center 居中后恒 [0,0,0]。

用法
    PITCH_MODE=peak python tools_99/99k_make_gs_params.py      # 两圈数据只保留一圈（避开空洞）
    python ...                                                 # 跑 main() 里的预设批次，默认PITCH_MODE=full
    SRC=D:/dataset/手部微动样例/01e954...-佳佳抱枕 python ...    # 更改输入路径
    DRY_RUN=1 python ...                                       # 只算不落盘
    ONLY=01e954...,10e3ec... python ...                        # 批次里只跑白名单
    PLOT_STYLE=lim_edges python ...                            # 顺便换一种图面风格
    PITCH_MODE=peak EL_PICK=init python ...                    # 保留首帧那一圈
    OUT_PLOTS=0 python ...                                     # 只要参数、不出图

Env vars
    SRC            单个采集包目录（或直接给 transforms.json 的路径）；设了就忽略 SRC_ROOT
    SRC_ROOT       批次根目录（其下每个子目录是一个采集包）
    RESULTS_ROOT   统一结果根（默认 ../../output/recon_human_results）
    OUT_ROOT       输出根，默认 <RESULTS_ROOT>/<批次名>/centered
    VL_TARGET      球心口径 sight(默认) | anchor
    VL_AZ_PAD / VL_EL_PAD / VL_R_PAD / VL_FULL_AZ   同 99g（默认 3.0 / 3.0 / 0.08 / 300.0）
    PITCH_MODE     full(默认) | peak —— peak=仰角 gap split，双峰数据只保留一圈
    EL_GAP_MIN     切段绝对下限（度，默认 8.0）：相邻仰角差 ≥ 它才算「洞」
    EL_GAP_K       切段相对倍率（默认 5.0）：阈值还要 ≥ K×同圈中位帧间距，自适应帧密度
    EL_PICK        peak 模式保留哪一圈：biggest(默认，帧数最多) | top(仰角最高=俯视圈)
                   | bottom(仰角最低=平视圈) | init(含首帧的圈)
    EL_PAD_GAP     peak 模式「洞侧」余量（度，默认 0.0）；外侧仍用 VL_EL_PAD
    EL_TIE_WARN    两圈帧数接近的告警阈值（默认 0.8：次大段/主段 ≥ 它就提示换 EL_PICK）
    EL_MIN_FRAMES  段的最小帧数（默认 5）：不足的段当离群/噪声丢掉
    RADIUS_MODE    scene.radius 取法：mid(默认) | mean | median | max | min
                   mid = 让 99b 的 [0.9r,1.2r] 恰好罩住实测半径区间
    SORT_BY_TIME   1(默认) 先按 file_path（时间戳）排序再解缠与取 init 帧；0=保持 json 原序
    INIT_FOV_SCALE setting.init_fov 的经验系数（默认 0.8，与 99b:466 一致；
                   想要原始竖直视场角就设 1.0）
    PHI_NORMALIZE  1(默认) 把 Phi 区间整体平移 k×360，使区间中心落进 (−180,180]
    OUT_CENTERED / OUT_PLY   1(默认) 是否落居中位姿 / 居中点云（源图一律不复制、不挂链）
    OUT_PLOTS      1(默认) 跑完是否调 99g 出图；0=不出图
    PLOT_SRC       centered(默认) | origin | both —— 画哪个包（居中包 / 原始包 / 两个都画）
    PLOT_STYLE     透传 99g 的 STYLE（默认 quad；可选 combo/minimal/darkspace/fov/iso/
                   vlimit/lim_band/lim_edges/lim_angle/lim_rings/lim_hull/lim_shell/all）
    PLOT_ROOT      图面输出根，默认 <OUT_ROOT>/../plots（OUT_ROOT 名字不是 centered 时
                   用 <OUT_ROOT 名>_plots，避免跨批次覆盖 index.html）
    PLOT_IDS       逗号分隔，只画这些包（留空 = 整批都画，总览页才完整）
    PLOT_PYTHON    调 99g 用的解释器（默认 sys.executable，即跑 99k 的这个）
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
import subprocess
import sys
import time
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
    # 闭式解 shift = ceil((c − 180)/360)，等价于「把中心折算进 (−180, 180]」。
    # ⚠️ 原来写的是 while c - shift*360 <= -180: shift += 1 —— 当 c 恰为 −180 时（绕整圈
    # 且首帧方位角恰为 0° → Phi 区间 (−360, 0] → 中心 −180）条件恒成立，shift 无限递增、
    # 进程死循环挂住（2026-10-15 用合成包实测命中）。闭式解无此风险且结果逐位相同。
    shift = float(math.ceil((c - 180.0) / 360.0))
    return lo - shift * 360.0, hi - shift * 360.0, shift


# --------------------------------------------------------------------------
# 仰角「空洞」保护：gap split（PITCH_MODE=peak）
#   99k 与 99g_plot_capture_trajectory.py 里是**逐字同源**的三份实现（median_of /
#   split_el_segments / pick_el_segment / el_span），env 名也一致 —— 改一处必须同步另一处，
#   否则图（99g 自己重算）与 gs json（99k 写）会不一致。
# --------------------------------------------------------------------------
def median_of(xs):
    """中位数（偶数长度取中间两个的平均）。空列表返回 0.0。"""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def split_el_segments(el, gap_min, gap_k, min_frames):
    """按仰角**排序后相邻样本的间距**切段（gap split）。

    为什么不用直方图投票：投票得先把角度离散成 bin，bin 宽度是纯调参项，而且投完还要再
    拍一个「区间长度系数」；切段的判据直接由数据给出，段的边界就是区间边界，不需要长度系数。

    判据 thr = max(gap_min, gap_k × median(gaps))：
      · median(gaps) = 同一圈内的正常帧间距，洞是它的几十倍 → gap_k（默认 5）即分开；
      · gap_min（默认 8°）是绝对下限 → 防止「稠密圈内部本来就稀疏」时被误切成多段。
    单帧离群（手抖拍到 -60°）会自成一段、再被 min_frames 丢掉；这是附带收益 ——
    原来的 min/max 取法对离群零抵抗。

    返回 (segments, diag)；segments 按 lo 升序，元素为 {"lo","hi","n"}。
    """
    e = sorted(float(v) for v in el)
    n = len(e)
    diag = {"n": n, "median_gap": 0.0, "gap_max": 0.0, "gap_max_span": None,
            "threshold": 0.0, "n_cuts": 0, "n_segments_all": 1, "n_dropped": 0}
    if n == 0:
        return [], diag
    if n == 1:
        return [{"lo": e[0], "hi": e[0], "n": 1}], diag

    gaps = [e[i + 1] - e[i] for i in range(n - 1)]
    med = median_of(gaps)
    thr = max(float(gap_min), float(gap_k) * med)
    cuts = [i for i, g in enumerate(gaps) if g >= thr]
    imax = gaps.index(max(gaps))
    diag.update({"median_gap": med, "gap_max": gaps[imax],
                 "gap_max_span": (e[imax], e[imax + 1]),
                 "threshold": thr, "n_cuts": len(cuts)})

    bounds = [0] + [i + 1 for i in cuts] + [n]
    segs = []
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        if b > a:
            segs.append({"lo": e[a], "hi": e[b - 1], "n": b - a})
    diag["n_segments_all"] = len(segs)

    keep = [s for s in segs if s["n"] >= max(1, int(min_frames))]
    if keep and len(keep) < len(segs):
        diag["n_dropped"] = len(segs) - len(keep)
        segs = keep
    return segs, diag


def pick_el_segment(segments, pick, init_el):
    """从切好的段里选「保留哪一圈」。

    biggest（默认）= 帧数最多的段（次大段接近时 el_span 会置 tie_warn 提醒）
    top            = 仰角最高的段（= 俯视那一圈）
    bottom         = 仰角最低的段（= 平视/仰视那一圈）
    init           = 含首帧的段（首帧若离群不在任何段里，退回 biggest）
    """
    if not segments:
        return None
    if pick == "top":
        return max(segments, key=lambda s: (s["hi"], s["n"]))
    if pick == "bottom":
        return min(segments, key=lambda s: (s["lo"], -s["n"]))
    if pick == "init":
        inside = [s for s in segments if s["lo"] - 1e-9 <= init_el <= s["hi"] + 1e-9]
        if inside:
            return max(inside, key=lambda s: s["n"])
    return max(segments, key=lambda s: s["n"])


def el_span(el, init_el, mode, pad, gap_min, gap_k, min_frames, pick, gap_pad, tie_warn):
    """仰角区间 [lo, hi]（已含余量）+ 诊断 dict。返回 (lo, hi, info)。

    full（默认）= min(el) − pad / max(el) + pad —— 与历史行为逐位一致。
    peak        = gap split 后只保留一圈，"余量分侧给"：
                    · 外侧（贴着全部数据的那一端）用 pad，与 full 同口径；
                    · 洞侧用 gap_pad（默认 0）—— 往洞里加 pad 等于把空洞又拉回来一截。

    单圈数据不存在 ≥thr 的间隙 → 只有一段 → 两侧都走 pad 分支 → 结果与 full **完全相同**，
    所以 peak 是「只对双峰生效」的安全开关。
    """
    lo_raw, hi_raw = min(el), max(el)
    info = {"mode": mode, "raw_span": (lo_raw, hi_raw), "pad_out": pad, "pad_gap": gap_pad,
            "segments": [], "picked": None, "diag": {}, "tie_warn": False, "fallback": None,
            "lo": lo_raw, "hi": hi_raw}
    if mode != "peak":
        info["lo"], info["hi"] = lo_raw - pad, hi_raw + pad
        return info["lo"], info["hi"], info

    segs, diag = split_el_segments(el, gap_min, gap_k, min_frames)
    info["diag"] = diag
    info["segments"] = [[s["lo"], s["hi"], s["n"]] for s in segs]
    picked = pick_el_segment(segs, pick, init_el)
    if picked is None:                     # n ≥ 1 时走不到，防御性回退
        info["fallback"] = "无有效段，退回 min/max"
        info["lo"], info["hi"] = lo_raw - pad, hi_raw + pad
        return info["lo"], info["hi"], info

    info["picked"] = [picked["lo"], picked["hi"], picked["n"]]
    tiny = 1e-9
    lo = picked["lo"] - (gap_pad if picked["lo"] > lo_raw + tiny else pad)
    hi = picked["hi"] + (gap_pad if picked["hi"] < hi_raw - tiny else pad)
    others = [s["n"] for s in segs if s is not picked]
    info["tie_warn"] = bool(others) and max(others) >= tie_warn * max(picked["n"], 1)
    info["lo"], info["hi"] = lo, hi
    return lo, hi, info


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
    # 仰角：PITCH_MODE=peak 时按 gap split 只保留一圈（避开两圈之间没拍到的空洞）
    el_lo, el_hi, pitch = el_span(
        el, el[0], cfg["pitch_mode"], el_pad,
        cfg["el_gap_min"], cfg["el_gap_k"], cfg["el_min_frames"],
        cfg["el_pick"], cfg["el_pad_gap"], cfg["el_tie_warn"])
    # 首帧（99b 的 1° 一致性告警 / 99g 的 trim 兜底锚点）可能落在被丢掉的那一圈里 →
    # 把开场角夹进保留区间，否则用户一开场就落在约束外（99g trim_view_limit 记过这条坑）
    init_el_raw = el[0]
    init_el_used = min(max(init_el_raw, el_lo), el_hi)
    pitch["init_original"] = init_el_raw
    pitch["init_used"] = init_el_used
    pitch["init_clamped"] = abs(init_el_used - init_el_raw) > 1e-9

    r_lo = max(0.0, min(rr) * (1.0 - r_pad))
    r_hi = max(rr) * (1.0 + r_pad)

    delta = math.dist(target, alt) if alt else 0.0
    alt_src = ("anchor_point" if src == "sight_center" else "sight_center") if alt else None
    return {
        "target": target, "target_src": src, "target_alt": alt, "target_alt_src": alt_src,
        "delta": delta,
        "fit": sight_fit(pos, look, target), "fit_alt": sight_fit(pos, look, alt),
        "az_lo": az_lo, "az_hi": az_hi, "full_az": full_az, "az_span": span,
        "el_lo": el_lo, "el_hi": el_hi, "pitch": pitch, "init_el_used": init_el_used,
        "r_lo": r_lo, "r_hi": r_hi,
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


def frame_intrinsics(f0: dict):
    """从采集包首帧抽 pinhole 内参六元组；缺字段返回 None（只在日志里告警，不编数）。

    width/height 输出成整数像素（与 3DGS 侧相机文件的写法一致），fx/fy/cx/cy 保留浮点。
    """
    keys = ("w", "h", "fl_x", "fl_y", "cx", "cy")
    if any(f0.get(k) is None for k in keys):
        return None
    return {"width": int(round(float(f0["w"]))), "height": int(round(float(f0["h"]))),
            "fx": float(f0["fl_x"]), "fy": float(f0["fl_y"]),
            "cx": float(f0["cx"]), "cy": float(f0["cy"])}


def init_fov_deg(intr: dict | None, scale: float):
    """init_camera 的初始视场角（度）。

    与 99b_pack_ply_to_mp4.py:466 同一式：2·atan(h / (2·fx))，再乘同一经验系数
    （INIT_FOV_SCALE，默认 0.8 = 99b 口径；99c 用 INIT_FOV_SCALE=1.25 反向补偿）。
    竖构图下 h 是长边，所以这个数是「竖直视场角」。
    """
    if not intr or not intr["fx"] or not intr["height"]:
        return None
    return math.degrees(2.0 * math.atan(intr["height"] / (2.0 * intr["fx"]))) * scale


def build_gs_params(vl: dict, cfg: dict, ordered: list):
    """组装 gs_camera_params_final.json 的对象。

    换算（与 99b/99c 一致的权威口径）：
      pitch = el（仰角）      → 99b: maxTheta = 90 − pitch_min、minTheta = 90 − pitch_max
      yaw   = −az = Phi       → 99b: maxPhi = yaw_max、minPhi = yaw_min
      init 取首帧（时间序第一帧），pitch_init/yaw_init 用同一换算。
      ⚠️ pitch_init 取的是 **vl["init_el_used"]** 而不是首帧原始仰角：peak 模式下若首帧落在
      被丢弃的那一圈里，原始角会落在 pitch 区间之外，必须夹进来（原始值记在
      center_report 的 pitch_split.init_original）。

    99b 实际只用 setting.pitch_min/pitch_max/yaw_min/yaw_max（+ pitch_init 做 1° 一致性
    告警）与 scene.radius；其余字段按 3DGS 侧相机文件的既有格式补齐，供人工核对与下游
    直接消费（向上游对齐，99b 多读到键也不会受影响）。
    """
    phi_lo = -vl["az_hi"]
    phi_hi = -vl["az_lo"]
    phi_lo, phi_hi, shift = normalize_phi(phi_lo, phi_hi, cfg["phi_normalize"])
    # init 的方位角必须跟着同一次平移，否则会落在规范化后的区间之外
    yaw_init = -vl["init"]["az"] - shift * 360.0
    radius, r_mode = pick_radius(vl, cfg["radius_mode"])

    intr = frame_intrinsics(ordered[0]) if ordered else None
    fov = init_fov_deg(intr, cfg["init_fov_scale"])
    # 居中后主体就在原点；未居中时如实写主体中心在原始世界系里的位置，
    # 好让下游（和 99b 写死的 view_limits target=[0,0,0]）一眼看出对不上。
    center = [0.0, 0.0, 0.0] if cfg["out_centered"] else list(vl["target"])

    return {
        "setting": {
            "init_fov": fov,
            "yaw_init": yaw_init,
            "yaw_min": phi_lo,
            "yaw_max": phi_hi,
            "pitch_init": vl["init_el_used"],
            "pitch_min": vl["el_lo"],
            "pitch_max": vl["el_hi"],
            "roll_init": 0.0,
        },
        "camera_type": "pinhole",
        "intrinsics": intr if intr else {},
        "scene": {"center": center, "radius": radius},
    }, {"phi_shift_deg": shift, "radius_mode": r_mode,
        "phi": (phi_lo, phi_hi), "theta": (90.0 - vl["el_hi"], 90.0 - vl["el_lo"]),
        "pitch": vl["pitch"], "init_clamped": vl["pitch"].get("init_clamped", False),
        "radius_band": (0.9 * radius, 1.2 * radius), "yaw_init": yaw_init,
        "init_fov": fov, "intrinsics": intr, "scene_center": center,
        "camera_type": "pinhole"}


# --------------------------------------------------------------------------
# 落盘：居中位姿 / 居中点云
# --------------------------------------------------------------------------
def write_json(path: Path, obj, indent=None) -> bool:
    """UTF-8 落盘（不转义非 ASCII，task 名常带中文）。

    indent=4 → 带换行缩进（gs_camera_params_final.json / center_report.json 用，
    便于人工核对与 diff）；indent=None → 单行紧凑（transforms.json 用，保持与上游
    采集包一致的书写风格；json.load 不关心空白，两者功能完全等价）。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(obj, ensure_ascii=False, indent=indent)
        # newline="\n"：Windows 上默认会把 \n 翻成 \r\n，而本仓规范是 LF
        # （.gitattributes: * text=auto eol=lf），这里显式压住
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
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
    gs, info = build_gs_params(vl, cfg, ordered)

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
               and vl["el_lo"] - 1e-6 <= vl["init_el_used"] <= vl["el_hi"] + 1e-6)
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
    pk = vl["pitch"]
    if pk["mode"] == "peak":
        if not pk["picked"]:
            print(f"  🗳️ 仰角切段(peak): {pk['fallback'] or '无有效段'}，退回 min/max")
        else:
            seg_txt = " ｜ ".join("[{:.1f},{:.1f}] {}帧".format(s[0], s[1], s[2])
                                  for s in pk["segments"])
            print(f"  🗳️ 仰角切段(peak · EL_PICK={cfg['el_pick']}): "
                  f"{len(pk['segments'])} 段 {seg_txt} → 保留 "
                  f"[{pk['picked'][0]:.1f},{pk['picked'][1]:.1f}]°（{pk['picked'][2]} 帧）")
            d, gsp = pk["diag"], pk["diag"].get("gap_max_span")
            print(f"     判据 thr=max({cfg['el_gap_min']:g}, {cfg['el_gap_k']:g}×中位间距 "
                  f"{d['median_gap']:.2f})={d['threshold']:.2f}° ｜ 最大间隙 "
                  f"{d['gap_max']:.2f}°"
                  + (f"（{gsp[0]:.1f}→{gsp[1]:.1f}°）" if gsp else "")
                  + f" ｜ 切 {d['n_cuts']} 刀"
                  + (f" ｜ 丢弃 {d['n_dropped']} 个小段" if d["n_dropped"] else ""))
            print(f"     余量分侧: 外侧 ±{cfg['el_pad']:g}° ｜ 洞侧 ±{cfg['el_pad_gap']:g}°"
                  f" → pitch [{vl['el_lo']:.2f},{vl['el_hi']:.2f}]"
                  f"（full 模式下为 [{min(vl['raw']['el']) - cfg['el_pad']:.2f},"
                  f"{max(vl['raw']['el']) + cfg['el_pad']:.2f}]）")
            if pk["tie_warn"]:
                print(f"     ⚠️ 两圈帧数接近（次大段 ≥ {cfg['el_tie_warn']:g}×主段）："
                      "biggest 会随帧数抖动，建议显式 EL_PICK=top/bottom")
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
    if pk["init_clamped"]:
        print(f"     ℹ️ 首帧原始仰角 {pk['init_original']:.2f}° 落在保留区间外，"
              f"pitch_init 已夹到 {pk['init_used']:.2f}°"
              f"（要让开场视角留在首帧那一圈 → EL_PICK=init）")
    intr, fov = info["intrinsics"], info["init_fov"]
    if intr:
        print(f"  🔭 内参(首帧): {intr['width']:.0f}×{intr['height']:.0f} "
              f"fx {intr['fx']:.2f} fy {intr['fy']:.2f} "
              f"cx {intr['cx']:.2f} cy {intr['cy']:.2f}"
              + (f" ｜ init_fov {fov:.2f}°(×{cfg['init_fov_scale']:g})" if fov else ""))
    else:
        print("  ⚠️ 首帧缺内参字段（w/h/fl_x/fl_y/cx/cy），intrinsics 与 init_fov 留空")
    if not cfg["out_centered"]:
        print(f"  ⚠️ OUT_CENTERED=0：scene.center 写的是主体中心在原始世界系的位置 "
              f"{tuple(round(v, 3) for v in info['scene_center'])}，"
              f"而 99b 的 view_limits target 写死 [0,0,0]，两边对不上")
    if dist_err > 1e-9:
        print(f"  ⚠️ 居中后距离校验偏差 {dist_err:.3e} m（应为 0，请检查实现）")

    if cfg["dry_run"]:
        print(f"  ⏭️  DRY_RUN=1，不落盘（目标目录 {out_dir}）")
        return "ok"

    # ---------- 落盘 ----------
    ok = True
    fp_gs = out_dir / cfg["gs_name"]
    ok &= write_json(fp_gs, gs, indent=4)

    if cfg["out_centered"] and centered is not None:
        # 紧凑单行：与上游采集包的书写风格一致，json.load 不受影响
        ok &= write_json(out_dir / cfg["transforms_name"], centered)
    if cfg["out_ply"]:
        src_ply = task_dir / cfg["ply_name"]
        if src_ply.is_file():
            st, note = recenter_ply(src_ply, vl["target"], out_dir / cfg["ply_name"])
            print(f"  {'✅' if st == 'ok' else ('⏭️' if st == 'skip' else '❌')} 居中点云: {note}")
            ok &= st != "failed"
        else:
            print(f"  ⏭️  源目录没有 {cfg['ply_name']}，跳过点云")

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
                    "phi_shift_deg": info["phi_shift_deg"],
                    "init_fov": info["init_fov"],
                    "init_fov_scale": cfg["init_fov_scale"],
                    "camera_type": info["camera_type"],
                    "intrinsics": info["intrinsics"],
                    "scene_center": info["scene_center"]},
        "init": {"pitch": gs["setting"]["pitch_init"], "yaw": yaw_init,
                 "r": vl["init"]["r"], "inside_limits": init_in,
                 "pitch_raw_first_frame": pk["init_original"],
                 "pitch_clamped": pk["init_clamped"]},
        "checks": {"center_dist_err_m": dist_err, "radius_band_covers": band_cover},
        "pitch_split": {
            "mode": pk["mode"], "pick": cfg["el_pick"],
            "raw_span": list(pk["raw_span"]),
            "segments": pk["segments"], "picked": pk["picked"],
            "threshold": pk["diag"].get("threshold"),
            "median_gap": pk["diag"].get("median_gap"),
            "gap_max": pk["diag"].get("gap_max"),
            "gap_max_span": (list(pk["diag"]["gap_max_span"])
                             if pk["diag"].get("gap_max_span") else None),
            "n_cuts": pk["diag"].get("n_cuts"),
            "n_dropped": pk["diag"].get("n_dropped"),
            "tie_warn": pk["tie_warn"], "fallback": pk["fallback"],
            "pad_out": pk["pad_out"], "pad_gap": pk["pad_gap"],
            "el_lo": vl["el_lo"], "el_hi": vl["el_hi"],
            "init_original": pk["init_original"], "init_used": pk["init_used"],
            "init_clamped": pk["init_clamped"],
        },
        "params": {"VL_TARGET": cfg["target_mode"], "VL_AZ_PAD": cfg["az_pad"],
                   "VL_EL_PAD": cfg["el_pad"], "VL_R_PAD": cfg["r_pad"],
                   "VL_FULL_AZ": cfg["full_az"], "RADIUS_MODE": cfg["radius_mode"],
                   "PHI_NORMALIZE": cfg["phi_normalize"],
                   "INIT_FOV_SCALE": cfg["init_fov_scale"],
                   "PITCH_MODE": cfg["pitch_mode"], "EL_PICK": cfg["el_pick"],
                   "EL_GAP_MIN": cfg["el_gap_min"], "EL_GAP_K": cfg["el_gap_k"],
                   "EL_PAD_GAP": cfg["el_pad_gap"], "EL_TIE_WARN": cfg["el_tie_warn"],
                   "EL_MIN_FRAMES": cfg["el_min_frames"]},
    }
    ok &= write_json(out_dir / "center_report.json", report, indent=4)
    print(f"  {'✅' if ok else '❌'} 输出: {out_dir}")
    return "ok" if ok else "failed"


# --------------------------------------------------------------------------
# 出图：子进程调同目录的 99g（复用，不复制代码，理由见文件头「与 99g 的关系」）
# --------------------------------------------------------------------------
PLOT_MODES = ("centered", "origin")
PLOT_SCRIPT = "99g_plot_capture_trajectory.py"


def parse_plot_src(raw: str) -> list:
    """PLOT_SRC → 模式列表。both / all 展开成两个模式，也接受逗号组合。"""
    out = []
    for s in (raw or "").split(","):
        s = s.strip().lower()
        if not s:
            continue
        for m in (PLOT_MODES if s in ("both", "all") else (s,)):
            if m not in out:
                out.append(m)
    return out or ["centered"]


def plot_script_path() -> Path:
    """99g 与本脚本同目录；用 __file__ 而不是 cwd，整个目录搬走也不会断。"""
    return Path(__file__).resolve().parent / PLOT_SCRIPT


def run_plots(cfg: dict, src_root: Path, out_root: Path) -> None:
    """把批次出图（每个模式一份 <task>.html/.png + 一个 index.html）。

    centered 模式画的是**居中后的包**（世界原点 = 主体中心），图上球心应落在原点、
    轨迹环绕原点——这是 99k 交付几何的直观校验；origin 模式画原始采集包做对照。
    IDS 默认不传 = 让 99g 扫整个批次（这样 index.html 覆盖完整；PLOT_IDS 可限制）。
    """
    script = plot_script_path()
    if not script.is_file():
        print(f"\n⚠️  找不到 {PLOT_SCRIPT}（应与 99k 同目录），跳过出图")
        return
    exe = cfg["plot_python"] or sys.executable

    print()
    print("-" * 60)
    print(f"🎨 出图：调 {PLOT_SCRIPT}  风格 {cfg['plot_style']}"
          f"  包 {','.join(cfg['plot_src'])}")
    for mode in cfg["plot_src"]:
        root = out_root if mode == "centered" else src_root
        if not root.is_dir():
            extra = "（OUT_CENTERED=0 时没有居中包可画）" if mode == "centered" else ""
            print(f"  ⏭️  [{mode}] 跳过：{root} 不存在{extra}")
            continue
        dst = cfg["plot_root"] / mode
        env = dict(os.environ)
        env.update({
            "SRC_ROOT": str(root),
            "DST_ROOT": str(dst),
            "STYLE": cfg["plot_style"],
            "IDS": ",".join(cfg["plot_ids"]),
            # 视角约束的球心口径与余量常量必须与本次 json 一致，否则图与参数对不上
            "VL_TARGET": cfg["target_mode"],
            "VL_AZ_PAD": f"{cfg['az_pad']:g}",
            "VL_EL_PAD": f"{cfg['el_pad']:g}",
            "VL_R_PAD": f"{cfg['r_pad']:g}",
            "VL_FULL_AZ": f"{cfg['full_az']:g}",
            # 仰角空洞保护也必须透传：否则 99g 会自己算出一版横跨空洞的 el 区间，
            # 图与 gs json 对不上（99g 是独立进程、自己读 transforms.json 重算的）
            "PITCH_MODE": cfg["pitch_mode"],
            "EL_GAP_MIN": f"{cfg['el_gap_min']:g}",
            "EL_GAP_K": f"{cfg['el_gap_k']:g}",
            "EL_MIN_FRAMES": str(cfg["el_min_frames"]),
            "EL_PICK": cfg["el_pick"],
            "EL_PAD_GAP": f"{cfg['el_pad_gap']:g}",
            "EL_TIE_WARN": f"{cfg['el_tie_warn']:g}",
            # 居中包不含源图（99k 不复制、不挂链）→ 告诉 99g 按同名 ID 回原始批次根取
            # 实拍首帧，quad 左栏照旧贴图。源图只被读、不动：见 99g 的 PHOTO_ROOT。
            "PHOTO_ROOT": str(src_root),
            "PYTHONUNBUFFERED": "1",  # pipe 下子进程默认块缓冲，进度行会攒着不吐
        })

        what = "居中包（世界原点=主体中心）" if mode == "centered" else "原始采集包（对照）"
        n_pkg = sum(1 for d in root.iterdir() if d.is_dir() and (d / "transforms.json").exists())
        print(f"  [{mode}] {what} → {dst}")
        print(f"  ⏳ {n_pkg} 个包，每包 ~20-60s（cairosvg 光栅化大 SVG），请耐心等进度行")
        t0 = time.time()
        try:
            # 实时透传 99g 进度：capture_output=True 会把整段进度憋到子进程结束才打，
            # 几十分钟零输出看着像卡死。PYTHONUNBUFFERED 防 pipe 下子进程块缓冲。
            proc = subprocess.Popen([exe, str(script)], env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True,
                                    encoding="utf-8", errors="replace", bufsize=1)
        except OSError as exc:
            print(f"  ❌ 调用 {PLOT_SCRIPT} 失败: {exc}")
            continue
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith(("📁 源:", "📁 输出:")):   # 与上面两行重复
                continue
            if line.strip():
                print(f"    {line}", flush=True)
        err = (proc.stderr.read() or "") if proc.stderr else ""
        proc.wait()
        if proc.returncode != 0:
            err_lines = [s for s in err.splitlines() if s.strip()]
            print(f"  ❌ 出图[{mode}] 失败（退出码 {proc.returncode}）"
                  + (f"：{err_lines[-1]}" if err_lines else ""))
            print(f"     常见原因：PLOT_STYLE 写法不在 99g 支持列表、批次下没有含 "
                  f"transforms.json 的包。单独重跑："
                  f"STYLE={cfg['plot_style']} SRC_ROOT=\"{root}\" DST_ROOT=\"{dst}\" "
                  f"python tools_99/{PLOT_SCRIPT}")
        else:
            print(f"  ✅ 出图[{mode}] 完成（{time.time() - t0:.1f}s）")

    if cfg["plot_ids"]:
        print(f"  ℹ️ PLOT_IDS 只画了 {len(cfg['plot_ids'])} 个包，index.html 也只列这些；"
              "要整批总览页就不设 PLOT_IDS 再跑一次")
    print(f"📁 图面: {cfg['plot_root']}")


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
          f" ｜ 余量 az±{cfg['az_pad']:g}° el±{cfg['el_pad']:g}° r×(1±{cfg['r_pad']:g})"
          + (f" ｜ 仰角 PITCH_MODE=peak(EL_PICK={cfg['el_pick']}，gap≥"
             f"{cfg['el_gap_min']:g}°/洞侧余量{cfg['el_pad_gap']:g}°)"
             if cfg["pitch_mode"] == "peak" else " ｜ 仰角 PITCH_MODE=full"))
    if cfg["dry_run"]:
        print("⏭️  DRY_RUN=1（只算不落盘）")
    if cfg["out_plots"]:
        print(f"🎨 出图根:   {cfg['plot_root']}  ← {PLOT_SCRIPT}"
              f"（PLOT_SRC={','.join(cfg['plot_src'])} STYLE={cfg['plot_style']}）")

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
        if cfg["out_plots"]:
            run_plots(cfg, src_root, out_root)


def main():
    # ===== 在这里直接改路径（或用环境变量覆盖）=====
    # 采集包批次根（本机样例集；服务器上换成实际采集目录）
    SRC_ROOT = Path(os.environ.get(
        "SRC_ROOT",
        # "D:/dataset/手部微动样例",
        "../../code/Reconstruction/dataset/scene_B004_0824_data-物体输入图像",
    ))
    # 输出目录（HTML 很小，放哪儿都行）
    RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", "../../output/recon_human_results"))
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
        # --- 仰角「空洞」保护 gap split（99g 同名 env，改一处要同步另一处）---
        "pitch_mode": os.environ.get("PITCH_MODE", "full").strip().lower(),
        "el_gap_min": float(os.environ.get("EL_GAP_MIN", "8.0")),
        "el_gap_k": float(os.environ.get("EL_GAP_K", "5.0")),
        "el_min_frames": int(os.environ.get("EL_MIN_FRAMES", "5")),
        "el_pick": os.environ.get("EL_PICK", "biggest").strip().lower(),
        "el_pad_gap": float(os.environ.get("EL_PAD_GAP", "0.0")),
        "el_tie_warn": float(os.environ.get("EL_TIE_WARN", "0.8")),
        "radius_mode": os.environ.get("RADIUS_MODE", "mid").strip().lower(),
        "init_fov_scale": float(os.environ.get("INIT_FOV_SCALE", "0.8")),
        "sort_by_time": os.environ.get("SORT_BY_TIME", "1") == "1",
        "phi_normalize": os.environ.get("PHI_NORMALIZE", "1") == "1",
        "out_centered": os.environ.get("OUT_CENTERED", "1") == "1",
        "out_ply": os.environ.get("OUT_PLY", "1") == "1",
        "force": os.environ.get("FORCE", "0") == "1",
        "dry_run": os.environ.get("DRY_RUN", "0") == "1",
        # --- 出图（复用 99g，见 run_plots）---
        "out_plots": os.environ.get("OUT_PLOTS", "1") == "1",
        "plot_src": parse_plot_src(os.environ.get("PLOT_SRC", "centered")),
        "plot_style": os.environ.get("PLOT_STYLE", "quad").strip() or "quad",
        "plot_ids": [s for s in os.environ.get("PLOT_IDS", "").split(",") if s],
        "plot_python": os.environ.get("PLOT_PYTHON", "").strip(),
        # 图面根：默认 OUT_ROOT=<批次>/centered → 放平级的 <批次>/plots；
        # OUT_ROOT 被自定义成别的名字时退化为 <OUT_ROOT 名>_plots，免得两个批次
        # 的图挤进同一个 plots/ 把 index.html 互相覆盖。
        "plot_root": (Path(os.environ["PLOT_ROOT"]) if os.environ.get("PLOT_ROOT")
                      else OUT_ROOT.parent / ("plots" if OUT_ROOT.name == "centered"
                                              else OUT_ROOT.name + "_plots")),
    }
    only = [s for s in os.environ.get("ONLY", "").split(",") if s]

    # --- 前置检查 ---
    if cfg["target_mode"] not in ("sight", "anchor"):
        sys.exit(f"❌ VL_TARGET 只能是 sight / anchor，实际 {cfg['target_mode']}")
    if cfg["radius_mode"] not in ("mean", "median", "mid", "max", "min"):
        sys.exit(f"❌ RADIUS_MODE 只能是 mean/median/mid/max/min，实际 {cfg['radius_mode']}")
    if cfg["pitch_mode"] not in ("full", "peak"):
        sys.exit(f"❌ PITCH_MODE 只能是 full / peak，实际 {cfg['pitch_mode']}")
    if cfg["el_pick"] not in ("biggest", "top", "bottom", "init"):
        sys.exit(f"❌ EL_PICK 只能是 biggest / top / bottom / init，实际 {cfg['el_pick']}")
    if cfg["el_gap_min"] < 0 or cfg["el_gap_k"] < 0 or cfg["el_pad_gap"] < 0:
        sys.exit("❌ EL_GAP_MIN / EL_GAP_K / EL_PAD_GAP 不能为负（实际 "
                 f"{cfg['el_gap_min']:g} / {cfg['el_gap_k']:g} / {cfg['el_pad_gap']:g}）")
    if cfg["el_min_frames"] < 1:
        sys.exit(f"❌ EL_MIN_FRAMES 需 ≥ 1，实际 {cfg['el_min_frames']}")
    if not 0.0 <= cfg["el_tie_warn"] <= 1.0:
        sys.exit(f"❌ EL_TIE_WARN 需在 [0,1]，实际 {cfg['el_tie_warn']}")
    if cfg["init_fov_scale"] <= 0:
        sys.exit(f"❌ INIT_FOV_SCALE 需 > 0，实际 {cfg['init_fov_scale']}")
    for k in ("az_pad", "el_pad", "r_pad"):
        if cfg[k] < 0:
            sys.exit(f"❌ 余量参数 {k}={cfg[k]} 不能为负")
    if not 0.0 <= cfg["r_pad"] < 1.0:
        sys.exit(f"❌ VL_R_PAD 需在 [0,1)，实际 {cfg['r_pad']}")
    bad_modes = [m for m in cfg["plot_src"] if m not in PLOT_MODES]
    if bad_modes:
        sys.exit(f"❌ PLOT_SRC 只能是 {'/'.join(PLOT_MODES)} 或逗号组合，"
                 f"实际 {'/'.join(bad_modes)}")

    if SRC:
        p = Path(SRC)
        task_dir = p if p.is_dir() else p.parent
        if not (task_dir / cfg["transforms_name"]).is_file():
            sys.exit(f"❌ {task_dir} 下没有 {cfg['transforms_name']}")
        out_dir = OUT_ROOT / task_dir.name
        print(f"🔍 单个采集包: {task_dir}")
        print(f"📁 输出:       {out_dir}")
        status = process_one(task_dir, out_dir, cfg)
        if status == "ok" and cfg["out_plots"]:
            run_plots(cfg, task_dir.parent, OUT_ROOT)
        print(f"\n🎉 Done. {'✅ ok' if status == 'ok' else '❌ failed'}")
        sys.exit(0 if status == "ok" else 1)

    run_batch(SRC_ROOT, OUT_ROOT, cfg, only)


if __name__ == "__main__":
    main()
