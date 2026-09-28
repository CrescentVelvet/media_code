#!/usr/bin/env python3
"""99g_plot_capture_trajectory.py — 把 Remy / HarmonyOS 采集包（transforms.json）的
相机轨迹画成自包含 HTML（内联 SVG），图内标题与文件名都用「文件夹 ID」。

背景（2026-09-22 实测核对过，详见 NOTES.md）
    Remy 采集包每个 ID 目录下是 transforms.json + image/*.heic + pcd.ply。
    · transform_matrix 是 camera-to-world：前 3 列 = 相机三轴在世界系的单位向量，
      第 4 列 = 相机光心（米）。世界系 +Y 向上（重力对齐），相机前向 = -Z（OpenGL 约定，
      注意 camera_model:"OPENCV" 只描述畸变参数集，不决定外参轴向）。
    · anchor_point = 环绕轴心；实测各帧到它的水平距离近似恒定 → 图里画成均值半径环。

只依赖标准库（json/math/struct），服务器与 WSL 都免装包。

⚠️ 兼容 Python 3.8+（服务器 vggt_human env 是 3.10，本机是 3.13）。**别用 3.12+ 才合法的语法**，
尤其 f-string 表达式段里复用同类引号——3.10 会直接 SyntaxError，本机 3.13 却看不出来。
提交前跑 `python vggt_human/99h_check_py_syntax.py` 自查。

用法:
    python vggt_human/99g_plot_capture_trajectory.py
        # 默认 STYLE=quad：四联对照图 2×2（左列＝原始采集轨迹俯视/等轴测，
        # 右列＝同取景下的视角约束：俯视角度标注 + 等轴测 3D 球壳）
    STYLE=combo python ...   # 只要原始双联图（无约束层）
    STYLE=all python ...     # 一次出全部风格（<ID>__<style>.html），便于挑图
    SRC_ROOT=... DST_ROOT=... python ...
        # 本机测试数据集在 D:/dataset/测试数据sample（脚本默认值是服务器路径）：
        # SRC_ROOT=D:/dataset/测试数据sample DST_ROOT=C:/code/output/xxx python vggt_human/99g_plot_capture_trajectory.py

    # 视角约束范围（view limit）：先看六种画法的总览再定稿
    STYLE=vlimit SRC_ROOT=... python ...
    # 定稿后只出一种、单独出大图 + 独立 .svg
    STYLE=lim_edges SRC_ROOT=... python ...
    # 把某一种约束叠进 combo 的俯视图（shell 叠进等轴测面板）
    VLIMIT_MODE=edges STYLE=combo SRC_ROOT=... python ...

    # 可选：只画其中几个 ID
    IDS=13a8ecadfeb448e890db319ac828befe,10e3ec6291c04bedaad735309e4bc43b STYLE=all python ...

路径与默认风格写在下方 main() 里。每张图旁会落一份同名独立 .svg（矢量原图，可拖进 PPT）。
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
from pathlib import Path

if sys.version_info < (3, 8):
    sys.exit("❌ 需要 Python ≥ 3.8（用到 math.dist）；服务器 vggt_human env 是 3.10")

# 读盘时最多保留多少个点（用于取 bbox 与后续按视野过滤）
PLY_MAX_POINTS = 300000
# 单块图最多画多少个点：先按「视野」过滤再限流。只做全局抽稀的话，
# 视野被放大时落在视野内的点只剩百分之几，点云会稀到看不见。
PLY_DRAW_POINTS = 30000

STYLES = ("quad", "combo", "minimal", "darkspace", "fov", "iso",
          "vlimit", "lim_band", "lim_edges", "lim_angle", "lim_rings", "lim_hull", "lim_shell")

STYLE_LABELS = {
    "quad": "四联对照图 2×2（左列原始采集轨迹 / 右列视角约束，共用取景）",
    "combo": "俯视 + 等轴测双联图（上下排列，俯视轨迹按帧序时间渐变）",
    "minimal": "浅色极简 · 轨迹 + 视线 + 均值半径环",
    "darkspace": "深色网格 · 按时间渐变的轨迹",
    "fov": "视锥扇形 + 点云底图（浅色）",
    "iso": "等轴测立体 · 含高度与垂直投影线",
    "vlimit": "视角约束范围 · 六种画法总览（一张页内竖向排开，用于挑图）",
    "lim_band": "视角约束 · 扇环填充（半透明，压在轨迹下层）",
    "lim_edges": "视角约束 · 四条边界虚线（零填充，对轨迹影响最小）",
    "lim_angle": "视角约束 · 边界虚线 + 角度/半径数值标注",
    "lim_rings": "视角约束 · 米制同心环 + 方位刻度（极坐标纸）",
    "lim_hull": "视角约束对照组 · 相机水平位置的凸包外扩（采集包络）",
    "lim_shell": "视角约束 · 等轴测 3D 球壳扇块（立体感最强）",
}

# 等轴测投用的视角因子
ISO_COS, ISO_SIN = math.cos(math.radians(30)), math.sin(math.radians(30))

FONT = "system-ui,-apple-system,'Segoe UI','Microsoft YaHei',sans-serif"
MONO = "ui-monospace,Consolas,'Courier New',monospace"


# ─────────────────────────── 数据加载 ───────────────────────────

def load_transforms(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not data.get("frames"):
        sys.exit(f"❌ frames 为空: {path}")
    return data


def load_ply_points(path: Path, max_points: int = PLY_MAX_POINTS):
    """读 binary_little_endian 的 xyz 点云，返回 (抽稀点列表, 总点数, 全量 bbox)。

    只支持 Remy 这种定点布局（float xyz），其它布局直接跳过而不是猜。
    bbox 用「全部点」统计（不是抽稀后的子集），否则极值点会被抽掉导致统计偏差。
    """
    if not path.exists():
        return [], None, None
    with path.open("rb") as fh:
        header = b""
        while b"end_header" not in header:
            chunk = fh.readline()
            if not chunk:
                return [], None, None
            header += chunk
        raw = fh.read()

    total, props = None, []
    for line in header.decode("ascii", "replace").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
            total = int(parts[2])
        elif len(parts) == 3 and parts[0] == "property":
            props.append((parts[1], parts[2]))

    if total is None or not props:
        return [], None, None
    size_of = {"float": 4, "float32": 4, "double": 8, "uchar": 1, "uint8": 1}
    if any(t not in size_of for t, _ in props):
        print("⚠️ 点云含不支持的属性类型，跳过点云")
        return [], total, None

    offsets, off = {}, 0
    for t, name in props:
        offsets[name] = (off, t)
        off += size_of[t]
    stride = off
    if len(raw) < total * stride or not {"x", "y", "z"} <= set(offsets):
        return [], total, None

    step = max(1, total // max_points)
    pts, lo, hi = [], [math.inf] * 3, [-math.inf] * 3
    for i in range(total):
        base = i * stride
        x = struct.unpack_from("<f", raw, base + offsets["x"][0])[0]
        y = struct.unpack_from("<f", raw, base + offsets["y"][0])[0]
        z = struct.unpack_from("<f", raw, base + offsets["z"][0])[0]
        if not all(math.isfinite(v) for v in (x, y, z)):
            continue
        for k, v in enumerate((x, y, z)):
            if v < lo[k]:
                lo[k] = v
            if v > hi[k]:
                hi[k] = v
        if i % step == 0 and len(pts) < max_points:
            pts.append((x, y, z))
    bbox = (tuple(lo), tuple(hi)) if lo[0] < math.inf else None
    return pts, total, bbox


def build_ctx(folder: str, data: dict, points, ply_total, point_bbox=None):
    """把原始 json 整理成绘图用的数组 + 统计量。"""
    frames = data["frames"]
    mats = [f["transform_matrix"] for f in frames]
    n = len(frames)

    pos = [(m[0][3], m[1][3], m[2][3]) for m in mats]
    # 视线方向 = -第3列（相机前向是 -Z）；第2列是相机「上」轴
    look = []
    for m in mats:
        v = (-m[0][2], -m[1][2], -m[2][2])
        L = math.sqrt(sum(c * c for c in v)) or 1.0
        look.append(tuple(c / L for c in v))

    anchor = data.get("anchor_point")
    if anchor:
        anchor = tuple(float(v) for v in anchor)

    # 帧文件名是自开机的纳秒时间戳 → 用来算时长与 fps
    stamps = []
    for f in frames:
        stem = Path(f["file_path"]).stem
        if stem.isdigit():
            stamps.append(int(stem))
    duration = (stamps[-1] - stamps[0]) / 1e9 if len(stamps) > 1 else 0.0
    fps = (len(stamps) - 1) / duration if duration > 0 else 0.0

    f0 = frames[0]
    w, h = f0["w"], f0["h"]
    fl = (f0["fl_x"] + f0["fl_y"]) / 2
    diag = math.hypot(w, h)
    fov_d = math.degrees(2 * math.atan(diag / 2 / fl))
    fov_h = math.degrees(2 * math.atan(w / 2 / f0["fl_x"]))   # 图像水平方向（对应相机 +X）
    f_equiv = 43.2666 * fl / diag                              # 35mm 等效焦距

    # 水平距离取 (X, Z)——注意不是 (X, Y)：Y 是高度轴
    dists = [math.dist((p[0], p[2]), (anchor[0], anchor[2])) for p in pos] if anchor else []
    dist3 = [math.dist(p, anchor) for p in pos] if anchor else []
    # 视线与「相机→锚点」的夹角：0 表示正对锚点
    angles = []
    if anchor:
        for p, d in zip(pos, look):
            v = (anchor[0] - p[0], anchor[1] - p[1], anchor[2] - p[2])
            L = math.sqrt(sum(c * c for c in v)) or 1.0
            dot = sum(v[i] / L * d[i] for i in range(3))
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, dot)))))

    xs = [p[0] for p in pos]
    ys = [p[1] for p in pos]
    zs = [p[2] for p in pos]
    bbox = None
    if point_bbox:
        bbox = {"min": point_bbox[0], "max": point_bbox[1]}

    return {
        "id": folder, "data": data, "frames": frames, "n": n,
        "pos": pos, "look": look, "anchor": anchor, "points": points,
        "ply_total": ply_total, "bbox": bbox,
        "duration": duration, "fps": fps, "stamps": stamps,
        "res": (w, h), "intr": f0, "fov_h": fov_h, "fov_d": fov_d,
        "f_equiv": f_equiv, "fl": fl,
        "dist_mean": sum(dists) / len(dists) if dists else 0.0,
        "dist_min": min(dists) if dists else 0.0,
        "dist_max": max(dists) if dists else 0.0,
        "dist3_mean": sum(dist3) / len(dist3) if dist3 else 0.0,
        "ang_max": max(angles) if angles else 0.0,
        "ang_mean": sum(angles) / len(angles) if angles else 0.0,
        "range": {
            "x": (min(xs), max(xs)), "y": (min(ys), max(ys)), "z": (min(zs), max(zs)),
        },
    }


# ─────────────────────────── 通用绘制元件 ───────────────────────────

class View:
    """世界 (u, v) → SVG (x, y)，两轴强制同一 scale（避免轨迹被拉扁变形）。"""

    def __init__(self, umin, umax, vmin, vmax, x0, y0, x1, y1, pad=0.07):
        du, dv = (umax - umin) or 1.0, (vmax - vmin) or 1.0
        umin, umax = umin - du * pad, umax + du * pad
        vmin, vmax = vmin - dv * pad, vmax + dv * pad
        self.s = min((x1 - x0) / (umax - umin), (y1 - y0) / (vmax - vmin))
        self.ox = x0 + ((x1 - x0) - (umax - umin) * self.s) / 2 - umin * self.s
        self.oy = y0 + ((y1 - y0) - (vmax - vmin) * self.s) / 2 - vmin * self.s
        self.box = (x0, y0, x1, y1)
        self.umin, self.umax, self.vmin, self.vmax = umin, umax, vmin, vmax

    def p(self, u, v):
        return self.ox + u * self.s, self.oy + v * self.s

    def rect(self):
        x0, y0 = self.p(self.umin, self.vmin)
        x1, y1 = self.p(self.umax, self.vmax)
        return x0, y0, x1 - x0, y1 - y0


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def ramp_color(t, stops):
    """在 stops=[(pos,(r,g,b)), ...] 之间线性插值，t ∈ [0,1]。"""
    t = max(0.0, min(1.0, t))
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        if p0 <= t <= p1:
            k = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
            return "#%02X%02X%02X" % tuple(int(round(c0[j] + (c1[j] - c0[j]) * k)) for j in range(3))
    return "#%02X%02X%02X" % stops[-1][1]


def svg_open(w, h, bg=None) -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" '
           f'font-family="{FONT}" role="img">']
    if bg:
        out.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{bg}"/>')
    return "\n".join(out)


def text_w(s, fs):
    """估算文本像素宽度：CJK/全角按 1.0em，其余按 0.55em。

    图例是横排布局，中文按 0.55em 估宽会导致相邻项互相压字，必须区分宽窄字符。
    """
    return sum(fs * (1.0 if ord(ch) > 0x2E7F else 0.55) for ch in s)


def legend_row(items, x, y, fs=13, gap=26, color_var="#4B5563"):
    """横排图例：items = [(kind, label, color)]，kind ∈ line/arrow/dot/ring。"""
    out, cx = [], x
    for kind, label, color in items:
        if kind == "line":
            out.append(f'<line x1="{cx}" y1="{y}" x2="{cx + 22}" y2="{y}" stroke="{color}" stroke-width="2.4"/>')
            out.append(f'<circle cx="{cx + 11}" cy="{y}" r="3" fill="{color}"/>')
            out.append(f'<text x="{cx + 30}" y="{y + 5}" font-size="{fs}" fill="{color_var}">{esc(label)}</text>')
            cx += 30 + text_w(label, fs)
        elif kind == "arrow":
            out.append(f'<line x1="{cx}" y1="{y}" x2="{cx + 22}" y2="{y}" stroke="{color}" stroke-width="1.4"/>')
            out.append(f'<path d="M{cx + 16},{y - 4} L{cx + 22},{y} L{cx + 16},{y + 4}" fill="none" '
                       f'stroke="{color}" stroke-width="1.4" stroke-linecap="round"/>')
            out.append(f'<text x="{cx + 30}" y="{y + 5}" font-size="{fs}" fill="{color_var}">{esc(label)}</text>')
            cx += 30 + text_w(label, fs)
        elif kind == "dot":
            out.append(f'<circle cx="{cx + 11}" cy="{y}" r="4.5" fill="{color}"/>')
            out.append(f'<text x="{cx + 30}" y="{y + 5}" font-size="{fs}" fill="{color_var}">{esc(label)}</text>')
            cx += 30 + text_w(label, fs)
        elif kind == "ring":
            out.append(f'<circle cx="{cx + 11}" cy="{y}" r="6" fill="none" stroke="{color}" stroke-width="1.8"/>')
            out.append(f'<text x="{cx + 30}" y="{y + 5}" font-size="{fs}" fill="{color_var}">{esc(label)}</text>')
            cx += 30 + text_w(label, fs)
        elif kind == "wedge":
            out.append(f'<path d="M{cx + 11},{y + 7} L{cx + 11},{y - 7} L{cx + 25},{y} Z" fill="{color}" opacity="0.35"/>')
            out.append(f'<text x="{cx + 32}" y="{y + 5}" font-size="{fs}" fill="{color_var}">{esc(label)}</text>')
            cx += 32 + text_w(label, fs)
        elif kind == "dash":
            out.append(f'<line x1="{cx}" y1="{y}" x2="{cx + 22}" y2="{y}" stroke="{color}" stroke-width="1.2" stroke-dasharray="4 4"/>')
            out.append(f'<text x="{cx + 30}" y="{y + 5}" font-size="{fs}" fill="{color_var}">{esc(label)}</text>')
            cx += 30 + text_w(label, fs)
        cx += gap
    return "\n".join(out), cx - gap


def scale_bar(x, y, meters, view, color="#4B5563", fs=12, label=None):
    px = meters * view.s
    txt = label or (f"{meters:g} m")
    return "\n".join([
        f'<line x1="{x}" y1="{y}" x2="{x + px}" y2="{y}" stroke="{color}" stroke-width="1.8"/>',
        f'<line x1="{x}" y1="{y - 6}" x2="{x}" y2="{y + 6}" stroke="{color}" stroke-width="1.8"/>',
        f'<line x1="{x + px}" y1="{y - 6}" x2="{x + px}" y2="{y + 6}" stroke="{color}" stroke-width="1.8"/>',
        f'<text x="{x}" y="{y + 22}" font-size="{fs}" fill="{color}">{esc(txt)}</text>',
    ])


def axis_hints(view, color, fs=12, flip=False):
    """在绘图区外侧标出世界 X / Z 的正方向（俯视图约定：X 右、Z 下）。

    位置贴着数据框外侧而不是框内：数据框本身只有数据那么宽，框内左下角常常正好被
    轨迹压住；框外紧邻位置是白边，放轴指示既清楚又不遮数据。

    flip=True：改成画在数据框**内侧**。四联图右列的面板右边距只剩画布边距（40px），
    框外的 +X 标签会飘到画布外 —— 这一列必须用内侧版本。
    """
    rx0, ry0, rw, rh = view.rect()
    out = []
    # 左侧：Z 向下
    ax, ay = (rx0 + 14 if flip else rx0 - 30), ry0 + rh - 200
    out.append(f'<line x1="{ax}" y1="{ay}" x2="{ax}" y2="{ay + 34}" stroke="{color}" stroke-width="1.4"/>')
    out.append(f'<path d="M{ax - 4},{ay + 28} L{ax},{ay + 34} L{ax + 4},{ay + 28}" fill="none" stroke="{color}" '
               f'stroke-width="1.4" stroke-linecap="round"/>')
    out.append(f'<text x="{ax + 9}" y="{ay + 34}" font-size="{fs}" fill="{color}">+Z</text>')
    # 右侧：X 向右
    if flip:
        # 内侧版：箭头贴框右下角，标签挪到箭头上方（右边真的一格都不剩）
        bx, by = rx0 + rw - 48, ry0 + rh - 20
        out.append(f'<line x1="{bx}" y1="{by}" x2="{bx + 34}" y2="{by}" stroke="{color}" stroke-width="1.4"/>')
        out.append(f'<path d="M{bx + 28},{by - 4} L{bx + 34},{by} L{bx + 28},{by + 4}" fill="none" '
                   f'stroke="{color}" stroke-width="1.4" stroke-linecap="round"/>')
        out.append(f'<text x="{bx + 34}" y="{by - 8}" font-size="{fs}" fill="{color}" '
                   f'text-anchor="end">+X</text>')
    else:
        bx, by = rx0 + rw + 12, ry0 + rh - 26
        out.append(f'<line x1="{bx}" y1="{by}" x2="{bx + 34}" y2="{by}" stroke="{color}" stroke-width="1.4"/>')
        out.append(f'<path d="M{bx + 28},{by - 4} L{bx + 34},{by} L{bx + 28},{by + 4}" fill="none" '
                   f'stroke="{color}" stroke-width="1.4" stroke-linecap="round"/>')
        out.append(f'<text x="{bx + 40}" y="{by + 4}" font-size="{fs}" fill="{color}">+X</text>')
    return "\n".join(out)


def caption(ctx, x, y, color, fs=13):
    d = ctx["data"]
    plat = (d.get("platform_version") or "").split("/")
    dev = plat[5] if len(plat) > 5 else "?"
    txt = (f"{ctx['n']} 帧 · {ctx['duration']:.1f} s · {ctx['fps']:.2f} fps · "
           f"{ctx['res'][0]}×{ctx['res'][1]} · 等效 {ctx['f_equiv']:.1f} mm · "
           f"anchor 水平距离 {ctx['dist_min']:.2f}–{ctx['dist_max']:.2f} m · {esc(dev)}")
    return f'<text x="{x}" y="{y}" font-size="{fs}" fill="{color}">{txt}</text>'


# ─────────────────────── 公共几何绘制（minimal / iso / combo 共用） ───────────────────────

PAL_LIGHT = {
    "ink": "#111827", "muted": "#4B5563", "line": "#D1D5DB", "plot": "#FAFAF9",
    "track": "#185FA5", "dot": "#378ADD", "arrow": "#6B7280",
    "ring": "#B45309", "anc": "#B45309",
    "start": "#4D7C0F", "end": "#BE123C",
    "cloud": "#B4B2A9", "axis": "#9CA3AF", "grid": "#E5E7EB",
}

# 浅色底上的时间渐变（蓝 → 紫 → 玫红）：纯青/纯粉在白底上对比度不够，换成深一档
RAMP_LIGHT = [(0.0, (0x1D, 0x4E, 0xD8)), (0.5, (0x7C, 0x3A, 0xED)), (1.0, (0xDB, 0x27, 0x77))]


def cloud_in_view(points, extent, proj=None, max_pts=PLY_DRAW_POINTS):
    """只保留落在视野内的点云点并限流到 max_pts 个，返回投影后的 [(u, v)]。

    extent = (u_min, u_max, v_min, v_max)，与 View 的 umin/umax/vmin/vmax 对应。
    proj 默认按俯视图取 (X, Z)；等轴测传 iso_proj_pt。
    """
    proj = proj or (lambda p: (p[0], p[2]))
    u0, u1, v0, v1 = extent
    sel = []
    for p in points:
        u, v = proj(p)
        if u0 <= u <= u1 and v0 <= v <= v1:
            sel.append((u, v))
    if len(sel) > max_pts:
        step = len(sel) / max_pts
        sel = [sel[int(i * step)] for i in range(max_pts)]
    return sel


def cloud_screen_for(view, ctx, proj=None):
    """按 view 的视野取点云、投影成画布坐标 [(x, y)]，供绘制函数直接落点。"""
    pts = cloud_in_view(ctx["points"], (view.umin, view.umax, view.vmin, view.vmax), proj=proj)
    return [view.p(u, v) for u, v in pts]


def iso_proj(x, y, z):
    """等轴测投影：世界 (X, Y, Z) → 平面 (u, v)，v 向下。"""
    return (x - z) * ISO_COS, (x + z) * ISO_SIN - y


def iso_proj_pt(p):
    """同上，但按「整点 (x, y, z)」调用——给 cloud_in_view 这类接口用。"""
    return iso_proj(*p)


def iso_extent(ctx):
    """等轴测视野范围：只按「相机 + 锚点 + 相机在地面的投影」取，点云只作背景。

    室内扫描的点云常比相机活动范围大好几倍（3.7×2.1×4.9 m 的点云 vs 1.2×1.7 m 的轨迹），
    把点云也算进范围会把轨迹压成角落一小团；超出视野的点由 clipPath 裁掉。
    返回 (u_min, u_max, v_min, v_max, floor)。
    """
    pts = [iso_proj(*p) for p in ctx["pos"]]
    floor = ctx["bbox"]["min"][1] if ctx["bbox"] else min(p[1] for p in ctx["pos"]) - 1.0
    us = [p[0] for p in pts]
    vs = [p[1] for p in pts] + [iso_proj(c[0], floor, c[2])[1] for c in ctx["pos"]]
    if ctx["anchor"]:
        au, av = iso_proj(*ctx["anchor"])
        us.append(au)
        vs.append(av)
    return min(us), max(us), min(vs), max(vs), floor


def label_box(tx, ty, text, fs, anchor):
    """按 text-anchor 与基线位置估算文字包围盒 (x, y, w, h)。"""
    w = text_w(text, fs)
    x0 = tx if anchor == "start" else (tx - w if anchor == "end" else tx - w / 2.0)
    return (x0, ty - fs, w, fs + 5)


def boxes_hit(a, b, pad=3.0):
    return not (a[0] + a[2] + pad < b[0] or b[0] + b[2] + pad < a[0]
                or a[1] + a[3] + pad < b[1] or b[1] + b[3] + pad < a[1])


def draw_world_origin(ctx, view, proj, panel, ink, muted, blocked=(), fs=12):
    """画世界原点 (0,0,0)：在取景内直接画标记，在取景外就画到绘图区边缘并指向它。

    为什么不把原点并进取景范围：Remy 的 AR 世界原点是会话起点，实测离场景 0.5–5.1 m
    （10 个样本里只有 4 个落在场景取景内），并进去会把整个场景压成一小团。
    所以取景维持不变，框外用「边缘标记 + 箭头 + 距离」表示。

    blocked 是已占用的文字包围盒（锚点/起终点标签），框内时会挑一个不压字的方位放标签。
    """
    u, v = proj(0.0, 0.0, 0.0)
    x0, y0, w, h = panel
    px, py = view.p(u, v)

    def glyph(cx, cy):
        return (f'<rect x="{cx - 4:.1f}" y="{cy - 4:.1f}" width="8" height="8" fill="none" '
                f'stroke="{ink}" stroke-width="1.6"/>'
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="1.5" fill="{ink}"/>')

    label = "世界原点 (0,0,0)"
    if ctx["anchor"]:
        # 距离先取出来再拼：f-string 的表达式段里再出现同类引号，只有 Python 3.12+ 才合法
        d_origin = math.dist((0.0, 0.0, 0.0), ctx["anchor"])
        label += f" · 距锚点 {d_origin:.2f} m"

    if view.umin <= u <= view.umax and view.vmin <= v <= view.vmax:
        # 框内：在四个方位里挑第一个不压字、且完整落在绘图区内的
        cands = [(px + 12, py + 4, "start"), (px - 12, py + 4, "end"),
                 (px, py - 13, "middle"), (px, py + 20, "middle")]
        pick = None
        for tx, ty, an in cands:
            box = label_box(tx, ty, label, fs, an)
            in_panel = box[0] >= x0 + 4 and box[0] + box[2] <= x0 + w - 4
            if in_panel and not any(boxes_hit(box, b) for b in blocked):
                pick = (tx, ty, an)
                break
        if pick is None:
            pick = cands[0]
        tx, ty, an = pick
        return "\n".join([
            glyph(px, py),
            f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="{fs}" text-anchor="{an}" fill="{muted}">{esc(label)}</text>',
        ])

    # 框外：把方向向量夹到绘图区边框内侧，画外指箭头 + 标记 + 标签
    cx, cy = x0 + w / 2.0, y0 + h / 2.0
    dx, dy = px - cx, py - cy
    margin = 30.0
    t = min((w / 2 - margin) / abs(dx) if dx else 1e9, (h / 2 - margin) / abs(dy) if dy else 1e9)
    ex, ey = cx + dx * t, cy + dy * t
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    out = [
        f'<line x1="{ex - ux * 15:.1f}" y1="{ey - uy * 15:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
        f'stroke="{ink}" stroke-width="1.4"/>',
        f'<path d="M{ex - ux * 7 - uy * 4:.1f},{ey - uy * 7 + ux * 4:.1f} L{ex:.1f},{ey:.1f} '
        f'L{ex - ux * 7 + uy * 4:.1f},{ey - uy * 7 - ux * 4:.1f}" fill="none" stroke="{ink}" '
        f'stroke-width="1.4" stroke-linecap="round"/>',
        glyph(ex - ux * 24, ey - uy * 24),
    ]
    # 文字要落在标记之外：箭头占内 0~15px、标记占内 20~28px，所以文字从内 40px 起排
    tw = text_w(label, fs)
    if abs(ux) >= abs(uy):            # 贴左右边：文字朝框内展开
        lx = ex - ux * 40
        ly = ey + 4
        an = "start" if ux < 0 else "end"
    else:                              # 贴上下边：文字居中，横向夹在框内
        lx = min(max(ex, x0 + tw / 2 + 6), x0 + w - tw / 2 - 6)
        ly = ey - uy * 40 + 4
        an = "middle"
    out.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="{fs}" text-anchor="{an}" fill="{muted}">{esc(label)}</text>')
    return "\n".join(out)


def draw_top_geometry(ctx, view, clip, pal=PAL_LIGHT, ramp=None, *,
                      arrows_every=20, dots_every=10, arrow_len=0.26,
                      arrow_color=None, ring=True, cloud_screen=None, track_width=2.0,
                      clip_rect=None, origin=False):
    """把俯视图的几何画进给定 view。

    ramp=None → 轨迹用 pal['track'] 单色；否则传 [(pos,(r,g,b)),...] 按帧序做时间渐变。
    clip_rect 给定时按它裁剪（用于「绘图区预留框比数据框宽」的场合：点云/半径环裁到
    预留框，把两侧的空档填满）；不给则按数据框 view.rect() 裁。
    cloud_screen = 已投影到画布的 [(x, y)]（cloud_screen_for 的返回值），None 则不画点云。
    锚点与起终点标记不裁剪（避免贴边被切掉）。
    """
    out = []
    x0, y0, bw, bh = clip_rect if clip_rect is not None else view.rect()
    out.append(f'<clipPath id="{clip}"><rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" '
               f'height="{bh:.1f}" rx="10"/></clipPath>')
    out.append(f'<g clip-path="url(#{clip})">')

    if cloud_screen:
        out.append("".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1" fill="{pal["cloud"]}" opacity="0.5"/>'
                           for x, y in cloud_screen))

    if ring and ctx["anchor"]:
        ax, ay = view.p(ctx["anchor"][0], ctx["anchor"][2])
        out.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="{ctx["dist_mean"] * view.s:.1f}" fill="none" '
                   f'stroke="{pal["ring"]}" stroke-width="1.2" stroke-dasharray="5 5" opacity="0.55"/>')

    if ramp is None:
        pts = " ".join("%.1f,%.1f" % view.p(p[0], p[2]) for p in ctx["pos"])
        out.append(f'<polyline points="{pts}" fill="none" stroke="{pal["track"]}" stroke-width="{track_width}" '
                   f'stroke-linejoin="round" opacity="0.92"/>')
    else:
        for i in range(ctx["n"] - 1):
            x1, y1 = view.p(ctx["pos"][i][0], ctx["pos"][i][2])
            x2, y2 = view.p(ctx["pos"][i + 1][0], ctx["pos"][i + 1][2])
            out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                       f'stroke="{ramp_color(i / max(1, ctx["n"] - 1), ramp)}" stroke-width="{track_width + 0.4}" '
                       f'stroke-linecap="round"/>')

    for i in range(0, ctx["n"], dots_every):
        x, y = view.p(ctx["pos"][i][0], ctx["pos"][i][2])
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.8" fill="{pal["dot"]}"/>')

    acolor = arrow_color or pal["arrow"]
    for i in range(0, ctx["n"], arrows_every):
        x, y = view.p(ctx["pos"][i][0], ctx["pos"][i][2])
        d = ctx["look"][i]
        L = math.hypot(d[0], d[2]) or 1.0
        ex, ey = x + d[0] / L * arrow_len * view.s, y + d[2] / L * arrow_len * view.s
        ux, uy = (ex - x) / (math.hypot(ex - x, ey - y) or 1), (ey - y) / (math.hypot(ex - x, ey - y) or 1)
        out.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{acolor}" '
                   f'stroke-width="1.2" opacity="0.5"/>')
        out.append(f'<path d="M{ex - ux * 6 - uy * 4:.1f},{ey - uy * 6 + ux * 4:.1f} L{ex:.1f},{ey:.1f} '
                   f'L{ex - ux * 6 + uy * 4:.1f},{ey - uy * 6 - ux * 4:.1f}" fill="none" stroke="{acolor}" '
                   f'stroke-width="1.2" stroke-linecap="round" opacity="0.5"/>')
    out.append('</g>')

    if origin:   # 世界原点：框内画标记、框外画边缘指示；放在锚点之前，锚点压在上层
        blocked = []
        if ctx["anchor"]:
            ax, ay = view.p(ctx["anchor"][0], ctx["anchor"][2])
            blocked.append(label_box(ax + 16, ay - 8, "anchor_point", 12, "start"))
        if ctx["pos"]:
            sx, sy = view.p(ctx["pos"][0][0], ctx["pos"][0][2])
            ex, ey = view.p(ctx["pos"][-1][0], ctx["pos"][-1][2])
            blocked.append(label_box(sx + 12, sy + 5, "frame 0", 12, "start"))
            blocked.append(label_box(ex + 12, ey + 5, f"frame {ctx['n'] - 1}", 12, "start"))
        out.append(draw_world_origin(ctx, view, lambda x, y, z: (x, z), (x0, y0, bw, bh),
                                     pal["ink"], pal["muted"], blocked=blocked))

    if ctx["anchor"]:
        ax, ay = view.p(ctx["anchor"][0], ctx["anchor"][2])
        out.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="8" fill="none" stroke="{pal["anc"]}" stroke-width="2"/>')
        out.append(f'<line x1="{ax - 12:.1f}" y1="{ay:.1f}" x2="{ax + 12:.1f}" y2="{ay:.1f}" stroke="{pal["anc"]}" stroke-width="1"/>')
        out.append(f'<line x1="{ax:.1f}" y1="{ay - 12:.1f}" x2="{ax:.1f}" y2="{ay + 12:.1f}" stroke="{pal["anc"]}" stroke-width="1"/>')
        out.append(f'<text x="{ax + 16:.1f}" y="{ay - 8:.1f}" font-size="12" fill="{pal["anc"]}">anchor_point</text>')

    # 起终点标签：默认放在点右侧，若压到 anchor_point 标签就换到左/上/下。
    # 不避让时 frame 0 离锚点近的样本会印成「anchor_poinrame 0」（实测多例）。
    anc_box = None
    if ctx["anchor"]:
        ax, ay = view.p(ctx["anchor"][0], ctx["anchor"][2])
        anc_box = label_box(ax + 16, ay - 8, "anchor_point", 12, "start")

    def _edge_label(kind, i, color):
        px_, py_ = view.p(ctx["pos"][i][0], ctx["pos"][i][2])
        txt = "frame 0" if kind == "s" else f"frame {ctx['n'] - 1}"
        cands = [(px_ + 12, py_ + 5, "start"), (px_ - 12, py_ + 5, "end"),
                 (px_, py_ - 13, "middle"), (px_, py_ + 20, "middle")]
        pick = cands[0]
        if anc_box is not None:
            for tx, ty, an in cands:
                if not boxes_hit(label_box(tx, ty, txt, 12, an), anc_box):
                    pick = (tx, ty, an)
                    break
        return (f'<circle cx="{px_:.1f}" cy="{py_:.1f}" r="6" fill="{color}"/>'
                f'<text x="{pick[0]:.1f}" y="{pick[1]:.1f}" font-size="12" text-anchor="{pick[2]}" '
                f'fill="{color}">{txt}</text>')

    out.append(_edge_label("s", 0, pal["start"]))
    out.append(_edge_label("e", ctx["n"] - 1, pal["end"]))
    return "\n".join(out)


def draw_iso_geometry(ctx, view, clip, pal=PAL_LIGHT, *, cloud_screen=None, grid=True, drop_every=5,
                      dots_every=10, floor=None, clip_rect=None, origin=False, drop_lines=True):
    """把等轴测视图的几何（地面网格 / 点云 / 相机与垂线 / 轨迹 / 轴三叉）画进给定 view。

    cloud_screen = 已投影到画布的 [(x, y)]，None 则不画点云。
    """
    out = []
    x0, y0, bw, bh = clip_rect if clip_rect is not None else view.rect()
    if floor is None:
        floor = ctx["bbox"]["min"][1] if ctx["bbox"] else min(p[1] for p in ctx["pos"]) - 0.2

    out.append(f'<clipPath id="{clip}"><rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" '
               f'height="{bh:.1f}" rx="10"/></clipPath>')
    out.append(f'<g clip-path="url(#{clip})">')

    x_lo, x_hi = ctx["range"]["x"]
    z_lo, z_hi = ctx["range"]["z"]
    # 地面网格铺到「可见范围」而不只是相机范围：预留框比数据框宽时，两侧露出的地面也要有网格
    pad = 0.3 + max(0.0, (bw - view.rect()[2]) / 2) / view.s
    if grid:   # 地面网格：0.5 m 一条，投影后仍是直线
        k = math.floor(x_lo / 0.5) * 0.5
        while k <= x_hi + pad:
            x1, y1 = view.p(*iso_proj(k, floor, z_lo - pad))
            x2, y2 = view.p(*iso_proj(k, floor, z_hi + pad))
            out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{pal["grid"]}" stroke-width="0.8"/>')
            k += 0.5
        k = math.floor(z_lo / 0.5) * 0.5
        while k <= z_hi + pad:
            x1, y1 = view.p(*iso_proj(x_lo - pad, floor, k))
            x2, y2 = view.p(*iso_proj(x_hi + pad, floor, k))
            out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{pal["grid"]}" stroke-width="0.8"/>')
            k += 0.5

    if cloud_screen:
        out.append("".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.1" fill="{pal["cloud"]}" opacity="0.45"/>'
                           for x, y in cloud_screen))

    if drop_lines:   # 相机到地面的垂线，表达高度。约束图里球壳网格已够密，可整体关掉
        for i in range(0, ctx["n"], drop_every):
            c = ctx["pos"][i]
            x1, y1 = view.p(*iso_proj(c[0], c[1], c[2]))
            x2, y2 = view.p(*iso_proj(c[0], floor, c[2]))
            out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#CBD5E1" '
                       f'stroke-width="0.8" stroke-dasharray="3 3"/>')

    pts = " ".join("%.1f,%.1f" % view.p(*iso_proj(*p)) for p in ctx["pos"])
    out.append(f'<polyline points="{pts}" fill="none" stroke="{pal["track"]}" stroke-width="2" stroke-linejoin="round"/>')
    for i in range(0, ctx["n"], dots_every):
        x, y = view.p(*iso_proj(*ctx["pos"][i]))
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{pal["track"]}"/>')
    out.append('</g>')

    # 世界轴三叉：固定像素尺寸画在绘图区左下角（按裁剪框定位，框比数据宽时才不会跑偏）
    ox, oy = x0 + 78, y0 + bh - 110
    L = 48
    for dx, dy, lab, col in ((ISO_COS, ISO_SIN, "+X", "#DC2626"), (0, -1, "+Y", "#16A34A"),
                             (-ISO_COS, ISO_SIN, "+Z", "#2563EB")):
        ex, ey = ox + dx * L, oy + dy * L
        out.append(f'<line x1="{ox:.1f}" y1="{oy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{col}" stroke-width="1.6"/>')
        out.append(f'<text x="{ex + 6 * (1 if dx > 0 else -1):.1f}" y="{ey + (4 if dy >= 0 else -6):.1f}" '
                   f'font-size="12" text-anchor="{"start" if dx > 0 else ("middle" if dx == 0 else "end")}" '
                   f'fill="{col}">{lab}</text>')

    if origin:   # 世界原点：框内画标记、框外画边缘指示
        blocked = []
        if ctx["anchor"]:
            ax, ay = view.p(*iso_proj(*ctx["anchor"]))
            blocked.append(label_box(ax + 14, ay - 10, "anchor_point", 12, "start"))
        out.append(draw_world_origin(ctx, view, iso_proj, (x0, y0, bw, bh), pal["ink"], pal["muted"],
                                     blocked=blocked))

    if ctx["anchor"]:
        ax, ay = view.p(*iso_proj(*ctx["anchor"]))
        out.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="7" fill="none" stroke="{pal["anc"]}" stroke-width="2"/>')
        out.append(f'<text x="{ax + 14:.1f}" y="{ay - 10:.1f}" font-size="12" fill="{pal["anc"]}">anchor_point</text>')

    sx, sy = view.p(*iso_proj(*ctx["pos"][0]))
    ex, ey = view.p(*iso_proj(*ctx["pos"][-1]))
    out.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6" fill="{pal["start"]}"/>')
    out.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="6" fill="{pal["end"]}"/>')
    return "\n".join(out), floor


# ─────────────────────── 视角约束范围（view limit） ───────────────────────
#
# 「视角约束范围」的本质 = 以 target 为心的一圈球壳扇块：
#     方位角 az ∈ [az_lo, az_hi] · 仰角 el ∈ [el_lo, el_hi] · 距离 r ∈ [r_lo, r_hi]
# 换算成 UWA 的 view_limits 字段就是 Phi / Theta / Radius 三个区间，换算约定见
# 99c_repack_view_limits.py:175-186（权威，与官方 pack 脚本一致）：
#     loc = cam − target
#     pitch = asin(loc.y / |loc|)      Theta = 90 − pitch      （自 +Y 轴量起）
#     yaw   = atan2(−loc.x, loc.z)     Phi   = yaw             （x 取了负号）
#     R     = |loc|
#
# 为什么不照抄外面那版 C++ 移植代码（四点，都是方法论层面的错，不是精度差一点）：
#   ① 它把相机中心当未知数做最小二乘射线交汇。但 Remy 的 transforms.json 里本来就有
#      anchor_point，NOTES.md 实测各帧视线与 (anchor − C) 的夹角余弦均值 0.9926
#      → 直接取用更准、也不引入额外误差；估法只在缺 anchor 时当回退。
#   ② 它的 vulkan_to_webgl / webgl_to_vulkan 是同一个函数 [x, −y, −z]，且只在对
#      camera 与 target 都做一次的情况下参与角度计算 —— 刚体变换（det=+1，绕 X 转 180°），
#      角度结果完全不变，属死代码。
#   ③ 半径区间用固定比例 0.9×avg / 1.2×avg，是拍脑袋常数：实测某样本 view_limits.json
#      为 R[0.432, 2.159] 而 init 半径 1.909，按固定比例只会给出 [1.72, 2.29]，量级都不对。
#   ④ 极角区间带「跨度 > 10° 才把 init 值并进来」的条件（逻辑不自洽）；方位角解缠只在
#      |az| > 0.8π 时才补偿（步进稍大就崩）；绘图用 cos(phi) 从 +Y 量起，而极角是
#      arccos(−vy) 从 −Y 量起 → 画出来的球壳纵向是反的。
#
# 本实现：区间 = 各帧实测 min/max，再各加一个显式、可调的余量（下面三个常量）。
# 余量给小的理由：约束的目的是「允许用户在采集范围内游走」，小于采集范围会穿帮，
# 大太多会露没采到的区域，所以贴着实测区间给一点点最稳。
VL_AZ_PAD = float(os.environ.get("VL_AZ_PAD", "3.0"))    # 方位角两侧余量（度）
VL_EL_PAD = float(os.environ.get("VL_EL_PAD", "3.0"))    # 仰角两侧余量（度）
VL_R_PAD = float(os.environ.get("VL_R_PAD", "0.08"))     # 半径两侧余量（比例）
# 方位角跨度超过这个值就按「整圈」处理（原版是 1.8π≈324°，这里收紧到 300°）
VL_FULL_AZ = float(os.environ.get("VL_FULL_AZ", "300.0"))
# 可选：把视角约束叠进 combo 风格的俯视图（band/edges/angle/rings/hull），
# 或叠进等轴测面板（shell）。留空 = 完全不叠，combo 行为与旧版一致。
VLIMIT_MODE = os.environ.get("VLIMIT_MODE", "")

# 约束层配色：紫（与蓝轨迹 #185FA5、琥珀锚点 #B45309、绿/玫红起终点都能分开）；
# 包络（语义不同的对照组）单独用青绿。
LIM_INK = "#6D28D9"
LIM_FILL = "#8B5CF6"
LIM_ALT = "#0D9488"

# 绘图区预留框 (x, y, w, h)：六种画法共用同一块，便于并排比对。
# 注意是 (x, y, w, h) 而**不是** (x0, y0, x1, y1) —— 与 draw_top_geometry /
# draw_iso_geometry 的 clip_rect 参数同构（它们内部就是 x0, y0, bw, bh = clip_rect）。
# 早先按 (x0,y0,x1,y1) 传，等轴测的轴三叉与裁剪框都按 w=840/h=528 算，直接跑到画布外。
LIMIT_PANEL = (60.0, 156.0, 780.0, 372.0)

LIMIT_MODES = ("band", "edges", "angle", "rings", "hull", "shell")

LIMIT_LABELS = {
    "band": "扇环填充",
    "edges": "边界虚线",
    "angle": "角度标注",
    "rings": "极坐标网格",
    "hull": "采集包络",
    "shell": "等轴测球壳",
}

# 每种画法的「遮挡程度」主观评级 + 一句话说明，直接印在图上供挑图
LIMIT_NOTES = {
    "band": ("遮挡 ★★☆☆☆", "半透明扇环压在轨迹下层，一眼看出允许范围；色块会淡化底层点云"),
    "edges": ("遮挡 ★☆☆☆☆", "只留四条细虚线边界，零填充，对轨迹观感影响最小"),
    "angle": ("遮挡 ★★☆☆☆", "边界虚线 + 角度/半径数值标注，信息密度最高，适合当交付说明图"),
    "rings": ("遮挡 ★★☆☆☆", "以 target 为心加米制同心环与方位刻度，像极坐标纸，能读绝对值"),
    "hull": ("遮挡 ★★★☆☆", "不假设环绕：直接画相机水平位置的凸包外扩，语义是「实际覆盖区」"),
    "shell": ("遮挡 ★★★★☆", "等轴测里的 3D 球壳扇块，立体感最强，但网格线最多、最挡轨迹"),
}


def _solve3(A, b):
    """3×3 线性方程组，带部分选主元的高斯消元。纯标准库（99g 不依赖 numpy）。"""
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


def sight_center(ctx):
    """最小二乘求所有视线的最近公共点，仅在没有 anchor_point 时当回退。

    最小化 Σ‖(I − dᵢdᵢᵀ)(m − cᵢ)‖²（dᵢ = 第 i 帧视线单位向量、cᵢ = 光心），
    正规方程 Σ(I − dᵢdᵢᵀ)·m = Σ(I − dᵢdᵢᵀ)·cᵢ 是一个 3×3 系统，直接解即可
    （原版把 λᵢ 也塞进未知数凑 (3n)×(3+n) 的大矩阵，等价但没必要）。
    """
    A = [[0.0] * 3 for _ in range(3)]
    b = [0.0] * 3
    for c, d in zip(ctx["pos"], ctx["look"]):
        P = [[(1.0 if i == j else 0.0) - d[i] * d[j] for j in range(3)] for i in range(3)]
        for i in range(3):
            for j in range(3):
                A[i][j] += P[i][j]
            b[i] += sum(P[i][j] * c[j] for j in range(3))
    m = _solve3(A, b)
    if m is None:   # 视线几乎共线（矩阵奇异）→ 退回相机质心，总比崩掉好
        n = len(ctx["pos"]) or 1
        return tuple(sum(p[i] for p in ctx["pos"]) / n for i in range(3))
    return (m[0], m[1], m[2])


def compute_view_limit(ctx):
    """算出视角约束区间（球壳扇块）。纯数据驱动：实测 min/max ± VL_*_PAD。

    返回 dict，其中：
      az/el/rho/r_*  = 本脚本绘图与判断用的量（az 自 +Z 轴起、朝 +X 为正；el 自水平面起）
      rho            = 相机位置到 target 的**水平投影**半径（俯视图里能画的就只有它）
      r              = 相机到 target 的 3D 距离（= UWA 的 Radius）
      uwa            = 换算成 Phi/Theta/Radius 的对照值（未加余量，方便与 view_limits.json 比）
    """
    target = ctx["anchor"]
    src = "anchor_point"
    if target is None:
        target = sight_center(ctx)
        src = "sight_center（无 anchor_point，最小二乘回退）"

    az, el, rho, rr = [], [], [], []
    for p in ctx["pos"]:
        dx, dy, dz = p[0] - target[0], p[1] - target[1], p[2] - target[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz) or 1e-9
        rr.append(d)
        rho.append(math.hypot(dx, dz))
        el.append(math.degrees(math.asin(max(-1.0, min(1.0, dy / d)))))
        az.append(math.degrees(math.atan2(dx, dz)))

    # 帧序解缠：每步增量归一到 (−180, 180] 后累加。比原版「只在 |az| > 0.8π 时才补偿」稳，
    # 慢速小步长、跨 ±180° 分支切、甚至绕两圈都能正确展开。
    azu = [az[0]]
    for a in az[1:]:
        azu.append(azu[-1] + ((a - azu[-1] + 180.0) % 360.0 - 180.0))

    az_span = max(azu) - min(azu)
    full_az = az_span >= VL_FULL_AZ
    if full_az:   # 绕了一整圈 → 方位角不设限，画成闭合环
        az_lo, az_hi = azu[0], azu[0] + 360.0
    else:
        az_lo, az_hi = min(azu) - VL_AZ_PAD, max(azu) + VL_AZ_PAD

    el_lo, el_hi = min(el) - VL_EL_PAD, max(el) + VL_EL_PAD
    rho_lo = max(0.0, min(rho) * (1.0 - VL_R_PAD))
    rho_hi = max(rho) * (1.0 + VL_R_PAD)
    r_lo = max(0.0, min(rr) * (1.0 - VL_R_PAD))
    r_hi = max(rr) * (1.0 + VL_R_PAD)

    return {
        "target": target, "target_src": src,
        "az_lo": az_lo, "az_hi": az_hi, "full_az": full_az,
        "el_lo": el_lo, "el_hi": el_hi,
        "rho_lo": rho_lo, "rho_hi": rho_hi, "r_lo": r_lo, "r_hi": r_hi,
        "raw": {"az": (min(azu), max(azu)), "el": (min(el), max(el)),
                "rho": (min(rho), max(rho)), "r": (min(rr), max(rr))},
        # UWA 对照（未加余量）：Phi = −az、Theta = 90 − el、Radius = r
        "uwa": {"phi": (-max(azu), -min(azu)),
                "theta": (90.0 - max(el), 90.0 - min(el)),
                "radius": (min(rr), max(rr))},
        "pos_az": azu, "pos_el": el, "pos_rho": rho,
    }


def _arc_pts(cx, cy, rad_px, a0, a1, step=2.0):
    """屏幕坐标下的圆弧折线点串。方位角 a 与屏幕的对应：x = cx + R·sin a、y = cy + R·cos a。

    因为俯视图里世界 X → 屏幕 +x、世界 Z → 屏幕 +y（View.p），而
    az = atan2(dx, dz) → 世界方向 (sin az, cos az)，所以映射是直截了当的。
    """
    n = max(2, int(abs(a1 - a0) / step) + 1)
    r0, r1 = math.radians(a0), math.radians(a1)
    return " ".join("%.1f,%.1f" % (cx + rad_px * math.sin(r0 + (r1 - r0) * k / (n - 1)),
                                   cy + rad_px * math.cos(r0 + (r1 - r0) * k / (n - 1)))
                    for k in range(n))


def _sector_poly(cx, cy, r_in_px, r_out_px, a0, a1, step=2.0):
    """扇环（annular sector）多边形顶点串：外弧正序 + 内弧逆序。"""
    n = max(2, int(abs(a1 - a0) / step) + 1)
    r0, r1 = math.radians(a0), math.radians(a1)
    outer, inner = [], []
    for k in range(n):
        a = r0 + (r1 - r0) * k / (n - 1)
        s, c = math.sin(a), math.cos(a)
        outer.append((cx + r_out_px * s, cy + r_out_px * c))
        inner.append((cx + r_in_px * s, cy + r_in_px * c))
    return " ".join("%.1f,%.1f" % p for p in outer + inner[::-1])


def _hull2d(pts):
    """Andrew monotone chain 凸包（屏幕坐标即可，绕向无所谓）。"""
    ps = sorted(set(pts))
    if len(ps) < 3:
        return ps

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and cross(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h

    return half(ps)[:-1] + half(ps[::-1])[:-1]


# ── 六种画法（都只返回绘图区内的 SVG 片段；底图与裁剪由 _limit_svg 负责）──

def _lim_band(ctx, vl, view):
    tx, ty = view.p(vl["target"][0], vl["target"][2])
    poly = _sector_poly(tx, ty, vl["rho_lo"] * view.s, vl["rho_hi"] * view.s, vl["az_lo"], vl["az_hi"])
    return "\n".join([
        f'<polygon points="{poly}" fill="{LIM_FILL}" opacity="0.14"/>',
        f'<polygon points="{poly}" fill="none" stroke="{LIM_INK}" stroke-width="1.3" '
        f'stroke-dasharray="7 5" opacity="0.62"/>',
    ])


def _lim_edges(ctx, vl, view, ink=LIM_INK):
    tx, ty = view.p(vl["target"][0], vl["target"][2])
    r0, r1 = vl["rho_lo"] * view.s, vl["rho_hi"] * view.s
    out = [
        f'<polyline points="{_arc_pts(tx, ty, r1, vl["az_lo"], vl["az_hi"])}" fill="none" '
        f'stroke="{ink}" stroke-width="1.3" stroke-dasharray="7 5" opacity="0.62"/>',
        f'<polyline points="{_arc_pts(tx, ty, r0, vl["az_lo"], vl["az_hi"])}" fill="none" '
        f'stroke="{ink}" stroke-width="1.2" stroke-dasharray="5 5" opacity="0.45"/>',
    ]
    for a in (vl["az_lo"], vl["az_hi"]):
        ar = math.radians(a)
        out.append(f'<line x1="{tx:.1f}" y1="{ty:.1f}" x2="{tx + r1 * math.sin(ar):.1f}" '
                   f'y2="{ty + r1 * math.cos(ar):.1f}" stroke="{ink}" stroke-width="1.2" '
                   f'stroke-dasharray="6 5" opacity="0.5"/>')
    return "\n".join(out)


def _lim_angle(ctx, vl, view, ink=LIM_INK, rho_text=True):
    """边界虚线 + 在边与外弧旁标注区间数值（信息最全的一种）。

    数值一律加底色描边（_limit_txt）：这些字注定要压在轨迹或锚点旁边，
    不描边就变成一团糊字。

    rho_text=False 时省掉 ρ 的数值文字（标尺线保留）。四联图右列用它：
    那个位置的底图标签（anchor_point / frame 0 / 世界原点）已经很密，
    再叠一行 ρ 数值就糊了，而面板下方的数值行本来就写着 ρ 区间。
    """
    out = [_lim_edges(ctx, vl, view, ink)]
    tx, ty = view.p(vl["target"][0], vl["target"][2])
    r0, r1 = vl["rho_lo"] * view.s, vl["rho_hi"] * view.s

    if not vl["full_az"]:
        for a in (vl["az_lo"], vl["az_hi"]):   # 两个方位边界值贴在弧端点外侧
            ar = math.radians(a)
            ex, ey = tx + (r1 + 16) * math.sin(ar), ty + (r1 + 16) * math.cos(ar)
            an = "middle" if abs(math.cos(ar)) > 0.5 else ("start" if math.sin(ar) > 0 else "end")
            out.append(_limit_txt(ex, ey + 4, f"{a:.1f}°", 12.5, ink, an))
    # 整圈时 az_lo/az_hi 是解缠后的原始值（可以是 447.3° 这种），标出来只会误导：
    # 起终点重合、区间无意义，改成一个「整圈 360°」说明放在外弧右侧。

    # 半径区间：沿 az 中线画一条带端点的径向标尺，数值甩到外弧之外
    am = math.radians((vl["az_lo"] + vl["az_hi"]) / 2.0)
    sx, sy = tx + r0 * math.sin(am), ty + r0 * math.cos(am)
    ex, ey = tx + r1 * math.sin(am), ty + r1 * math.cos(am)
    out.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{ink}" '
               f'stroke-width="1.6" opacity="0.7"/>')
    for px_, py_ in ((sx, sy), (ex, ey)):
        out.append(f'<circle cx="{px_:.1f}" cy="{py_:.1f}" r="3" fill="{ink}" opacity="0.8"/>')
    if rho_text:
        lx, ly = tx + (r1 + 26) * math.sin(am), ty + (r1 + 26) * math.cos(am)
        out.append(_limit_txt(lx, ly + 4, f"ρ {vl['rho_lo']:.2f} – {vl['rho_hi']:.2f} m", 12.5, ink))

    # 张角：标在外弧中点之外
    mx, my = tx + (r1 + 16) * math.sin(am), ty + (r1 + 16) * math.cos(am) - 18
    out.append(_limit_txt(mx, my, f"张角 {vl['az_hi'] - vl['az_lo']:.1f}°", 12.5, ink, "middle"))
    return "\n".join(out)


def _lim_rings(ctx, vl, view):
    """极坐标纸：米制同心环 + 30° 方位刻度，再把约束区间用粗弧高亮出来。"""
    tx, ty = view.p(vl["target"][0], vl["target"][2])
    out = []
    step = 0.25 if vl["rho_hi"] <= 1.5 else (0.5 if vl["rho_hi"] <= 4.0 else 1.0)
    k = step
    while k <= vl["rho_hi"] + step:
        rr = k * view.s
        out.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="{rr:.1f}" fill="none" stroke="{LIM_INK}" '
                   f'stroke-width="0.7" opacity="0.16"/>')
        if rr > 34:   # 太小就不写字，免得糊成一团
            out.append(_limit_txt(tx + 3, ty - rr + 11, f"{k:g} m", 10, LIM_INK))
        k += step
    for a in range(0, 360, 30):
        ar = math.radians(a)
        ex, ey = tx + (vl["rho_hi"] + step * 0.25) * view.s * math.sin(ar), \
                 ty + (vl["rho_hi"] + step * 0.25) * view.s * math.cos(ar)
        out.append(f'<line x1="{tx:.1f}" y1="{ty:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{LIM_INK}" '
                   f'stroke-width="0.6" opacity="0.12"/>')
    # 高亮：区间外弧 + 两条边界
    mid = (vl["rho_lo"] + vl["rho_hi"]) / 2.0
    out.append(f'<polyline points="{_arc_pts(tx, ty, mid * view.s, vl["az_lo"], vl["az_hi"], 1.5)}" '
               f'fill="none" stroke="{LIM_INK}" stroke-width="2.6" opacity="0.55"/>')
    for a in (vl["az_lo"], vl["az_hi"]):
        ar = math.radians(a)
        out.append(f'<line x1="{tx + vl["rho_lo"] * view.s * math.sin(ar):.1f}" '
                   f'y1="{ty + vl["rho_lo"] * view.s * math.cos(ar):.1f}" '
                   f'x2="{tx + vl["rho_hi"] * view.s * math.sin(ar):.1f}" '
                   f'y2="{ty + vl["rho_hi"] * view.s * math.cos(ar):.1f}" stroke="{LIM_INK}" '
                   f'stroke-width="2.2" opacity="0.55"/>')
    return "\n".join(out)


def _lim_hull(ctx, vl, view):
    """对照组：不假设环绕，直接取相机水平位置的凸包并沿质心外扩 VL_R_PAD。"""
    pts = [view.p(p[0], p[2]) for p in ctx["pos"]]
    hull = _hull2d(pts)
    if len(hull) < 3:
        return ""
    cx = sum(p[0] for p in hull) / len(hull)
    cy = sum(p[1] for p in hull) / len(hull)
    grown = [(cx + (p[0] - cx) * (1.0 + VL_R_PAD), cy + (p[1] - cy) * (1.0 + VL_R_PAD))
             for p in hull]
    poly = " ".join("%.1f,%.1f" % p for p in grown)
    return "\n".join([
        f'<polygon points="{poly}" fill="{LIM_ALT}" opacity="0.12"/>',
        f'<polygon points="{poly}" fill="none" stroke="{LIM_ALT}" stroke-width="1.4" '
        f'stroke-dasharray="8 5" opacity="0.65"/>',
    ])


def _lim_shell(ctx, vl, view, ink=LIM_INK, fill=LIM_FILL):
    """等轴测里的 3D 球壳扇块：内外两层球面片网格 + 4 条径向棱 + 淡填充。

    填充不能用「四条边界围一个闭合多边形」——az 跨度大（这里能到 270°）时，等轴测
    投影会让这个多边形自相交，填出一块形状完全不对的色斑。改成按方位角切成
    互不重叠的四边形小片逐片填（相邻片共享边，所以叠加后浓淡仍然均匀）。
    """
    t = vl["target"]
    az_n, el_n = 24, 6        # 网格密度：方位 24 条 / 仰角 6 条，够看出是「壳」了
    azs = [vl["az_lo"] + (vl["az_hi"] - vl["az_lo"]) * k / (az_n - 1) for k in range(az_n)]
    els = [vl["el_lo"] + (vl["el_hi"] - vl["el_lo"]) * k / (el_n - 1) for k in range(el_n)]

    def world(r, a_deg, e_deg):
        ar, er = math.radians(a_deg), math.radians(e_deg)
        return iso_proj(t[0] + r * math.cos(er) * math.sin(ar),
                        t[1] + r * math.sin(er),
                        t[2] + r * math.cos(er) * math.cos(ar))

    out = []
    # 填充：按方位角切成互不重叠的四边形小片逐片填。
    # 不用「四条边界围一个闭合多边形」——az 跨度大（这里能到 270°）时等轴测投影会让
    # 那个多边形自相交，填出一块形状完全不对的色斑。但也只在仰角跨度够宽时才填：
    # el 很窄时球壳本身就是一条薄带，再叠填充会在投影重叠处积成突兀的深色块。
    if vl["el_hi"] - vl["el_lo"] >= 8.0:
        for k in range(az_n - 1):
            quad = [world(vl["r_hi"], azs[k], vl["el_lo"]), world(vl["r_hi"], azs[k + 1], vl["el_lo"]),
                    world(vl["r_hi"], azs[k + 1], vl["el_hi"]), world(vl["r_hi"], azs[k], vl["el_hi"])]
            # 同色细描边：相邻片共享边，抗锯齿各画一半会留一道更浅的缝，
            # 叠 24 片就成了一把「扇骨」。描边把缝填掉，整块看起来才是连续曲面。
            out.append('<polygon points="%s" fill="%s" fill-opacity="0.07" stroke="%s" '
                       'stroke-width="0.8" stroke-opacity="0.07"/>'
                       % (" ".join("%.1f,%.1f" % view.p(*q) for q in quad), fill, fill))

    for r in (vl["r_hi"], vl["r_lo"]):   # 内外两层网格
        op = 0.34 if r == vl["r_hi"] else 0.2
        for e in els:
            pts = " ".join("%.1f,%.1f" % view.p(*world(r, a, e)) for a in azs)
            out.append(f'<polyline points="{pts}" fill="none" stroke="{ink}" '
                       f'stroke-width="0.7" opacity="{op}"/>')
        for a in azs:
            pts = " ".join("%.1f,%.1f" % view.p(*world(r, a, e)) for e in els)
            out.append(f'<polyline points="{pts}" fill="none" stroke="{ink}" '
                       f'stroke-width="0.7" opacity="{op}"/>')

    for a in (vl["az_lo"], vl["az_hi"]):        # 4 条径向棱
        for e in (vl["el_lo"], vl["el_hi"]):
            p1, p2 = view.p(*world(vl["r_lo"], a, e)), view.p(*world(vl["r_hi"], a, e))
            out.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                       f'stroke="{ink}" stroke-width="1.3" opacity="0.55"/>')
    return "\n".join(out)


LIMIT_DRAWS = {
    "band": _lim_band, "edges": _lim_edges, "angle": _lim_angle,
    "rings": _lim_rings, "hull": _lim_hull, "shell": _lim_shell,
}


def _limit_iso_extent(ctx, vl):
    """等轴测视野：相机活动范围 ∪ 球壳扇块的 8 个角点（否则球壳会被裁掉半截）。"""
    u0, u1, v0, v1, floor = iso_extent(ctx)
    t = vl["target"]
    corners = []
    for r in (vl["r_lo"], vl["r_hi"]):
        for a in (vl["az_lo"], vl["az_hi"]):
            for e in (vl["el_lo"], vl["el_hi"]):
                ar, er = math.radians(a), math.radians(e)
                corners.append(iso_proj(t[0] + r * math.cos(er) * math.sin(ar),
                                        t[1] + r * math.sin(er),
                                        t[2] + r * math.cos(er) * math.cos(ar)))
    us = [c[0] for c in corners]
    vs = [c[1] for c in corners]
    return min(u0, min(us)), max(u1, max(us)), min(v0, min(vs)), max(v1, max(vs)), floor


def _fit_view(u0, u1, v0, v1, panel_xywh, pad=0.07):
    """按面板长宽比扩张较短的一维，让内容填满预留框。

    不这么做的话 View 是「等比例尺 + 居中」，数据总是窄于面板的长宽比，
    于是 view.rect() 只覆盖中间一小条 —— 沿外弧甩出去的数值标注会被裁到框外
    （这正是第一版 angle 图里「张角 269.5°」被切掉的原因）。

    ⚠️ panel_xywh 是 **(x, y, w, h)**，与 LIMIT_PANEL / clip_rect 同构，
    **不是 (x0, y0, x1, y1)**。传错的话 a/b 长宽比会在两个调用点得到不同值
    （w/h 被当成 x1/y1），同一份数据在左右两列被扩张成不同的比例尺 ——
    四联图里表现为右列内容莫名放大两倍且偏位。

    注意 pad 只加在 View 内部，所以这里只要让**原始**跨度比等于面板比即可：
    View 两轴同比例加 pad，比值得以保持。
    """
    x0, y0, w, h = panel_xywh
    target = w / h
    du, dv = (u1 - u0) or 1.0, (v1 - v0) or 1.0
    if du / dv < target:
        need = dv * target
        c = (u0 + u1) / 2.0
        u0, u1 = c - need / 2.0, c + need / 2.0
    else:
        need = du / target
        c = (v0 + v1) / 2.0
        v0, v1 = c - need / 2.0, c + need / 2.0
    return View(u0, u1, v0, v1, x0, y0, x0 + w, y0 + h, pad=pad)


def _limit_top_view(ctx, vl):
    """俯视图取景 = 相机范围 ∪ 约束扇环外缘（环必须完整入框，否则看不出区间）。"""
    x_lo, x_hi = ctx["range"]["x"]
    z_lo, z_hi = ctx["range"]["z"]
    t = vl["target"]
    r = vl["rho_hi"]
    return _fit_view(min(x_lo, t[0] - r), max(x_hi, t[0] + r),
                     min(z_lo, t[2] - r), max(z_hi, t[2] + r), LIMIT_PANEL)


def _limit_head(ctx, vl, mode):
    """标题 + 采集统计 + 三行约束数值（含 UWA 对照）。"""
    OUT = "#111827"
    MUTED = "#4B5563"
    out = [
        f'<text x="40" y="42" font-size="21" font-weight="500" fill="{OUT}">'
        f'{esc(ctx["id"])} · 视角约束范围 · {esc(LIMIT_LABELS[mode])}</text>',
        caption(ctx, 40, 66, MUTED),
    ]
    t = vl["target"]
    tgt = f'({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})'
    out.append(f'<text x="40" y="92" font-size="12.5" fill="{MUTED}">'
               f'target = {esc(vl["target_src"])} {esc(tgt)} ｜ 区间 = 各帧实测 min/max ± '
               f'余量（方位 {VL_AZ_PAD:g}° · 仰角 {VL_EL_PAD:g}° · 半径 {VL_R_PAD * 100:g}%）'
               f'{" ｜ ⚠️ 方位角近似整圈，按 360° 处理" if vl["full_az"] else ""}</text>')
    if vl["full_az"]:
        az_txt = "整圈 360°（不设限）"
    else:
        az_txt = (f'{vl["az_lo"]:.1f}° → {vl["az_hi"]:.1f}°'
                  f'（跨度 {vl["az_hi"] - vl["az_lo"]:.1f}°）')
    out.append(f'<text x="40" y="112" font-size="12.5" fill="{LIM_INK}">'
               f'方位角 az {az_txt} ｜ 仰角 el {vl["el_lo"]:.1f}° → '
               f'{vl["el_hi"]:.1f}° ｜ 水平半径 ρ {vl["rho_lo"]:.2f} → {vl["rho_hi"]:.2f} m ｜ '
               f'3D 距离 R {vl["r_lo"]:.2f} → {vl["r_hi"]:.2f} m</text>')
    u = vl["uwa"]
    out.append(f'<text x="40" y="132" font-size="12" fill="#9CA3AF">'
               f'UWA view_limits 对照（实测值，未加余量）：Phi {u["phi"][0]:.1f}° → '
               f'{u["phi"][1]:.1f}° ｜ Theta {u["theta"][0]:.1f}° → {u["theta"][1]:.1f}° ｜ '
               f'Radius {u["radius"][0]:.3f} → {u["radius"][1]:.3f} m</text>')
    return "\n".join(out)


def _limit_txt(x, y, s, fs, color, anchor="start"):
    """约束层的数值文字：一律加底色描边，压在轨迹/网格上也能读清。"""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{fs}" fill="{color}" '
            f'text-anchor="{anchor}" paint-order="stroke" stroke="#FAFAF9" '
            f'stroke-width="3.5" stroke-linejoin="round">{esc(s)}</text>')


def _limit_foot(ctx, vl, mode, view, y_note):
    """说明行 + 图例 + 比例尺 + 右下角脚注。"""
    MUTED = "#4B5563"
    rate, note = LIMIT_NOTES[mode]
    out = [f'<text x="40" y="{y_note}" font-size="13" fill="{LIM_INK}">'
           f'{esc(rate)} · {esc(note)}</text>']
    items = [("ring", "target（球壳中心）", "#B45309"),
             ("dash", "视角约束边界", LIM_INK),
             ("line", f"相机轨迹（{ctx['n']} 帧）", PAL_LIGHT["track"])]
    if mode == "band":
        items.insert(1, ("wedge", "允许范围（弧带）", LIM_FILL))
    elif mode == "hull":
        items = [("ring", "target（球壳中心）", "#B45309"),
                 ("dash", f"采集包络（凸包外扩 {VL_R_PAD * 100:g}%）", LIM_ALT),
                 ("line", f"相机轨迹（{ctx['n']} 帧）", PAL_LIGHT["track"])]
    elif mode == "shell":
        items = [("ring", "target（球壳中心）", "#B45309"),
                 ("dash", "3D 球壳扇块（内外两层球面片）", LIM_INK),
                 ("line", f"相机轨迹（{ctx['n']} 帧）", PAL_LIGHT["track"])]
    row, cx = legend_row(items, 40, y_note + 34, fs=12.5, gap=24, color_var=MUTED)
    out.append(row)
    out.append(scale_bar(cx + 16, y_note + 34, 0.5, view, color=MUTED))
    return "\n".join(out)


def _limit_svg(ctx, vl, mode):
    """一张 900×632 的视角约束图：底图（轨迹/锚点/起终点）+ 约束层 + 说明。"""
    W, H = 900, 632
    LINE = "#D1D5DB"
    iso = mode == "shell"
    # clipPath 的 id 必须带 mode：vlimit 总览把 6 张图内联进同一个文档，
    # 同一文档里重复 id 的话 url(#id) 只会解析到第一个，第 2~6 张会被第一张的
    # 裁剪框裁掉（实测过：单独看每张都正常，拼进总览就只剩第一张有内容）。
    if iso:
        u0, u1, v0, v1, floor = _limit_iso_extent(ctx, vl)
        view = _fit_view(u0, u1, v0, v1, LIMIT_PANEL)
        # 等轴测底图把地面网格调淡、并砍掉相机垂线：球壳网格本身已经够密，
        # 三套线叠在一起会糊成一片，看不出球壳形状。
        base, _ = draw_iso_geometry(ctx, view, f"clip_{ctx['id'][:8]}_vls_{mode}",
                                    pal=dict(PAL_LIGHT, grid="#F1F5F9"), floor=floor,
                                    clip_rect=LIMIT_PANEL, drop_lines=False)
    else:
        view = _limit_top_view(ctx, vl)
        base = draw_top_geometry(ctx, view, f"clip_{ctx['id'][:8]}_vlt_{mode}", ring=False,
                                 origin=False, clip_rect=LIMIT_PANEL)

    # 约束层画在底图**下面**：轨迹永远压在最上层，这是「不影响原有轨迹」的关键。
    # 面板 rect / 裁剪框都用预留框（不是 view.rect()）：View 是等比例尺 + 居中，
    # 数据框可能只占预留框中间一条，用它当面板会把甩到外弧之外的角度标注裁掉。
    x0, y0, bw, bh = LIMIT_PANEL
    cid = f"clip_{ctx['id'][:8]}_vllim_{mode}"
    layer = [
        f'<clipPath id="{cid}"><rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="10"/></clipPath>',
        f'<g clip-path="url(#{cid})">',
        LIMIT_DRAWS[mode](ctx, vl, view),
        '</g>',
    ]

    out = [svg_open(W, H, "#FFFFFF")]
    out.append(_limit_head(ctx, vl, mode))
    out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="10" '
               f'fill="#FAFAF9" stroke="{LINE}"/>')
    out.append("\n".join(layer))
    out.append(base)
    # target 标记：底图（draw_top_geometry / draw_iso_geometry）只画 anchor_point，
    # 所以仅当 target 走了 fallback（无 anchor_point）时才补画，免得叠出双圈
    if vl["target_src"].startswith("sight_center"):
        tx, ty = (view.p(*iso_proj(*vl["target"])) if iso else view.p(vl["target"][0], vl["target"][2]))
        out.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="7" fill="none" stroke="#B45309" stroke-width="2"/>')
        out.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="2" fill="#B45309"/>')
        out.append(f'<text x="{tx + 12:.1f}" y="{ty - 10:.1f}" font-size="12" fill="#B45309">target</text>')
    out.append(_limit_foot(ctx, vl, mode, view, 556))
    out.append(f'<text x="860" y="626" font-size="12" fill="#9CA3AF" text-anchor="end">'
               f'{"等轴测投影：(X−Z)·cos30°, (X+Z)·sin30° − Y" if iso else "俯视图（+X 向右、+Z 向下）"}'
               f' ｜ 等比例尺 {view.s:.1f} px/m ｜ '
               f'UWA 换算见 99c：Phi = atan2(−x, z)、Theta = 90 − pitch</text>')
    out.append("</svg>")
    return "\n".join(out)


def _limit_style(mode):
    """把某个画法包成 99g 的 style 函数（供 RENDERERS 用）。"""
    def _render(ctx):
        vl = compute_view_limit(ctx)
        svg = _limit_svg(ctx, vl, mode)
        return svg, 632
    return _render


def style_vlimit(ctx):
    """六种画法总览：一张 HTML 里内联六张 900×632 的 SVG，竖向排开便于挑图。"""
    vl = compute_view_limit(ctx)
    return "\n".join(_limit_svg(ctx, vl, m) for m in LIMIT_MODES), 632 * len(LIMIT_MODES)


def ramp_bar(x, y, w, h, ramp, label, ink, fs=12, steps=48):
    """时间渐变色条 + 两端帧号，用于说明轨迹颜色随时间变化。"""
    out = [f'<text x="{x}" y="{y - 8}" font-size="{fs}" fill="{ink}">{esc(label)}</text>']
    for k in range(steps):
        out.append(f'<rect x="{x + w * k / steps:.1f}" y="{y}" width="{w / steps + 1:.2f}" height="{h}" '
                   f'fill="{ramp_color(k / (steps - 1), ramp)}"/>')
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{ink}" stroke-width="0.5" opacity="0.35"/>')
    return "\n".join(out)


# ─────────────────────────── 四种风格 ───────────────────────────

def style_minimal(ctx):
    """浅色极简：轨迹 + 视线箭头 + 锚点均值半径环。"""
    W, H = 900, 620
    C_TRACK, C_DOT, C_ARR = "#185FA5", "#378ADD", "#6B7280"
    C_START, C_END, C_ANC = "#4D7C0F", "#BE123C", "#B45309"
    INK, MUTED, LINE = "#111827", "#4B5563", "#D1D5DB"

    view = View(*ctx["range"]["x"], *ctx["range"]["z"], 60, 110, 840, 500)
    out = [svg_open(W, H, "#FFFFFF")]
    out.append(f'<text x="40" y="42" font-size="21" font-weight="500" fill="{INK}">'
               f'{esc(ctx["id"])} · 采集轨迹俯视图（世界系 XZ）</text>')
    out.append(caption(ctx, 40, 66, MUTED))

    x0, y0, bw, bh = view.rect()
    out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="10" '
               f'fill="#FAFAF9" stroke="{LINE}" stroke-width="1"/>')
    out.append(draw_top_geometry(ctx, view, f"clip_{ctx['id'][:8]}_min", origin=True))

    out.append(axis_hints(view, "#9CA3AF"))

    row1, _ = legend_row([
        ("line", f"相机中心（{ctx['n']} 帧）", C_TRACK),
        ("arrow", "视线方向（均指向锚点）", C_ARR),
        ("dot", "起点 frame 0", C_START),
        ("dot", f"末帧 frame {ctx['n'] - 1}", C_END),
    ], 60, 545, color_var=MUTED)
    row2, cx2 = legend_row([
        ("ring", "anchor_point（环绕轴心）", C_ANC),
        ("ring", f"均值半径 {ctx['dist_mean']:.2f} m", C_ANC),
    ], 60, 578, color_var=MUTED)
    out.append(row1)
    out.append(row2)
    out.append(scale_bar(cx2 + 20, 578, 0.5, view, color=MUTED))
    out.append(f'<text x="840" y="604" font-size="12" fill="#9CA3AF" text-anchor="end">'
               f'横轴 = 世界 X ｜ 纵轴 = 世界 Z ｜ 等比例尺 {view.s:.1f} px/m</text>')
    out.append("</svg>")
    return "\n".join(out), H


def style_darkspace(ctx):
    """深色网格：轨迹按帧序做时间渐变，带米制网格与时间色标。"""
    W, H = 900, 620
    BG, GRID, INK, MUTED = "#0B1220", "#1E293B", "#E2E8F0", "#64748B"
    RAMP = [(0.0, (0x22, 0xD3, 0xEE)), (0.5, (0xA7, 0x8B, 0xFA)), (1.0, (0xFB, 0x71, 0x85))]

    view = View(*ctx["range"]["x"], *ctx["range"]["z"], 60, 110, 840, 500)
    out = [svg_open(W, H, BG)]
    out.append(f'<text x="40" y="42" font-size="21" font-weight="500" fill="{INK}">'
               f'{esc(ctx["id"])} · 采集轨迹俯视图（世界系 XZ）</text>')
    out.append(caption(ctx, 40, 66, MUTED))

    x0, y0, bw, bh = view.rect()
    out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="10" '
               f'fill="#0F172A" stroke="#334155" stroke-width="1"/>')
    clip = f"clip_{ctx['id'][:8]}_dk"
    out.append(f'<clipPath id="{clip}"><rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="10"/></clipPath>')
    out.append(f'<g clip-path="url(#{clip})">')

    # 0.5 m 网格（世界坐标对齐，不是按像素硬画）
    import math as _m
    gx = _m.floor(view.umin / 0.5) * 0.5
    while gx <= view.umax:
        x, _ = view.p(gx, view.vmin)
        out.append(f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y0 + bh:.1f}" stroke="{GRID}" stroke-width="0.6" opacity="0.7"/>')
        out.append(f'<text x="{x + 3:.1f}" y="{y0 + bh - 6:.1f}" font-size="10" fill="#475569">{gx:.1f}</text>')
        gx += 0.5
    gz = _m.floor(view.vmin / 0.5) * 0.5
    while gz <= view.vmax:
        _, y = view.p(view.umin, gz)
        out.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + bw:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="0.6" opacity="0.7"/>')
        out.append(f'<text x="{x0 + 5:.1f}" y="{y - 4:.1f}" font-size="10" fill="#475569">{gz:.1f}</text>')
        gz += 0.5

    # 分段着色 = 时间渐变
    seg = []
    for i in range(ctx["n"] - 1):
        x1, y1 = view.p(ctx["pos"][i][0], ctx["pos"][i][2])
        x2, y2 = view.p(ctx["pos"][i + 1][0], ctx["pos"][i + 1][2])
        seg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                   f'stroke="{ramp_color(i / max(1, ctx["n"] - 1), RAMP)}" stroke-width="2.6" stroke-linecap="round"/>')
    out.append('<g opacity="0.95">' + "".join(seg) + '</g>')

    arr = []
    for i in range(0, ctx["n"], 20):
        x, y = view.p(ctx["pos"][i][0], ctx["pos"][i][2])
        d = ctx["look"][i]
        L = math.hypot(d[0], d[2]) or 1.0
        ex, ey = x + d[0] / L * 0.26 * view.s, y + d[2] / L * 0.26 * view.s
        arr.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="#94A3B8" '
                   f'stroke-width="1.1" opacity="0.42"/>')
    out.append("".join(arr))
    out.append('</g>')

    _rx, _ry, _rw, _rh = view.rect()
    out.append(draw_world_origin(ctx, view, lambda x, y, z: (x, z), (_rx, _ry, _rw, _rh), INK, MUTED))

    if ctx["anchor"]:
        ax, ay = view.p(ctx["anchor"][0], ctx["anchor"][2])
        r = ctx["dist_mean"] * view.s
        out.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="{r:.1f}" fill="none" stroke="#FBBF24" '
                   f'stroke-width="1.2" stroke-dasharray="5 5" opacity="0.55"/>')
        out.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="7" fill="none" stroke="#FBBF24" stroke-width="2"/>')
        out.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="2" fill="#FBBF24"/>')
        out.append(f'<text x="{ax + 14:.1f}" y="{ay - 10:.1f}" font-size="12" fill="#FBBF24">anchor_point</text>')

    sx, sy = view.p(ctx["pos"][0][0], ctx["pos"][0][2])
    ex, ey = view.p(ctx["pos"][-1][0], ctx["pos"][-1][2])
    out.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5.5" fill="#34D399" stroke="{BG}" stroke-width="2"/>')
    out.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="5.5" fill="#FB7185" stroke="{BG}" stroke-width="2"/>')
    out.append(f'<text x="{sx + 12:.1f}" y="{sy + 5:.1f}" font-size="12" fill="#34D399">frame 0</text>')
    out.append(f'<text x="{ex + 12:.1f}" y="{ey + 5:.1f}" font-size="12" fill="#FB7185">frame {ctx["n"] - 1}</text>')

    out.append(f'<text x="60" y="545" font-size="13" fill="{MUTED}">时间渐变（frame 0 → {ctx["n"] - 1}）</text>')
    bx, by, bw2 = 260, 541, 220
    for k in range(60):
        c = ramp_color(k / 59, RAMP)
        out.append(f'<rect x="{bx + bw2 * k / 60:.1f}" y="{by - 6}" width="{bw2 / 60 + 1:.2f}" height="12" fill="{c}"/>')
    out.append(f'<text x="{bx}" y="{by + 22}" font-size="11" fill="{MUTED}">0</text>')
    out.append(f'<text x="{bx + bw2}" y="{by + 22}" font-size="11" fill="{MUTED}" text-anchor="end">{ctx["n"] - 1}</text>')
    out.append(legend_row([
        ("ring", f"anchor_point ＋均值半径 {ctx['dist_mean']:.2f} m", "#FBBF24"),
        ("arrow", "视线方向", "#94A3B8"),
    ], 520, 541, color_var=MUTED)[0])
    out.append(scale_bar(60, 590, 0.5, view, color=MUTED))
    out.append(f'<text x="840" y="604" font-size="12" fill="#475569" text-anchor="end">'
               f'网格 0.5 m ｜ 横轴 = 世界 X ｜ 纵轴 = 世界 Z ｜ 世界 +Y 向上</text>')
    out.append("</svg>")
    return "\n".join(out), H


def style_fov(ctx):
    """浅色 + 点云底图 + 每 15 帧画水平视锥扇形，能看出「覆盖是否均匀」。"""
    W, H = 900, 620
    C_TRACK, C_ANC = "#1D4ED8", "#B45309"
    INK, MUTED, LINE = "#111827", "#4B5563", "#D1D5DB"
    half = ctx["fov_h"] / 2.0

    view = View(*ctx["range"]["x"], *ctx["range"]["z"], 60, 110, 840, 500)
    out = [svg_open(W, H, "#FFFFFF")]
    out.append(f'<text x="40" y="42" font-size="21" font-weight="500" fill="{INK}">'
               f'{esc(ctx["id"])} · 采集轨迹与视锥俯视图（世界系 XZ）</text>')
    out.append(caption(ctx, 40, 66, MUTED))

    x0, y0, bw, bh = view.rect()
    out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="10" '
               f'fill="#FAFAF9" stroke="{LINE}" stroke-width="1"/>')

    clip = f'clip_path_{ctx["id"][:8]}'
    out.append(f'<clipPath id="{clip}"><rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="10"/></clipPath>')
    out.append(f'<g clip-path="url(#{clip})">')

    cs = cloud_screen_for(view, ctx)
    if cs:
        out.append("".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1" fill="#B4B2A9" opacity="0.5"/>'
                           for x, y in cs))

    cone, lines = [], []
    for i in range(0, ctx["n"], 15):
        x, y = view.p(ctx["pos"][i][0], ctx["pos"][i][2])
        d = ctx["look"][i]
        base = math.degrees(math.atan2(d[2], d[0]))          # 视线在 XZ 面内的方位角
        R = 1.05 * view.s
        a1, a2 = math.radians(base - half), math.radians(base + half)
        cone.append('<path d="M%.1f,%.1f L%.1f,%.1f A%.1f,%.1f 0 0 1 %.1f,%.1f Z" fill="#93C5FD" '
                    'opacity="0.10"/>' % (x, y, x + math.cos(a1) * R, y + math.sin(a1) * R,
                                          R, R, x + math.cos(a2) * R, y + math.sin(a2) * R))
        # 只画两条扇边 + 一条中心虚线，比实心块干净得多
        for a in (a1, a2):
            lines.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + math.cos(a) * R:.1f}" '
                         f'y2="{y + math.sin(a) * R:.1f}" stroke="#93C5FD" stroke-width="0.9" opacity="0.75"/>')
        lines.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + math.cos(math.radians(base)) * R:.1f}" '
                     f'y2="{y + math.sin(math.radians(base)) * R:.1f}" stroke="#60A5FA" stroke-width="0.8" '
                     f'opacity="0.45" stroke-dasharray="4 4"/>')
    out.append("".join(cone))
    out.append("".join(lines))

    pts = " ".join("%.1f,%.1f" % view.p(p[0], p[2]) for p in ctx["pos"])
    out.append(f'<polyline points="{pts}" fill="none" stroke="{C_TRACK}" stroke-width="2" stroke-linejoin="round"/>')
    for i in range(0, ctx["n"], 15):
        x, y = view.p(ctx["pos"][i][0], ctx["pos"][i][2])
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{C_TRACK}"/>')
    out.append('</g>')

    _rx, _ry, _rw, _rh = view.rect()
    out.append(draw_world_origin(ctx, view, lambda x, y, z: (x, z), (_rx, _ry, _rw, _rh), INK, MUTED))

    if ctx["anchor"]:
        ax, ay = view.p(ctx["anchor"][0], ctx["anchor"][2])
        out.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="7" fill="none" stroke="{C_ANC}" stroke-width="2"/>')
        out.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="2" fill="{C_ANC}"/>')
        out.append(f'<text x="{ax + 14:.1f}" y="{ay - 10:.1f}" font-size="12" fill="{C_ANC}">anchor_point</text>')

    sx, sy = view.p(ctx["pos"][0][0], ctx["pos"][0][2])
    ex, ey = view.p(ctx["pos"][-1][0], ctx["pos"][-1][2])
    out.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6" fill="#4D7C0F"/>')
    out.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="6" fill="#BE123C"/>')

    n_cone = len(range(0, ctx["n"], 15))
    row1, cx1 = legend_row([
        ("line", f"相机轨迹（{ctx['n']} 帧）", C_TRACK),
        ("wedge", f"每 15 帧画一次水平视锥（共 {n_cone} 个）", "#93C5FD"),
        ("dot", "pcd.ply 点云（AR 稠密点云）", "#B4B2A9"),
    ], 60, 545, color_var=MUTED)
    row2, cx2 = legend_row([
        ("ring", "anchor_point", C_ANC),
        ("dot", "起点 frame 0", "#4D7C0F"),
        ("dot", f"末帧 frame {ctx['n'] - 1}", "#BE123C"),
    ], 60, 578, color_var=MUTED)
    out.append(row1)
    out.append(row2)
    out.append(scale_bar(cx2 + 20, 578, 0.5, view, color=MUTED))
    out.append(f'<text x="840" y="604" font-size="12" fill="#9CA3AF" text-anchor="end">'
               f'扇形 = 图像水平视场 {ctx["fov_h"]:.1f}°（投影到 XZ 面）</text>')
    out.append("</svg>")
    return "\n".join(out), H


def style_iso(ctx):
    """等轴测：把 (X, Y, Z) 投到 2D，点云 + 相机高度 + 垂直投影线都看得到。"""
    W, H = 900, 660
    INK, MUTED, LINE = "#111827", "#4B5563", "#D1D5DB"

    # 视野只按相机活动范围（与 combo 一致）：大场景点云会把轨迹压成角落一小团
    u0, u1, v0, v1, floor_guess = iso_extent(ctx)
    view = View(u0, u1, v0, v1, 60, 110, 860, 545)

    out = [svg_open(W, H, "#FFFFFF")]
    out.append(f'<text x="40" y="42" font-size="21" font-weight="500" fill="{INK}">'
               f'{esc(ctx["id"])} · 采集轨迹等轴测视图（含世界 Y 高度）</text>')
    out.append(caption(ctx, 40, 66, MUTED))
    x0, y0, bw, bh = view.rect()
    out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="10" '
               f'fill="#FAFAF9" stroke="{LINE}"/>')
    geom, floor = draw_iso_geometry(ctx, view, f"clip_{ctx['id'][:8]}_iso",
                                    cloud_screen=cloud_screen_for(view, ctx, proj=iso_proj_pt),
                                    origin=True)
    out.append(geom)

    row1, cx1 = legend_row([
        ("line", f"相机轨迹（{ctx['n']} 帧）", PAL_LIGHT["track"]),
        ("dot", "pcd.ply 点云", PAL_LIGHT["cloud"]),
        ("ring", "anchor_point", PAL_LIGHT["anc"]),
    ], 60, 578, color_var=MUTED)
    row2, _ = legend_row([
        ("dot", "起点 frame 0", PAL_LIGHT["start"]),
        ("dot", f"末帧 frame {ctx['n'] - 1}", PAL_LIGHT["end"]),
        ("dash", "相机到地面的垂直投影线", "#CBD5E1"),
    ], 60, 610, color_var=MUTED)
    out.append(row1)
    out.append(row2)
    out.append(scale_bar(cx1 + 20, 578, 0.5, view, color=MUTED))
    out.append(f'<text x="860" y="640" font-size="12" fill="#9CA3AF" text-anchor="end">'
               f'等轴测投影：(X−Z)·cos30°, (X+Z)·sin30° − Y ｜ 地面 Y = {floor:.2f} m</text>')
    out.append("</svg>")
    return "\n".join(out), H


def legend_col(items, x, y, fs=12, line_h=24, color_var="#4B5563"):
    """竖排图例（给绘图区右侧的窄列用）。返回 (svg, 末行之后的 y)。"""
    out, cy = [], y
    for kind, label, color in items:
        if kind == "line":
            out.append(f'<line x1="{x}" y1="{cy}" x2="{x + 22}" y2="{cy}" stroke="{color}" stroke-width="2.2"/>')
            out.append(f'<circle cx="{x + 11}" cy="{cy}" r="3" fill="{color}"/>')
        elif kind == "arrow":
            out.append(f'<line x1="{x}" y1="{cy}" x2="{x + 22}" y2="{cy}" stroke="{color}" stroke-width="1.4"/>')
            out.append(f'<path d="M{x + 16},{cy - 4} L{x + 22},{cy} L{x + 16},{cy + 4}" fill="none" '
                       f'stroke="{color}" stroke-width="1.4" stroke-linecap="round"/>')
        elif kind == "dot":
            out.append(f'<circle cx="{x + 11}" cy="{cy}" r="4.5" fill="{color}"/>')
        elif kind == "ring":
            out.append(f'<circle cx="{x + 11}" cy="{cy}" r="6" fill="none" stroke="{color}" stroke-width="1.8"/>')
        elif kind == "dash":
            out.append(f'<line x1="{x}" y1="{cy}" x2="{x + 22}" y2="{cy}" stroke="{color}" stroke-width="1.2" stroke-dasharray="4 4"/>')
        out.append(f'<text x="{x + 30}" y="{cy + 4}" font-size="{fs}" fill="{color_var}">{esc(label)}</text>')
        cy += line_h
    return "\n".join(out), cy


def legend_grid(items, x, y, col_w, cols, line_h=26, fs=12, color_var="#4B5563"):
    """把图例项按固定列宽摆成 cols 列的网格（用于把图例摊开到整幅宽度上）。"""
    out = []
    for idx, (kind, label, color) in enumerate(items):
        r, c = divmod(idx, cols)
        svg, _ = legend_col([(kind, label, color)], x + c * col_w, y + r * line_h,
                            fs=fs, line_h=line_h, color_var=color_var)
        out.append(svg)
    return "\n".join(out)


def style_combo(ctx):
    """双联图（上下排列）：上＝俯视图（世界系 XZ，轨迹按帧序时间渐变），下＝等轴测（含 Y 高度）。

    排版约定（2026-09-22 定稿）：
    · 两块绘图区**统一预留框 800×450**，图框就按预留框画（两块一样大、左右对齐）。
    · 俯视图数据通常窄于 1.78 的长宽比，两侧会空出来 → 把点云画进去填满，并作为背景裁到预留框。
    · 两个图例都**铺在各自绘图区下方、占满整幅宽度**（俯视图 5 项一行 + 色条行；
      等轴测 2 行 × 3 列）。
    """
    INK, MUTED, LINE = "#111827", "#4B5563", "#D1D5DB"
    PANEL_W, PANEL_H = 800.0, 450.0      # 统一预留框
    BOX_L = 40.0
    BOX_R = BOX_L + PANEL_W
    TOP_Y = 120.0
    W = BOX_R + 40

    # 俯视图取景 = 相机范围 ∪ 锚点±均值半径。
    # 画了均值半径环就得让环完整入框：相机只沿锚点一侧走一小段弧时（浅弧样例），
    # 只按相机范围取景会把锚点和环一起裁到框外，画面上只剩一条大轨迹。
    x_lo, x_hi = ctx["range"]["x"]
    z_lo, z_hi = ctx["range"]["z"]
    if ctx["anchor"]:
        ax_, az_, r_ = ctx["anchor"][0], ctx["anchor"][2], ctx["dist_mean"]
        x_lo, x_hi = min(x_lo, ax_ - r_), max(x_hi, ax_ + r_)
        z_lo, z_hi = min(z_lo, az_ - r_), max(z_hi, az_ + r_)

    # 可选：把视角约束范围叠进俯视图（VLIMIT_MODE=band|edges|angle|rings|hull 时生效）。
    # 不设该变量时行为与旧版完全一致。叠进来就把取景再撑到 ρ_hi，否则约束外弧会被裁。
    vl_combo = compute_view_limit(ctx) if VLIMIT_MODE in LIMIT_MODES else None
    if vl_combo is not None and VLIMIT_MODE != "shell":
        r_ = vl_combo["rho_hi"]
        ax_, az_ = vl_combo["target"][0], vl_combo["target"][2]
        x_lo, x_hi = min(x_lo, ax_ - r_), max(x_hi, ax_ + r_)
        z_lo, z_hi = min(z_lo, az_ - r_), max(z_hi, az_ + r_)
    view_top = View(x_lo, x_hi, z_lo, z_hi, BOX_L, TOP_Y, BOX_R, TOP_Y + PANEL_H)

    # 等轴测：视野由 iso_extent() 决定（只按相机活动范围，点云作背景）
    u0, u1, v0, v1, _ = iso_extent(ctx)
    if vl_combo is not None and VLIMIT_MODE == "shell":
        u0, u1, v0, v1, _ = _limit_iso_extent(ctx, vl_combo)
    iso_y = TOP_Y + PANEL_H + 170.0
    view_iso = View(u0, u1, v0, v1, BOX_L, iso_y, BOX_R, iso_y + PANEL_H)

    out = [svg_open(W, 100, "#FFFFFF")]
    out.append(f'<text x="40" y="42" font-size="21" font-weight="500" fill="{INK}">'
               f'{esc(ctx["id"])} · 采集轨迹图（俯视 + 等轴测）</text>')
    out.append(caption(ctx, 40, 66, MUTED))

    # ── ① 俯视图 ──
    out.append(f'<text x="40" y="104" font-size="13" font-weight="500" fill="{INK}">'
               f'① 俯视图（世界系 XZ · 轨迹按帧序时间渐变）</text>')
    panel_top = (BOX_L, TOP_Y, PANEL_W, PANEL_H)
    out.append(f'<rect x="{BOX_L}" y="{TOP_Y}" width="{PANEL_W}" height="{PANEL_H}" rx="10" '
               f'fill="#FAFAF9" stroke="{LINE}"/>')
    if vl_combo is not None and VLIMIT_MODE != "shell":
        # 约束层垫在轨迹下面：轨迹与点云照旧压在最上层，观感几乎不变
        cid = f"clip_{ctx['id'][:8]}_ctlim"
        out.append(f'<clipPath id="{cid}"><rect x="{BOX_L}" y="{TOP_Y}" width="{PANEL_W}" '
                   f'height="{PANEL_H}" rx="10"/></clipPath>')
        out.append(f'<g clip-path="url(#{cid})">'
                   f'{LIMIT_DRAWS[VLIMIT_MODE](ctx, vl_combo, view_top)}</g>')
    out.append(draw_top_geometry(ctx, view_top, f"clip_{ctx['id'][:8]}_ct", ramp=RAMP_LIGHT,
                                 cloud_screen=cloud_screen_for(view_top, ctx), clip_rect=panel_top,
                                 origin=True,
                                 ring=vl_combo is None))
    if VLIMIT_MODE in LIMIT_MODES:
        out.append(f'<text x="{BOX_R}" y="{TOP_Y + PANEL_H - 14:.0f}" font-size="12" fill="{LIM_INK}" '
                   f'text-anchor="end">视角约束范围（{esc(LIMIT_LABELS[VLIMIT_MODE])}·'
                   f'实测区间 ±{VL_AZ_PAD:g}°/{VL_EL_PAD:g}°/{VL_R_PAD * 100:g}%）</text>')
    out.append(axis_hints(view_top, PAL_LIGHT["axis"]))
    out.append(scale_bar(BOX_L + 16, TOP_Y + PANEL_H - 44, 0.5, view_top, color=MUTED))

    # 俯视图图例：铺在绘图区下方，5 项等距排满整幅宽度
    ly = TOP_Y + PANEL_H + 40
    out.append(legend_grid([
        ("line", f"相机轨迹（{ctx['n']} 帧）", PAL_LIGHT["track"]),
        ("arrow", "视线方向", PAL_LIGHT["arrow"]),
        # 叠了视角约束就用它顶掉「均值半径环」那一项：两者都是 target 周围的参考圈，
        # 同时画既冗余又互相干扰（约束外缘 ρ_hi 与均值半径环常常只差几个像素）。
        (("dash", "视角约束边界", LIM_INK) if vl_combo is not None
         else ("ring", f"均值半径 {ctx['dist_mean']:.2f} m", PAL_LIGHT["ring"])),
        ("dot", "起点 frame 0", PAL_LIGHT["start"]),
        ("dot", f"末帧 frame {ctx['n'] - 1}", PAL_LIGHT["end"]),
    ], BOX_L, ly, PANEL_W / 5, 5, color_var=MUTED))
    out.append(ramp_bar(BOX_L, ly + 54, 200, 12, RAMP_LIGHT, "轨迹颜色 = 帧序", MUTED, fs=12))
    out.append(f'<text x="{BOX_L + 250}" y="{ly + 66}" font-size="12" fill="#9CA3AF">'
               f'点云 pcd.ply（背景，超出绘图区已裁）｜ 世界 +Y 向上 ｜ 等比例尺 {view_top.s:.1f} px/m</text>')

    # ── ② 等轴测 ──
    out.append(f'<text x="40" y="{iso_y - 16:.0f}" font-size="13" font-weight="500" fill="{INK}">'
               f'② 等轴测视图（含世界 Y 高度）</text>')
    panel_iso = (BOX_L, iso_y, PANEL_W, PANEL_H)
    out.append(f'<rect x="{BOX_L}" y="{iso_y}" width="{PANEL_W}" height="{PANEL_H}" rx="10" '
               f'fill="#FAFAF9" stroke="{LINE}"/>')
    if vl_combo is not None and VLIMIT_MODE == "shell":
        cid = f"clip_{ctx['id'][:8]}_cilim"
        out.append(f'<clipPath id="{cid}"><rect x="{BOX_L}" y="{iso_y:.0f}" width="{PANEL_W}" '
                   f'height="{PANEL_H}" rx="10"/></clipPath>')
        out.append(f'<g clip-path="url(#{cid})">'
                   f'{LIMIT_DRAWS["shell"](ctx, vl_combo, view_iso)}</g>')
    geom, floor = draw_iso_geometry(ctx, view_iso, f"clip_{ctx['id'][:8]}_ci", clip_rect=panel_iso,
                                    cloud_screen=cloud_screen_for(view_iso, ctx, proj=iso_proj_pt),
                                    origin=True)
    out.append(geom)
    out.append(scale_bar(BOX_L + 16, iso_y + PANEL_H - 44, 0.5, view_iso, color=MUTED))

    # 等轴测图例：同样铺在绘图区下方（2 行 × 3 列）
    ly2 = iso_y + PANEL_H + 40
    out.append(legend_grid([
        ("line", f"相机轨迹（{ctx['n']} 帧）", PAL_LIGHT["track"]),
        ("dot", "点云 pcd.ply（背景）", PAL_LIGHT["cloud"]),
        ("ring", "anchor_point", PAL_LIGHT["anc"]),
        ("dash", "地面投影线", "#CBD5E1"),
        ("dot", "起点 frame 0", PAL_LIGHT["start"]),
        ("dot", f"末帧 frame {ctx['n'] - 1}", PAL_LIGHT["end"]),
    ], BOX_L, ly2, PANEL_W / 3, 3, color_var=MUTED))
    out.append(f'<text x="40" y="{ly2 + 66}" font-size="12" fill="#9CA3AF">'
               f'等轴测投影：(X−Z)·cos30°, (X+Z)·sin30° − Y ｜ 地面 Y = {floor:.2f} m ｜ '
               f'横轴 = 世界 X ｜ 纵轴 = 世界 Z（仅俯视图）｜ 两块图均为等比例尺、同为 {PANEL_W:.0f}×{PANEL_H:.0f} 预留框</text>')
    H = ly2 + 92
    out[0] = svg_open(W, H, "#FFFFFF")
    out.append("</svg>")
    return "\n".join(out), H


# ── 四联对照图（2×2）的排版常量 ──
QUAD_PW, QUAD_PH = 800.0, 450.0      # 单块绘图区预留框（与 combo 同尺寸，便于沿用原有排法）
QUAD_M = 40.0                        # 画布边距
QUAD_GAP = 72.0                      # 列间距
QUAD_LEG = 112.0                     # 每行面板下方图例带高度
QUAD_ROW_GAP = 56.0                  # 行间距（含第二行的小标题）
QUAD_TOP_Y = 120.0
# 四联图的约束层配色：**不用** vlimit 的紫（LIM_INK）。左列轨迹是按帧序的蓝→紫→玫红渐变，
# 其中段正好也是紫，紫虚线约束压上去会和轨迹糊成一片、分不清哪条是轨迹。改用青绿，
# 与渐变三色 + 灰点云 + 琥珀锚点全都分得开。
QUAD_INK = "#0F766E"
QUAD_FILL = "#14B8A6"


def style_quad(ctx):
    """四联对照图（2×2）：左列＝原始采集轨迹（俯视 / 等轴测），右列＝同一轨迹上的视角约束。

    排版约定（2026-09-28）：
    · 四块绘图区统一 800×450 预留框，2×2 严格对齐（行高 = 面板 + 图例带）。
    · **四张图共用同一取景与比例尺**。这张图存在的意义就是对照「约束落在轨迹的哪一段」，
      左右各自取景会让同一段轨迹在两张俯视图里大小不同，对照反而失真。代价是左列比
      combo 单看时小一圈（约束范围通常是相机范围的 1.2–1.9 倍）。
    · 右列只比左列多一层约束，并省掉「均值半径环」——两者都是 target 周围的参考圈，
      同时画既冗余又互相干扰（约束外缘 ρ_hi 与均值半径常只差几个像素）。
    · 约束层画在底图**之下**：轨迹与点云照旧压在最上层，对原有轨迹的观感影响最小。
    """
    INK, MUTED, LINE = "#111827", "#4B5563", "#D1D5DB"
    PW, PH, M, GAP, LEG = QUAD_PW, QUAD_PH, QUAD_M, QUAD_GAP, QUAD_LEG
    X_L = M
    X_R = X_L + PW + GAP
    W = X_R + PW + M
    TOP_Y = QUAD_TOP_Y
    ROW2_Y = TOP_Y + PH + LEG + QUAD_ROW_GAP
    H = ROW2_Y + PH + LEG + 28.0

    vl = compute_view_limit(ctx)
    sid = ctx["id"][:8]

    # ── 共用取景：相机活动范围 ∪ 锚点±均值半径 ∪ 约束范围 ──
    x_lo, x_hi = ctx["range"]["x"]
    z_lo, z_hi = ctx["range"]["z"]
    if ctx["anchor"]:
        ax_, az_, r_ = ctx["anchor"][0], ctx["anchor"][2], ctx["dist_mean"]
        x_lo, x_hi = min(x_lo, ax_ - r_), max(x_hi, ax_ + r_)
        z_lo, z_hi = min(z_lo, az_ - r_), max(z_hi, az_ + r_)
    tx_, tz_, rp_ = vl["target"][0], vl["target"][2], vl["rho_hi"]
    top_box = (min(x_lo, tx_ - rp_), max(x_hi, tx_ + rp_),
               min(z_lo, tz_ - rp_), max(z_hi, tz_ + rp_))
    u0, u1, v0, v1, floor = _limit_iso_extent(ctx, vl)
    iso_box = (u0, u1, v0, v1)

    def vw(box, x, y):
        """同一数据范围 → 指定像素框的等效 View（两列得到严格相同的比例尺）。

        面板必须按 (x, y, w, h) 传 —— 见 _fit_view 的 ⚠️。
        """
        return _fit_view(box[0], box[1], box[2], box[3], (x, y, PW, PH))

    v_top_l, v_top_r = vw(top_box, X_L, TOP_Y), vw(top_box, X_R, TOP_Y)
    v_iso_l, v_iso_r = vw(iso_box, X_L, ROW2_Y), vw(iso_box, X_R, ROW2_Y)

    def panel(x, y):
        return (f'<rect x="{x}" y="{y}" width="{PW}" height="{PH}" rx="10" '
                f'fill="#FAFAF9" stroke="{LINE}"/>')

    def clipped(cid, x, y, body):
        return (f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{PW}" height="{PH}" '
                f'rx="10"/></clipPath>\n<g clip-path="url(#{cid})">{body}</g>')

    def head(x, y, num, txt):
        return (f'<text x="{x}" y="{y - 12:.0f}" font-size="13" font-weight="500" fill="{INK}">'
                f'{num} {esc(txt)}</text>')

    out = [svg_open(W, 100, "#FFFFFF")]
    out.append(f'<text x="{M}" y="42" font-size="21" font-weight="500" fill="{INK}">'
               f'{esc(ctx["id"])} · 采集轨迹 × 视角约束对照图（2×2）</text>')
    out.append(caption(ctx, M, 66, MUTED))
    out.append(f'<text x="{M}" y="88" font-size="12" fill="#9CA3AF">'
               f'左列＝原始采集轨迹 ｜ 右列＝视角约束范围（{esc(LIMIT_LABELS["angle"])} / '
               f'{esc(LIMIT_LABELS["shell"])}）｜ 四张图共用同一取景与比例尺 '
               f'{v_top_l.s:.1f} px/m，可逐点对照 ｜ 约束层画在轨迹下层 ｜ '
               f'约束用青绿以区分轨迹的蓝→紫→玫红渐变</text>')

    # ── ① 左上：俯视轨迹（与原 combo 完全一致）──
    out.append(head(X_L, TOP_Y, "①", "俯视图 · 采集轨迹（世界系 XZ · 轨迹按帧序时间渐变）"))
    out.append(panel(X_L, TOP_Y))
    out.append(draw_top_geometry(ctx, v_top_l, f"clip_{sid}_q1", ramp=RAMP_LIGHT,
                                 cloud_screen=cloud_screen_for(v_top_l, ctx),
                                 clip_rect=(X_L, TOP_Y, PW, PH), origin=True))
    out.append(axis_hints(v_top_l, PAL_LIGHT["axis"]))
    out.append(scale_bar(X_L + 16, TOP_Y + PH - 44, 0.5, v_top_l, color=MUTED))

    # ── ② 右上：俯视图 + 角度标注 ──
    out.append(head(X_R, TOP_Y, "②", "俯视图 · 视角约束范围（角度标注）"))
    out.append(panel(X_R, TOP_Y))
    out.append(clipped(f"clip_{sid}_q2", X_R, TOP_Y,
                       _lim_angle(ctx, vl, v_top_r, ink=QUAD_INK, rho_text=False)))
    out.append(draw_top_geometry(ctx, v_top_r, f"clip_{sid}_q2b", ramp=RAMP_LIGHT,
                                 cloud_screen=cloud_screen_for(v_top_r, ctx),
                                 clip_rect=(X_R, TOP_Y, PW, PH), origin=True, ring=False))
    out.append(axis_hints(v_top_r, PAL_LIGHT["axis"], flip=True))
    out.append(scale_bar(X_R + 16, TOP_Y + PH - 44, 0.5, v_top_r, color=MUTED))

    # ── ③ 左下：等轴测轨迹（与原 combo 完全一致）──
    out.append(head(X_L, ROW2_Y, "③", "等轴测视图 · 采集轨迹（含世界 Y 高度）"))
    out.append(panel(X_L, ROW2_Y))
    geom_l, _ = draw_iso_geometry(ctx, v_iso_l, f"clip_{sid}_q3",
                                  clip_rect=(X_L, ROW2_Y, PW, PH),
                                  cloud_screen=cloud_screen_for(v_iso_l, ctx, proj=iso_proj_pt),
                                  origin=True)
    out.append(geom_l)
    out.append(scale_bar(X_L + 16, ROW2_Y + PH - 44, 0.5, v_iso_l, color=MUTED))

    # ── ④ 右下：等轴测 + 3D 球壳扇块 ──
    out.append(head(X_R, ROW2_Y, "④", "等轴测视图 · 视角约束范围（3D 球壳扇块）"))
    out.append(panel(X_R, ROW2_Y))
    out.append(clipped(f"clip_{sid}_q4", X_R, ROW2_Y,
                       _lim_shell(ctx, vl, v_iso_r, ink=QUAD_INK, fill=QUAD_FILL)))
    geom_r, _ = draw_iso_geometry(ctx, v_iso_r, f"clip_{sid}_q4b",
                                  clip_rect=(X_R, ROW2_Y, PW, PH),
                                  cloud_screen=cloud_screen_for(v_iso_r, ctx, proj=iso_proj_pt),
                                  origin=True)
    out.append(geom_r)
    out.append(scale_bar(X_R + 16, ROW2_Y + PH - 44, 0.5, v_iso_r, color=MUTED))

    # ── 图例带：左列沿用 combo 的原排法，右列给约束数值 ──
    u = vl["uwa"]
    ly = TOP_Y + PH + 40
    out.append(legend_grid([
        ("line", f"相机轨迹（{ctx['n']} 帧）", PAL_LIGHT["track"]),
        ("arrow", "视线方向", PAL_LIGHT["arrow"]),
        ("ring", f"均值半径 {ctx['dist_mean']:.2f} m", PAL_LIGHT["ring"]),
        ("dot", "起点 frame 0", PAL_LIGHT["start"]),
        ("dot", f"末帧 frame {ctx['n'] - 1}", PAL_LIGHT["end"]),
    ], X_L, ly, PW / 5, 5, color_var=MUTED))
    out.append(ramp_bar(X_L, ly + 54, 200, 12, RAMP_LIGHT, "轨迹颜色 = 帧序", MUTED, fs=12))
    out.append(f'<text x="{X_L + 250}" y="{ly + 66}" font-size="12" fill="#9CA3AF">'
               f'点云 pcd.ply（背景，超出绘图区已裁）｜ 世界 +Y 向上 ｜ 等比例尺 {v_top_l.s:.1f} px/m</text>')

    az_txt = ("整圈 360°" if vl["full_az"]
              else f'{vl["az_lo"]:.1f}°→{vl["az_hi"]:.1f}°（跨度 {vl["az_hi"] - vl["az_lo"]:.1f}°）')
    out.append(f'<text x="{X_R}" y="{ly}" font-size="12.5" fill="{QUAD_INK}">'
               f'视角约束范围（{esc(LIMIT_LABELS["angle"])}）· {esc(LIMIT_NOTES["angle"][0])} · '
               f'区间 = 各帧实测 min/max ± 余量 {VL_AZ_PAD:g}°/{VL_EL_PAD:g}°/{VL_R_PAD * 100:g}%'
               f'{" ｜ ⚠️ 方位角近似整圈" if vl["full_az"] else ""}</text>')
    out.append(f'<text x="{X_R}" y="{ly + 24}" font-size="12" fill="{QUAD_INK}">'
               f'az {az_txt} ｜ el {vl["el_lo"]:.1f}°→{vl["el_hi"]:.1f}° ｜ '
               f'ρ {vl["rho_lo"]:.2f}→{vl["rho_hi"]:.2f} m ｜ '
               f'R {vl["r_lo"]:.2f}→{vl["r_hi"]:.2f} m</text>')
    row, cx = legend_row([("ring", "target", "#B45309"),
                          ("dash", "约束边界", QUAD_INK),
                          ("line", "相机轨迹", PAL_LIGHT["track"])],
                         X_R, ly + 54, fs=12, gap=22, color_var=MUTED)
    out.append(row)
    out.append(scale_bar(cx + 16, ly + 54, 0.5, v_top_r, color=MUTED))

    ly2 = ROW2_Y + PH + 40
    out.append(legend_grid([
        ("line", f"相机轨迹（{ctx['n']} 帧）", PAL_LIGHT["track"]),
        ("dot", "点云 pcd.ply（背景）", PAL_LIGHT["cloud"]),
        ("ring", "anchor_point", PAL_LIGHT["anc"]),
        ("dash", "地面投影线", "#CBD5E1"),
        ("dot", "起点 frame 0", PAL_LIGHT["start"]),
        ("dot", f"末帧 frame {ctx['n'] - 1}", PAL_LIGHT["end"]),
    ], X_L, ly2, PW / 3, 3, color_var=MUTED))
    out.append(f'<text x="{X_L}" y="{ly2 + 66}" font-size="12" fill="#9CA3AF">'
               f'等轴测投影：(X−Z)·cos30°, (X+Z)·sin30° − Y ｜ 地面 Y = {floor:.2f} m</text>')

    out.append(f'<text x="{X_R}" y="{ly2}" font-size="12.5" fill="{QUAD_INK}">'
               f'视角约束范围（{esc(LIMIT_LABELS["shell"])}）· {esc(LIMIT_NOTES["shell"][0])} · '
               f'内层球面 R {vl["r_lo"]:.2f} m / 外层球面 R {vl["r_hi"]:.2f} m</text>')
    out.append(f'<text x="{X_R}" y="{ly2 + 24}" font-size="12" fill="#9CA3AF">'
               f'UWA view_limits 对照（实测值，未加余量）：Phi {u["phi"][0]:.1f}°→{u["phi"][1]:.1f}° ｜ '
               f'Theta {u["theta"][0]:.1f}°→{u["theta"][1]:.1f}° ｜ '
               f'Radius {u["radius"][0]:.3f}→{u["radius"][1]:.3f} m</text>')
    row2, cx2 = legend_row([("ring", "target（球壳中心）", "#B45309"),
                            ("dash", "3D 球壳扇块", QUAD_INK),
                            ("line", "相机轨迹", PAL_LIGHT["track"])],
                           X_R, ly2 + 54, fs=12, gap=22, color_var=MUTED)
    out.append(row2)
    out.append(scale_bar(cx2 + 16, ly2 + 54, 0.5, v_iso_r, color=MUTED))

    out[0] = svg_open(W, H, "#FFFFFF")
    out.append("</svg>")
    return "\n".join(out), H


RENDERERS = {
    "quad": style_quad,
    "combo": style_combo,
    "minimal": style_minimal,
    "darkspace": style_darkspace,
    "fov": style_fov,
    "iso": style_iso,
    "vlimit": style_vlimit,
    "lim_band": _limit_style("band"),
    "lim_edges": _limit_style("edges"),
    "lim_angle": _limit_style("angle"),
    "lim_rings": _limit_style("rings"),
    "lim_hull": _limit_style("hull"),
    "lim_shell": _limit_style("shell"),
}


# ─────────────────────────── HTML 组装 ───────────────────────────

def stat_cards(ctx) -> str:
    d = ctx["data"]
    r = ctx["range"]
    intr = ctx["intr"]
    pos = ctx["pos"]
    cen = tuple(sum(p[i] for p in pos) / len(pos) for i in range(3))
    d_origin_cen = math.dist(cen, (0.0, 0.0, 0.0))
    d_origin_anc = math.dist(ctx["anchor"], (0.0, 0.0, 0.0)) if ctx["anchor"] else float("nan")
    cards = [
        ("帧数 / 时长", f"{ctx['n']} 帧 · {ctx['duration']:.1f} s · {ctx['fps']:.2f} fps"),
        ("相机-锚点水平距离", f"{ctx['dist_min']:.2f} – {ctx['dist_max']:.2f} m（均值 {ctx['dist_mean']:.2f}）"),
        ("视线对准偏差", f"均值 {ctx['ang_mean']:.1f}° · 最大 {ctx['ang_max']:.1f}°"),
        ("轨迹范围 (X×Z)", f"{r['x'][1] - r['x'][0]:.2f} × {r['z'][1] - r['z'][0]:.2f} m"),
        ("相机高度范围 (Y)", f"{r['y'][0]:.2f} – {r['y'][1]:.2f} m（跨度 {r['y'][1] - r['y'][0]:.2f}）"),
        ("内参（全帧一致）", f"{intr['w']}×{intr['h']} · fl {ctx['fl']:.1f}px · 等效 {ctx['f_equiv']:.1f}mm · 对角 {ctx['fov_d']:.1f}°"),
        ("畸变", f"k1 {intr['k1']:.4f} · k2 {intr['k2']:.4f} · k3 {intr['k3']:.4f} · p1 {intr['p1']:.2e} · p2 {intr['p2']:.2e}"),
        ("点云 pcd.ply", (f"{ctx['ply_total']} 点（每块图最多绘 {PLY_DRAW_POINTS} 点）· "
                          f"bbox "
                          f"{ctx['bbox']['max'][0] - ctx['bbox']['min'][0]:.2f} × "
                          f"{ctx['bbox']['max'][1] - ctx['bbox']['min'][1]:.2f} × "
                          f"{ctx['bbox']['max'][2] - ctx['bbox']['min'][2]:.2f} m")
         if ctx["bbox"] else "未找到或无有效点"),
        ("世界原点 (0,0,0)", (f"距锚点 {d_origin_anc:.2f} m · 距相机质心 {d_origin_cen:.2f} m"
                              if ctx["anchor"] else f"距相机质心 {d_origin_cen:.2f} m")),
        ("采集端", f'{d.get("platform", "?")} · {str(d.get("platform_version", "?")).split("/")[5] if len(str(d.get("platform_version", "")).split("/")) > 5 else "?"} · capture_mode {d.get("capture_mode", "?")}'),
    ]
    return "\n".join(
        f'<div class="card"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>' for k, v in cards
    )


PAGE_CSS = """
:root{--ink:#111827;--muted:#4B5563;--line:#E5E7EB;--bg:#F3F4F6;--card:#FFFFFF}
*{box-sizing:border-box}
body{margin:0;padding:28px 24px 40px;background:var(--bg);color:var(--ink);
     font-family:system-ui,-apple-system,'Segoe UI','Microsoft YaHei',sans-serif}
.wrap{max-width:980px;margin:0 auto}
.wrap.wide{max-width:1820px}
.figrow{display:flex;gap:16px;align-items:flex-start}
.figrow .fig{flex:1 1 auto;min-width:0}
.figrow .side{flex:0 0 296px;display:flex;flex-direction:column;gap:8px}
.figrow .side .card{padding:9px 11px}
@media (max-width:1040px){.figrow{flex-direction:column}.figrow .side{flex:1 1 auto;width:100%}}
h1{font-size:22px;font-weight:600;margin:0 0 4px;letter-spacing:.2px}
.sub{font-size:13px;color:var(--muted);margin-bottom:16px}
.fig{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:10px 12px 4px;overflow:hidden}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px;margin-top:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.card .k{font-size:12px;color:var(--muted);margin-bottom:4px}
.card .v{font-size:13px;font-weight:500;line-height:1.5;word-break:break-all}
.foot{margin-top:18px;font-size:12px;color:#9CA3AF;line-height:1.7}
a{color:#1D4ED8;text-decoration:none}
a:hover{text-decoration:underline}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
th,td{padding:9px 12px;font-size:13px;text-align:left;border-bottom:1px solid var(--line)}
th{background:#F9FAFB;font-weight:500;color:var(--muted)}
tr:last-child td{border-bottom:none}
"""


def render_html(ctx, style, svg) -> str:
    title = f'{ctx["id"]} · {STYLE_LABELS[style]}'
    # combo（880 宽）与 quad（1752 宽）都比默认容器宽，容器跟着放宽，
    # 否则整张图被缩到 980px 宽、字变小。SVG 自身是固定宽度，不会反过来被拉伸。
    wrap_cls = "wrap wide" if style in ("combo", "quad") else "wrap"
    cards = stat_cards(ctx)
    # combo：图在左、统计卡片竖排在右侧；其它风格沿用下方卡片网格
    if style == "combo":
        body = (f'<div class="figrow"><div class="fig">{svg}</div>'
                f'<aside class="side">{cards}</aside></div>')
    else:
        body = f'<div class="fig">{svg}</div>\n<div class="grid">{cards}</div>'
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<style>{PAGE_CSS}</style></head>
<body><div class="{wrap_cls}">
<h1>{esc(ctx["id"])}</h1>
<div class="sub">Remy 采集包 · {esc(STYLE_LABELS[style])} · {ctx["n"]} 帧</div>
{body}
<div class="foot">
  图内坐标系：世界 XZ 俯视，世界 +Y 向上（重力对齐）；相机前向 = −Z（OpenGL 约定）。
  transform_matrix 第 4 列即相机光心，单位米；各帧到 anchor_point 的距离近似恒定 → 属「等距环绕」采集。<br>
  由 vggt_human/99g_plot_capture_trajectory.py 生成（纯标准库，无外部依赖）。
</div>
</div></body></html>"""


def render_index(entries, styles) -> str:
    def links(e):
        return " · ".join(
            f'<a href="{esc(e["files"][s])}">{esc(s)}</a>' for s in styles if s in e["files"]
        )

    rows = "\n".join(
        f'<tr><td><a href="{esc(next(iter(e["files"].values())))}">{esc(e["id"])}</a></td>'
        f'<td>{links(e)}</td>'
        f'<td>{e["n"]}</td><td>{e["duration"]:.1f} s</td><td>{e["fps"]:.2f}</td>'
        f'<td>{e["dist_mean"]:.2f} m</td><td>{e["ply_total"] if e["ply_total"] is not None else "—"}</td>'
        f'<td>{e["range"]:.2f} m</td></tr>' for e in entries
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Remy 采集包轨迹总览（{esc("+".join(styles))}）</title>
<style>{PAGE_CSS}</style></head>
<body><div class="wrap">
<h1>Remy 采集包轨迹总览</h1>
<div class="sub">共 {len(entries)} 个 ID · 风格 {esc("、".join(styles))} · 点击 ID 看默认风格，点右侧风格名看对应版本</div>
<table><thead><tr><th>文件夹 ID</th><th>风格</th><th>帧数</th><th>时长</th><th>fps</th>
<th>相机-锚点均值</th><th>点云点数</th><th>轨迹跨度</th></tr></thead>
<tbody>{rows}</tbody></table>
<div class="foot">由 vggt_human/99g_plot_capture_trajectory.py 生成。</div>
</div></body></html>"""


# ─────────────────────────── 主流程 ───────────────────────────

def process_one(folder: Path, dst_root: Path, styles, suffix=""):
    """画一个 ID 目录，每个风格落一份 .html + 一份独立 .svg，返回摘要行供 index 用。"""
    tj = folder / "transforms.json"
    data = load_transforms(tj)
    points, total, pbbox = load_ply_points(folder / "pcd.ply")
    ctx = build_ctx(folder.name, data, points, total, point_bbox=pbbox)

    out = []
    for style in styles:
        stem = folder.name if len(styles) == 1 else f"{folder.name}__{style}"
        svg, _ = RENDERERS[style](ctx)
        p = dst_root / f"{stem}.html"
        p.write_text(render_html(ctx, style, svg), encoding="utf-8")
        # 同时落一份独立 .svg（矢量原图，可直接拖进 PPT / 报告排版）。
        # vlimit 是「多张 SVG 拼在一页」的总览，落成单文件 .svg 会变成多个根元素（非法）→ 跳过。
        if style != "vlimit":
            (dst_root / f"{stem}.svg").write_text(svg, encoding="utf-8")
        out.append((style, p))
        print(f"  ✅ {style:<10} {p.name}" + ("" if style == "vlimit" else f" + {stem}.svg"))

    summary = {
        "id": folder.name, "files": {style: p.name for style, p in out},
        "n": ctx["n"], "duration": ctx["duration"], "fps": ctx["fps"],
        "dist_mean": ctx["dist_mean"], "ply_total": ctx["ply_total"],
        "range": max(ctx["range"]["x"][1] - ctx["range"]["x"][0], ctx["range"]["z"][1] - ctx["range"]["z"][0]),
    }
    return out, summary


def run(src_root: Path, dst_root: Path, styles, ids=None):
    if not src_root.is_dir():
        sys.exit(f"❌ 源目录不存在: {src_root}")
    subdirs = sorted([d for d in src_root.iterdir() if d.is_dir()], key=lambda p: p.name)
    if ids:
        want = set(ids)
        subdirs = [d for d in subdirs if d.name in want]
    if not subdirs:
        sys.exit(f"❌ {src_root} 下没有子目录")

    dst_root.mkdir(parents=True, exist_ok=True)
    print(f"📁 源: {src_root}")
    print(f"📁 输出: {dst_root}")
    print(f"🎨 风格: {', '.join(styles)}")
    print()

    entries, done, skip = [], 0, 0
    for sub in subdirs:
        if not (sub / "transforms.json").exists():
            print(f"⏭️  skip  {sub.name}  (无 transforms.json)")
            skip += 1
            continue
        print(f"🔍 {sub.name}")
        _, summary = process_one(sub, dst_root, styles)
        entries.append(summary)
        done += 1

    if entries:
        idx = dst_root / "index.html"
        idx.write_text(render_index(entries, styles), encoding="utf-8")
        print(f"\n📑 总览页: {idx}")
    print(f"\n🎉 Done.  ✅ {done} 个 ID  ⏭️ 跳过 {skip}  ❌ 0")
    print(f"📁 {dst_root}")


def main():
    # ===== 在这里直接改路径 =====
    # 批次根目录：其下每个子目录 = 一个采集 ID（含 transforms.json）
    # 默认值指向服务器路径；本机测试用 SRC_ROOT=D:/dataset/测试数据sample 覆盖即可。
    SRC_ROOT = Path(os.environ.get(
        "SRC_ROOT",
        "../../code/Reconstruction/dataset/43例人像数据-增加佳佳版",
    ))
    # 输出目录（HTML 很小，放哪儿都行）
    RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", "../../output/recon_human_results"))
    DST_ROOT = Path(os.environ.get("DST_ROOT", str(RESULTS_ROOT / SRC_ROOT.resolve().name / "html")))
    # 风格：quad（**默认**：四联对照图 2×2，左列原始采集轨迹、右列视角约束）/
    #       combo（俯视+等轴测双联，无约束）/ minimal / darkspace / fov / iso /
    #       vlimit（视角约束六种画法总览）/ lim_band / lim_edges / lim_angle /
    #       lim_rings / lim_hull / lim_shell / all
    STYLE = os.environ.get("STYLE", "quad")
    # 只画指定 ID（逗号分隔，留空 = 全部）
    IDS = [s for s in os.environ.get("IDS", "").split(",") if s.strip()]
    # ===========================

    styles = list(STYLES) if STYLE == "all" else [s for s in STYLE.split(",") if s in STYLES]
    if not styles:
        sys.exit(f"❌ STYLE 非法: {STYLE}（可选 {', '.join(STYLES)} 或 all）")
    run(SRC_ROOT, DST_ROOT, styles, ids=IDS)


if __name__ == "__main__":
    main()
