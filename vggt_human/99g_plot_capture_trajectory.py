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

用法:
    python vggt_human/99g_plot_capture_trajectory.py
    STYLE=all       ...   # 一次出全部风格（<ID>__<style>.html），便于挑图
    STYLE=minimal   ...   # 只出一种（<ID>.html）
    SRC_ROOT=... DST_ROOT=... bash ...

    # 可选：只画其中几个 ID
    IDS=13a8ecadfeb448e890db319ac828befe,10e3ec6291c04bedaad735309e4bc43b STYLE=all ...

路径与默认风格写在下方 main() 里。
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
from pathlib import Path

# 点云最多画多少个点（超出按等间隔抽稀，保证渲染不卡）
PLY_MAX_POINTS = 12000

STYLES = ("combo", "minimal", "darkspace", "fov", "iso")

STYLE_LABELS = {
    "combo": "俯视 + 等轴测双联图（上下排列，俯视轨迹按帧序时间渐变）",
    "minimal": "浅色极简 · 轨迹 + 视线 + 均值半径环",
    "darkspace": "深色网格 · 按时间渐变的轨迹",
    "fov": "视锥扇形 + 点云底图（浅色）",
    "iso": "等轴测立体 · 含高度与垂直投影线",
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


def axis_hints(view, color, fs=12):
    """在绘图区外侧标出世界 X / Z 的正方向（俯视图约定：X 右、Z 下）。

    位置贴着数据框外侧而不是框内：数据框本身只有数据那么宽，框内左下角常常正好被
    轨迹压住；框外紧邻位置是白边，放轴指示既清楚又不遮数据。
    """
    rx0, ry0, rw, rh = view.rect()
    out = []
    # 左侧：Z 向下
    ax, ay = rx0 - 30, ry0 + rh - 200
    out.append(f'<line x1="{ax}" y1="{ay}" x2="{ax}" y2="{ay + 34}" stroke="{color}" stroke-width="1.4"/>')
    out.append(f'<path d="M{ax - 4},{ay + 28} L{ax},{ay + 34} L{ax + 4},{ay + 28}" fill="none" stroke="{color}" '
               f'stroke-width="1.4" stroke-linecap="round"/>')
    out.append(f'<text x="{ax + 9}" y="{ay + 34}" font-size="{fs}" fill="{color}">+Z</text>')
    # 右侧：X 向右
    bx, by = rx0 + rw + 12, ry0 + rh - 26
    out.append(f'<line x1="{bx}" y1="{by}" x2="{bx + 34}" y2="{by}" stroke="{color}" stroke-width="1.4"/>')
    out.append(f'<path d="M{bx + 28},{by - 4} L{bx + 34},{by} L{bx + 28},{by + 4}" fill="none" stroke="{color}" '
               f'stroke-width="1.4" stroke-linecap="round"/>')
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


def iso_proj(x, y, z):
    """等轴测投影：世界 (X, Y, Z) → 平面 (u, v)，v 向下。"""
    return (x - z) * ISO_COS, (x + z) * ISO_SIN - y


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


def draw_top_geometry(ctx, view, clip, pal=PAL_LIGHT, ramp=None, *,
                      arrows_every=20, dots_every=10, arrow_len=0.26,
                      arrow_color=None, ring=True, cloud=False, track_width=2.0,
                      clip_rect=None):
    """把俯视图的几何画进给定 view。

    ramp=None → 轨迹用 pal['track'] 单色；否则传 [(pos,(r,g,b)),...] 按帧序做时间渐变。
    clip_rect 给定时按它裁剪（用于「绘图区预留框比数据框宽」的场合：点云/半径环裁到
    预留框，把两侧的空档填满）；不给则按数据框 view.rect() 裁。
    锚点与起终点标记不裁剪（避免贴边被切掉）。
    """
    out = []
    x0, y0, bw, bh = clip_rect if clip_rect is not None else view.rect()
    out.append(f'<clipPath id="{clip}"><rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" '
               f'height="{bh:.1f}" rx="10"/></clipPath>')
    out.append(f'<g clip-path="url(#{clip})">')

    if cloud and ctx["points"]:
        out.append("".join(
            f'<circle cx="{view.p(p[0], p[2])[0]:.1f}" cy="{view.p(p[0], p[2])[1]:.1f}" r="1" '
            f'fill="{pal["cloud"]}" opacity="0.5"/>' for p in ctx["points"]))

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

    if ctx["anchor"]:
        ax, ay = view.p(ctx["anchor"][0], ctx["anchor"][2])
        out.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="8" fill="none" stroke="{pal["anc"]}" stroke-width="2"/>')
        out.append(f'<line x1="{ax - 12:.1f}" y1="{ay:.1f}" x2="{ax + 12:.1f}" y2="{ay:.1f}" stroke="{pal["anc"]}" stroke-width="1"/>')
        out.append(f'<line x1="{ax:.1f}" y1="{ay - 12:.1f}" x2="{ax:.1f}" y2="{ay + 12:.1f}" stroke="{pal["anc"]}" stroke-width="1"/>')
        out.append(f'<text x="{ax + 16:.1f}" y="{ay - 8:.1f}" font-size="12" fill="{pal["anc"]}">anchor_point</text>')

    sx, sy = view.p(ctx["pos"][0][0], ctx["pos"][0][2])
    ex, ey = view.p(ctx["pos"][-1][0], ctx["pos"][-1][2])
    out.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6" fill="{pal["start"]}"/>')
    out.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="6" fill="{pal["end"]}"/>')
    out.append(f'<text x="{sx + 12:.1f}" y="{sy + 5:.1f}" font-size="12" fill="{pal["start"]}">frame 0</text>')
    out.append(f'<text x="{ex + 12:.1f}" y="{ey + 5:.1f}" font-size="12" fill="{pal["end"]}">frame {ctx["n"] - 1}</text>')
    return "\n".join(out)


def draw_iso_geometry(ctx, view, clip, pal=PAL_LIGHT, *, cloud=True, grid=True, drop_every=5,
                      dots_every=10, floor=None, clip_rect=None):
    """把等轴测视图的几何（地面网格 / 点云 / 相机与垂线 / 轨迹 / 轴三叉）画进给定 view。"""
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

    if cloud and ctx["points"]:
        out.append("".join(
            f'<circle cx="{view.p(*iso_proj(*p))[0]:.1f}" cy="{view.p(*iso_proj(*p))[1]:.1f}" r="1.1" '
            f'fill="{pal["cloud"]}" opacity="0.45"/>' for p in ctx["points"]))

    for i in range(0, ctx["n"], drop_every):   # 相机到地面的垂线，表达高度
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

    if ctx["anchor"]:
        ax, ay = view.p(*iso_proj(*ctx["anchor"]))
        out.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="7" fill="none" stroke="{pal["anc"]}" stroke-width="2"/>')
        out.append(f'<text x="{ax + 14:.1f}" y="{ay - 10:.1f}" font-size="12" fill="{pal["anc"]}">anchor_point</text>')

    sx, sy = view.p(*iso_proj(*ctx["pos"][0]))
    ex, ey = view.p(*iso_proj(*ctx["pos"][-1]))
    out.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6" fill="{pal["start"]}"/>')
    out.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="6" fill="{pal["end"]}"/>')
    return "\n".join(out), floor


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
    out.append(draw_top_geometry(ctx, view, f"clip_{ctx['id'][:8]}_min"))

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

    if ctx["points"]:
        pd = []
        for p in ctx["points"]:
            x, y = view.p(p[0], p[2])
            pd.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1" fill="#B4B2A9" opacity="0.5"/>')
        out.append("".join(pd))

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
    geom, floor = draw_iso_geometry(ctx, view, f"clip_{ctx['id'][:8]}_iso")
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

    view_top = View(*ctx["range"]["x"], *ctx["range"]["z"], BOX_L, TOP_Y, BOX_R, TOP_Y + PANEL_H)

    # 等轴测：视野由 iso_extent() 决定（只按相机活动范围，点云作背景）
    u0, u1, v0, v1, _ = iso_extent(ctx)
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
    out.append(draw_top_geometry(ctx, view_top, f"clip_{ctx['id'][:8]}_ct", ramp=RAMP_LIGHT,
                                 cloud=True, clip_rect=panel_top))
    out.append(axis_hints(view_top, PAL_LIGHT["axis"]))
    out.append(scale_bar(BOX_L + 16, TOP_Y + PANEL_H - 44, 0.5, view_top, color=MUTED))

    # 俯视图图例：铺在绘图区下方，5 项等距排满整幅宽度
    ly = TOP_Y + PANEL_H + 40
    out.append(legend_grid([
        ("line", f"相机轨迹（{ctx['n']} 帧）", PAL_LIGHT["track"]),
        ("arrow", "视线方向", PAL_LIGHT["arrow"]),
        ("ring", f"均值半径 {ctx['dist_mean']:.2f} m", PAL_LIGHT["ring"]),
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
    geom, floor = draw_iso_geometry(ctx, view_iso, f"clip_{ctx['id'][:8]}_ci", clip_rect=panel_iso)
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


RENDERERS = {
    "combo": style_combo,
    "minimal": style_minimal,
    "darkspace": style_darkspace,
    "fov": style_fov,
    "iso": style_iso,
}


# ─────────────────────────── HTML 组装 ───────────────────────────

def stat_cards(ctx) -> str:
    d = ctx["data"]
    r = ctx["range"]
    intr = ctx["intr"]
    cards = [
        ("帧数 / 时长", f"{ctx['n']} 帧 · {ctx['duration']:.1f} s · {ctx['fps']:.2f} fps"),
        ("相机-锚点水平距离", f"{ctx['dist_min']:.2f} – {ctx['dist_max']:.2f} m（均值 {ctx['dist_mean']:.2f}）"),
        ("视线对准偏差", f"均值 {ctx['ang_mean']:.1f}° · 最大 {ctx['ang_max']:.1f}°"),
        ("轨迹范围 (X×Z)", f"{r['x'][1] - r['x'][0]:.2f} × {r['z'][1] - r['z'][0]:.2f} m"),
        ("相机高度范围 (Y)", f"{r['y'][0]:.2f} – {r['y'][1]:.2f} m（跨度 {r['y'][1] - r['y'][0]:.2f}）"),
        ("内参（全帧一致）", f"{intr['w']}×{intr['h']} · fl {ctx['fl']:.1f}px · 等效 {ctx['f_equiv']:.1f}mm · 对角 {ctx['fov_d']:.1f}°"),
        ("畸变", f"k1 {intr['k1']:.4f} · k2 {intr['k2']:.4f} · k3 {intr['k3']:.4f} · p1 {intr['p1']:.2e} · p2 {intr['p2']:.2e}"),
        ("点云 pcd.ply", (f"{ctx['ply_total']} 点（图上抽稀 {len(ctx['points'])}）· "
                          f"bbox "
                          f"{ctx['bbox']['max'][0] - ctx['bbox']['min'][0]:.2f} × "
                          f"{ctx['bbox']['max'][1] - ctx['bbox']['min'][1]:.2f} × "
                          f"{ctx['bbox']['max'][2] - ctx['bbox']['min'][2]:.2f} m")
         if ctx["bbox"] else "未找到或无有效点"),
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
.wrap.wide{max-width:1240px}
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
    # 双联图更宽，容器跟着放宽，否则整张图被缩到 980px 宽、字变小
    wrap_cls = "wrap wide" if style == "combo" else "wrap"
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
        # 同时落一份独立 .svg（矢量原图，可直接拖进 PPT / 报告排版）
        (dst_root / f"{stem}.svg").write_text(svg, encoding="utf-8")
        out.append((style, p))
        print(f"  ✅ {style:<10} {p.name} + {stem}.svg")

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
    SRC_ROOT = Path(os.environ.get("SRC_ROOT", "/mnt/d/dataset/测试数据sample"))
    # 输出目录（HTML 很小，放哪儿都行）
    DST_ROOT = Path(os.environ.get("DST_ROOT", "../../output/remy_traj_html"))
    # 风格：combo（等轴测+俯视双联）/ minimal / darkspace / fov / iso / all
    STYLE = os.environ.get("STYLE", "combo")
    # 只画指定 ID（逗号分隔，留空 = 全部）
    IDS = [s for s in os.environ.get("IDS", "").split(",") if s.strip()]
    # ===========================

    styles = list(STYLES) if STYLE == "all" else [s for s in STYLE.split(",") if s in STYLES]
    if not styles:
        sys.exit(f"❌ STYLE 非法: {STYLE}（可选 {', '.join(STYLES)} 或 all）")
    run(SRC_ROOT, DST_ROOT, styles, ids=IDS)


if __name__ == "__main__":
    main()
