# -*- coding: utf-8 -*-
"""PPT 配图排版库：常量即规格，改数值前先读同目录 SPEC.md。

画布满版 16:9（1920×1080），导出 PNG。色板语义：靛蓝=数据/主 · 青=核心 ·
琥珀=警示/补充 · 红=负结果。方法身份色另列（见 SPEC.md §5）。

根目录：默认当前工作目录，可用环境变量 FIG_ROOT 或 set_root() 指定；
成品写到 <root>/figures/，读数读 <root>/metrics.json。

文字一律微软雅黑（msyh.ttc / msyhbd.ttc / msyhl.ttc）。
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- 画布（§2）
W, H = 1920, 1080
M = 64                      # 左右边距
BG = (255, 255, 255)
BG_SOFT = (246, 248, 251)   # 表格斑马 / 淡底

# ---------------------------------------------------------------- 字号阶梯（§4）
FS_TITLE = 42               # 页标题（bold）
FS_SUB = 22                 # 副标题
FS_CARD = 32                # 卡片名（bold）
FS_CARD_TAG = 19            # 卡片脚注（light）
FS_STEP = 23                # 步骤主标题（bold）
FS_STEP_SUB = 17            # 步骤副说明（light）
FS_BADGE = 16               # 徽章（bold / 白字）
FS_TH = 20                  # 表头
FS_CELL = 19                # 表格正文
FS_NOTE = 17                # 页脚口径（light）

# ---------------------------------------------------------------- 栅格（§3）
RULE_Y = 132                # 页眉分隔线
CARD_Y, CARD_H = 142, 80    # 卡片头
CARD_W = 852                # 双栏卡宽（左右各一，中间 88px 沟）
CARD_INSET = 28             # 卡内步骤块左右缩进
STEP_TOP = 92               # 卡头顶 → 第一步（相对 CARD_Y）
STEP_H, STEP_GAP = 66, 6
STEP_N = 6                  # 标准行数（两栏必须等行）
TABLE_Y = 706               # 表头基线
ROW_H = 57                  # 表格行高
DIM_W = 200                 # 维度列宽

# ---------------------------------------------------------------- 色板（§5）
INK = (27, 42, 65)          # 主文字
INK2 = (90, 104, 122)       # 次文字
RULE = (214, 221, 230)      # 分隔线
INDIGO = (59, 91, 165)      # 数据 / 主色
BLUE = (46, 92, 138)
CYAN = (23, 140, 158)       # 核心
AMBER = (214, 145, 45)      # 警示 / 补充
RED = (192, 71, 60)         # 负结果
GREEN = (60, 140, 100)
GRAY = (150, 158, 170)      # 弱化 / 已去掉

C_DA3 = (15, 110, 110)      # 方法身份：DA3-GS
C_V2 = (139, 111, 168)      # 方法身份：InfiniSplat v2
C_V3 = (47, 107, 181)       # 方法身份：InfiniSplat v3
C_QS = (91, 127, 181)       # 方法身份：QuerySplat
C_REF = (181, 118, 58)      # 方法身份：3DGS 精化系（参照）

CARD_FILL = (250, 251, 253)
CARD_EDGE = (226, 231, 238)
DIM_EDGE = (203, 209, 217)  # 灰化块的边
DIM_FILL = (247, 248, 250)  # 灰化块的底
RAIL = (205, 212, 222)      # 流转轨道

FONT_R = 'C:/Windows/Fonts/msyh.ttc'
FONT_B = 'C:/Windows/Fonts/msyhbd.ttc'
FONT_L = 'C:/Windows/Fonts/msyhl.ttc'
_cache = {}

# ---------------------------------------------------------------- 根目录
ROOT = Path(os.environ.get('FIG_ROOT', '.')).resolve()
OUT = ROOT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)


def set_root(p):
    """切换出图根目录（成品落到 <p>/figures/）。"""
    global ROOT, OUT
    ROOT = Path(p).resolve()
    OUT = ROOT / 'figures'
    OUT.mkdir(parents=True, exist_ok=True)
    return ROOT


# ---------------------------------------------------------------- 字体 / 画布
def font(size, bold=False, light=False):
    key = (size, bold, light)
    if key not in _cache:
        path = FONT_B if bold else (FONT_L if light else FONT_R)
        _cache[key] = ImageFont.truetype(path, size)
    return _cache[key]


def canvas(w=W, h=H, bg=BG):
    im = Image.new('RGB', (w, h), bg)
    return im, ImageDraw.Draw(im)


def text(d, xy, s, size=24, color=INK, bold=False, light=False, anchor='la'):
    d.text(xy, s, font=font(size, bold, light), fill=color, anchor=anchor)


def fit_size(d, s, size, min_size, max_w, bold=False, light=False):
    """按实际字宽降字号，直到放得下（PIL 无自动换行，先量后画）。"""
    while size > min_size and d.textlength(s, font=font(size, bold, light)) > max_w:
        size -= 1
    return size


def wrap(d, s, size, max_w, max_lines=2, bold=False):
    """按实际字宽折行：CJK 逐字可断，拉丁/数字串整词换行（不拦腰截断），
    超出行数时末行加省略号。"""
    lines, cur = [], ''
    for ch in s:
        if ch == '\n':
            lines.append(cur)
            cur = ''
            continue
        if cur and d.textlength(cur + ch, font=font(size, bold)) > max_w:
            cut = cur.rfind(' ')
            k = len(cur)
            while k > 0 and ord(cur[k - 1]) < 128 and cur[k - 1] != ' ':
                k -= 1
            if k > 0 and k < len(cur):      # 行尾是拉丁词 → 整词挪到下一行
                cut = k
            if cut <= 0:
                cut = len(cur)
            lines.append(cur[:cut].strip())
            cur = (cur[cut:] + ch).lstrip()
        else:
            cur += ch
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and d.textlength(lines[-1] + '…', font=font(size, bold)) > max_w:
            lines[-1] = lines[-1][:-1]
        lines[-1] += '…'
    return lines


# ---------------------------------------------------------------- 页眉 / 页脚（§6.1）
def header(d, title, sub=None, size=None):
    """左上竖条 + 标题 + 副标题 + 通栏分隔线。size 可显式指定（长标题需降字号时）。"""
    d.rectangle([M, 40, M + 8, 92], fill=INDIGO)
    text(d, (M + 32, 32), title, size or FS_TITLE, INK, bold=True)
    if sub:
        text(d, (M + 34, 90), sub, FS_SUB, INK2)
    d.line([M, RULE_Y, W - M, RULE_Y], fill=RULE, width=2)
    return RULE_Y


def footer(d, s):
    """页脚口径说明：必填，写清数据来源与不可比之处。"""
    text(d, (M, H - 36), s, FS_NOTE, INK2, light=True)


# ---------------------------------------------------------------- 卡片（§6.2）
def rrect(d, box, r=12, fill=None, outline=None, w=2):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=w)


def card_head(d, x, y, color, name, tag):
    """色条 + 方法名 + 一行定位。y 为卡片顶，高 CARD_H。"""
    rrect(d, [x, y, x + CARD_W, y + CARD_H], 14, CARD_FILL, CARD_EDGE)
    rrect(d, [x + 2, y + 2, x + 12, y + CARD_H - 2], 5, color)
    text(d, (x + 34, y + 6), name, FS_CARD, INK, bold=True)
    text(d, (x + 34, y + 52), tag, FS_CARD_TAG, INK2, light=True)


def _badge_w(tag):
    # 无字体度量时的保守估算：中文 15 / 西文 9，两侧 padding 26
    return 26 + int(sum(15 if ord(c) > 127 else 9 for c in tag))


def badge_w(tag):
    """胶囊徽章宽度估算（中文 15 / 西文 9 + padding 26）。标签建议 ≤6 汉字。"""
    return _badge_w(tag)


def step(d, x, y, w, h, color, main, sub=None, tag=None, dim=False):
    """一个流水线步骤块。dim=True 为灰化（表示这一步被去掉/不属于本方法）。"""
    edge = DIM_EDGE if dim else color
    fill = DIM_FILL if dim else (255, 255, 255)
    rrect(d, [x, y, x + w, y + h], 10, fill, edge)
    text(d, (x + 24, y + 4), main, FS_STEP, INK2 if dim else INK, bold=True)
    if sub:
        text(d, (x + 24, y + 34), sub, FS_STEP_SUB, INK2, light=True)
    if tag:
        tw = _badge_w(tag)
        bx = x + w - tw - 12
        by = y + (h - 28) // 2
        rrect(d, [bx, by, bx + tw, by + 28], 14, DIM_EDGE if dim else color)
        text(d, (bx + tw / 2, by + 6), tag, FS_BADGE, (255, 255, 255),
             bold=True, anchor='ma')


def rail(d, x, y0, y1):
    """块间短竖线（流程轨道），x 取块左边缘 +12。"""
    d.line([x, y0, x, y1], fill=RAIL, width=2)


# ---------------------------------------------------------------- 差异表（§6.3）
def table(d, x, y, w, rows, cols, dim_w=DIM_W, row_h=ROW_H):
    """rows: [(维度, 左格, 右格)]；cols: [(标题, 列宽, 色)]。

    单元格支持 '\\n' 换行（最多两行）；超宽自动降字号（≤3 级，不低于 16）。
    """
    xs = [x + dim_w]
    for c in cols[:-1]:
        xs.append(xs[-1] + c[1])

    text(d, (x, y), '维度', FS_TH, INK2, bold=True)
    for i, (label, _cw, color) in enumerate(cols):
        text(d, (xs[i] + 2, y - 1), label, FS_TH + 1, color, bold=True)
    d.line([x, y + 30, x + w, y + 30], fill=RULE, width=2)

    ry = y + 38
    for k, (dim, a, b) in enumerate(rows):
        if k % 2 == 1:
            d.rectangle([x - 12, ry - 4, x + w + 6, ry + row_h - 8], fill=BG_SOFT)
        text(d, (x, ry + 6), dim, FS_CELL + 2, INK, bold=True)
        for i, cell in enumerate((a, b)):
            colw = cols[i][1]
            size = FS_CELL
            lines = cell.split('\n')
            while size > 16 and max(d.textlength(s, font=font(size)) for s in lines) > colw - 24:
                size -= 1
            for j, ln in enumerate(lines):
                text(d, (xs[i] + 2, ry + 4 + j * 26), ln, size, INK)
        ry += row_h
    return ry


def table_n(d, x, y, headers, widths, rows, row_h=ROW_H, colors=None,
            first_bold=True, size=FS_CELL, min_size=16):
    """通用多列表格（≥2 列）：headers / widths / rows 一一对应。

    列标题用 colors[i]（未给则 INK2）；首列默认加粗当"维度列"。
    单元格自动折行（≤2 行，超出加省略号），仍放不下时降字号到 min_size。
    """
    xs = [x]
    for w in widths[:-1]:
        xs.append(xs[-1] + w)
    total = sum(widths)
    for i, h in enumerate(headers):
        text(d, (xs[i] + 2, y - 1), h, FS_TH + 1,
             colors[i] if colors else INK2, bold=True)
    d.line([x, y + 30, x + total, y + 30], fill=RULE, width=2)

    ry = y + 38
    for k, cells in enumerate(rows):
        if k % 2 == 1:
            d.rectangle([x - 12, ry - 4, x + total + 6, ry + row_h - 8], fill=BG_SOFT)
        for i, cell in enumerate(cells):
            bold = first_bold and i == 0
            avail = widths[i] - 26
            sz = size + (2 if bold else 0)
            while sz > min_size and max(
                    d.textlength(s, font=font(sz, bold)) for s in str(cell).split('\n')) > avail:
                sz -= 1
            lines = []
            for part in str(cell).split('\n'):
                lines += wrap(d, part, sz, avail, max_lines=2 - min(len(lines), 1), bold=bold)
            top = ry + max(4, (row_h - 8 - 28 * len(lines[:2])) // 2)   # 单行也垂直居中
            for j, ln in enumerate(lines[:2]):
                text(d, (xs[i] + 2, top + j * 28), ln, sz, INK, bold=bold)
        ry += row_h
    return ry


# ---------------------------------------------------------------- 图像贴入
def fit_square(path, size):
    """等比缩放并居中裁成正方形。"""
    im = Image.open(path).convert('RGB')
    sw, sh = im.size
    side = min(sw, sh)
    im = im.crop(((sw - side) // 2, (sh - side) // 2,
                  (sw + side) // 2, (sh + side) // 2))
    return im.resize((size, size), Image.LANCZOS)


def fit_cover(path, w, h):
    """等比缩放填满 w×h，居中裁切（不拉伸）。"""
    im = Image.open(path).convert('RGB')
    sw, sh = im.size
    scale = max(w / sw, h / sh)
    nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
    im = im.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return im.crop((left, top, left + w, top + h))


def paste_card(im, tile, xy, border=None, label=None, lab_size=22,
               lab_color=INK, lab_bold=False):
    """贴图并可选描边、图下标签。返回占用高度。"""
    x, y = xy
    im.paste(tile, (x, y))
    if border:
        ImageDraw.Draw(im).rectangle([x, y, x + tile.width, y + tile.height],
                                     outline=border, width=2)
    if label:
        ImageDraw.Draw(im).text((x + tile.width // 2, y + tile.height + 14), label,
                                font=font(lab_size, lab_bold), fill=lab_color,
                                anchor='ma')
        return tile.height + 14 + lab_size + 8
    return tile.height


def diff_heat(gt_path, rd_path, size, gain=1.0, colormap=True):
    """GT 与渲染的绝对差，放大 gain 倍后上色（白 → 琥珀 → 红）。"""
    a = np.asarray(Image.open(gt_path).convert('RGB').resize((size, size), Image.LANCZOS),
                   np.float32)
    b = np.asarray(Image.open(rd_path).convert('RGB').resize((size, size), Image.LANCZOS),
                   np.float32)
    d = np.abs(a - b).mean(axis=2) * gain
    d = np.clip(d / 64.0, 0, 1)          # 64/255 ≈ 25% 差异封顶
    if not colormap:
        g = (d * 255).astype(np.uint8)
        return Image.fromarray(np.stack([g] * 3, -1))
    stops = np.array([[255, 255, 255], [232, 176, 88], [186, 56, 48]], np.float32)
    idx = d * (len(stops) - 1)
    lo = np.clip(np.floor(idx).astype(int), 0, len(stops) - 2)
    frac = (idx - lo)[..., None]
    rgb = stops[lo] * (1 - frac) + stops[lo + 1] * frac
    return Image.fromarray(rgb.astype(np.uint8))


# ---------------------------------------------------------------- 读数 / 输出
def load_metrics():
    return json.loads((ROOT / 'metrics.json').read_text(encoding='utf-8'))


def ids_in(sub):
    """某个 pairs 子目录里的视角 id（已排序）。"""
    d = ROOT / 'pairs' / sub
    return [p.stem[3:] for p in sorted(d.glob('rd_*.png'))]


def pick(ids, k):
    """均匀取 k 个。"""
    if k >= len(ids):
        return ids
    step_ = (len(ids) - 1) / (k - 1) if k > 1 else 0
    return [ids[round(i * step_)] for i in range(k)]


def save(im, name):
    p = OUT / name
    im.save(p, optimize=True)
    print(f'  ✓ {p}  {im.size[0]}x{im.size[1]}  {p.stat().st_size / 1e6:.2f} MB')
    return p
