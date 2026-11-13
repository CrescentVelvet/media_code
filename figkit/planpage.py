# -*- coding: utf-8 -*-
"""计划页版式：把"实验方案 / 改进计划 / 评审结论"排成一页。

四种版式组件（规格见 SPEC.md §5.7）：
    checks    六格清单（3×2）——每条修正一句话，讲"要做哪几件事"
    contrast  两组口径对照——讲"两个数字为什么不能横比"
    matrix    多列表格——讲"假设 / 检查 / 判据"这类三列关系
    notes     小标题 + N 条带点说明——讲红线、产物、禁区

调用顺序自定，但页脚口径必填（F.footer）。标准五段式（阶段流水线 + 三卡 +
两列表 + 一句话）直接复用 onepager.page，传 meta['secs'] / meta['num_cols'] 改区标签。
"""
import figlib as F
from onepager import takeaway   # noqa: F401  （复用底部结论条）

SEC_Y = 146


def sec(d, y, s):
    """区标签：19 bold INK2。"""
    F.text(d, (F.M, y), s, 19, F.INK2, bold=True)


# ------------------------------------------------------------------ 六格清单
def checks(d, y, items, cols=3, h=160, gap_x=26, gap_y=18, color=F.INDIGO, w=580):
    """勾选清单：编号 + 标题 + ≤2 行说明。items=[(标题, 说明), ...]。"""
    inner = w - 52
    for i, (title, body) in enumerate(items):
        cx = F.M + (i % cols) * (w + gap_x)
        cy = y + (i // cols) * (h + gap_y)
        F.rrect(d, [cx, cy, cx + w, cy + h], 12, F.CARD_FILL, F.CARD_EDGE)
        F.rrect(d, [cx + 22, cy + 20, cx + 52, cy + 50], 8, color)
        F.text(d, (cx + 37, cy + 27), str(i + 1), 17, (255, 255, 255), bold=True, anchor='ma')
        F.text(d, (cx + 66, cy + 18), title, 22, F.INK, bold=True)
        for j, ln in enumerate(F.wrap(d, body, 19, inner, max_lines=2)):
            F.text(d, (cx + 26, cy + 62 + j * 28), ln, 19, F.INK2)
    rows = (len(items) + cols - 1) // cols
    return y + rows * h + (rows - 1) * gap_y


# ------------------------------------------------------------------ 口径对照
def contrast(d, y, items, h=176, gap=88, w=852):
    """两组数字对照：label / 大数字串 / 一行说明 / 右上胶囊标签。"""
    for i, (label, value, sub, tag, tag_color) in enumerate(items[:2]):
        x = F.M + i * (w + gap)
        F.rrect(d, [x, y, x + w, y + h], 14, F.CARD_FILL, tag_color)
        F.rrect(d, [x + 2, y + 2, x + 12, y + h - 2], 5, tag_color)
        F.text(d, (x + 30, y + 18), label, 19, F.INK2, bold=True)
        F.text(d, (x + 30, y + 44), value, 46, tag_color, bold=True)
        F.text(d, (x + 30, y + 122), sub, 19, F.INK2)
        tw = F.badge_w(tag)
        bx = x + w - tw - 18
        F.rrect(d, [bx, y + 20, bx + tw, y + 48], 14, tag_color)
        F.text(d, (bx + tw / 2, y + 26), tag, 16, (255, 255, 255), bold=True, anchor='ma')
    return y + h


# ---------------------------------------------------------------- 单行结论条
def strip(d, y, s, color=F.AMBER, h=46):
    """一行结论：左色条 + 21 bold（超宽降到 17）。"""
    F.rrect(d, [F.M, y, F.W - F.M, y + h], 10, F.CARD_FILL, F.CARD_EDGE)
    F.rrect(d, [F.M + 2, y + 2, F.M + 12, y + h - 2], 5, color)
    size = F.fit_size(d, s, 21, 17, F.W - 2 * F.M - 60, bold=True)
    F.text(d, (F.M + 30, y + 11), s, size, F.INK, bold=True)
    return y + h


# ------------------------------------------------------------------ 多列表格
def matrix(d, y, headers, widths, rows, row_h=100, colors=None):
    """三列及以上关系表（假设 / 检查 / 判据）；首列加粗当维度列。"""
    return F.table_n(d, F.M, y, headers, widths, rows, row_h=row_h, colors=colors)


# ------------------------------------------------------------------ 说明清单
def notes(d, y, title, items, color=F.INK2, gap=34, size=18, w=None):
    """小标题 + N 条带点说明（红线 / 产物 / 禁区）。"""
    F.text(d, (F.M, y), title, 19, color, bold=True)
    w = w or (F.W - 2 * F.M - 30)
    for i, s in enumerate(items):
        yy = y + 34 + i * gap
        F.rrect(d, [F.M + 4, yy + 8, F.M + 10, yy + 14], 2, color)
        F.text(d, (F.M + 22, yy), s, F.fit_size(d, s, size, 15, w), F.INK)
    return y + 34 + len(items) * gap
