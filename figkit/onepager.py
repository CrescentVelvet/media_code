# -*- coding: utf-8 -*-
"""单方法页（论文一页纸）版式：给一个方法一整页，讲清"它怎么走 / 三个关键设计 /
关键数字 / 一句话结论"。规格见 SPEC.md §5.6。

调用顺序固定：
    page(d, meta, steps, cards, numbers, takeaway)
    footer(d, 口径)
其中 steps 5 步（横向）、cards 3 张（每张 ≤3 行）、numbers 4 行（论文 vs 本仓实测）。
"""
import figlib as F

SEC_Y = 146          # 区标签基线（① / ② / ③）
STEP_Y, STEP_H = 176, 100
STEP_W, STEP_GAP = 336, 28
CARD_Y, CARD_H = 330, 232
CARD_W, CARD_GAP = 580, 26
NUM_Y = 616
TAKE_Y, TAKE_H = 900, 92

FS_SEC = 19          # 区标签
FS_STEP = 22         # 步骤主标题
FS_STEP_SUB = 17
FS_CARD_T = 23       # 卡标题
FS_BULLET = 19       # 卡内要点
FS_TAKE = 21


def _sec(d, x, y, s):
    F.text(d, (x, y), s, FS_SEC, F.INK2, bold=True)


def _fit(d, s, size, min_size, max_w, bold=False, light=False):
    """按实际字宽降字号，避免长英文串冲出块边框（PIL 无自动换行）。"""
    while size > min_size and d.textlength(s, font=F.font(size, bold, light)) > max_w:
        size -= 1
    return size


def _steps(d, y, steps, color):
    """横向 5 步 + 箭头。steps=[(主, 副)]×5"""
    inner_w = STEP_W - 52
    for i, (main, sub) in enumerate(steps):
        x = F.M + i * (STEP_W + STEP_GAP)
        F.rrect(d, [x, y, x + STEP_W, y + STEP_H], 10, (255, 255, 255), color)
        F.rrect(d, [x + 2, y + 2, x + 12, y + STEP_H - 2], 4, color)
        F.text(d, (x + 26, y + 22), main,
               _fit(d, main, FS_STEP, 17, inner_w, bold=True), F.INK, bold=True)
        if sub:
            F.text(d, (x + 26, y + 56), sub,
                   _fit(d, sub, FS_STEP_SUB, 13, inner_w, light=True), F.INK2, light=True)
        if i < 4:
            F.text(d, (x + STEP_W + STEP_GAP / 2, y + STEP_H / 2), '→',
                   20, F.RAIL, anchor='mm')


def _cards(d, y, cards, color):
    """三张卡：标题 + ≤3 行要点。cards=[(标题, [行, ...)]×3"""
    inner_w = CARD_W - 66
    for i, (title, lines) in enumerate(cards):
        x = F.M + i * (CARD_W + CARD_GAP)
        F.rrect(d, [x, y, x + CARD_W, y + CARD_H], 14, F.CARD_FILL, F.CARD_EDGE)
        F.rrect(d, [x + 2, y + 2, x + 12, y + 34], 5, color)
        F.text(d, (x + 28, y + 14), title, FS_CARD_T, F.INK, bold=True)
        for j, ln in enumerate(lines[:3]):
            by = y + 66 + j * 40
            F.rrect(d, [x + 28, by + 9, x + 34, by + 15], 2, color)
            F.text(d, (x + 46, by), ln,
                   _fit(d, ln, FS_BULLET, 16, inner_w), F.INK)


def _numbers(d, y, rows, color):
    """关键数字：4 行（维度 / 论文口径 / 本仓实测）。"""
    F.table(d, F.M, y, F.W - 2 * F.M, rows,
            [('论文口径', 796, color), ('本仓实测（3090）', 796, F.INDIGO)],
            dim_w=200, row_h=57)


def _takeaway(d, y, s, color):
    F.rrect(d, [F.M, y, F.W - F.M, y + TAKE_H], 12, F.CARD_FILL, F.CARD_EDGE)
    F.rrect(d, [F.M + 2, y + 2, F.M + 12, y + TAKE_H - 2], 5, color)
    F.text(d, (F.M + 30, y + 12), '一句话', 18, F.INK2, bold=True)
    size = FS_TAKE
    while size > 17 and d.textlength(s, font=F.font(size, bold=True)) > F.W - 2 * F.M - 60:
        size -= 1
    F.text(d, (F.M + 30, y + 42), s, size, F.INK, bold=True)


def page(d, meta, steps, cards, numbers, takeaway):
    """meta: name / sub / source / color。"""
    color = meta['color']
    title_size = _fit(d, meta['name'], F.FS_TITLE, 32, F.W - 2 * F.M - 480, bold=True)
    F.header(d, meta['name'], meta['sub'], size=title_size)
    if meta.get('source'):
        F.text(d, (F.W - F.M, 44), meta['source'], 19, F.INK2, light=True, anchor='ra')

    _sec(d, F.M, SEC_Y, '①  它怎么走')
    _steps(d, STEP_Y, steps, color)

    _sec(d, F.M, CARD_Y - 30, '②  三个关键设计')
    _cards(d, CARD_Y, cards, color)

    _sec(d, F.M, NUM_Y - 30, '③  关键数字：论文口径 vs 本仓实测')
    _numbers(d, NUM_Y, numbers, color)

    _takeaway(d, TAKE_Y, takeaway, color)
