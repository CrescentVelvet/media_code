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
    return F.fit_size(d, s, size, min_size, max_w, bold, light)


def _steps(d, y, steps, color):
    """横向 5 步 + 箭头。steps=[(主, 副[, 徽章])]×5；同级文字统一字号（取最紧的一格）。"""
    inner_w = STEP_W - 52
    tags = [it[2] if len(it) > 2 else None for it in steps]
    text_ws = [inner_w - (F.badge_w(t) + 10 if t else 0) for t in tags]
    main_sz = min(_fit(d, it[0], FS_STEP, 17, tw, bold=True)
                  for it, tw in zip(steps, text_ws))
    sub_sz = min(_fit(d, it[1], FS_STEP_SUB, 13, inner_w, light=True)
                 for it in steps if it[1])
    for i, item in enumerate(steps):
        main, sub, tag = item[0], item[1], tags[i]
        tw = F.badge_w(tag) if tag else 0
        x = F.M + i * (STEP_W + STEP_GAP)
        F.rrect(d, [x, y, x + STEP_W, y + STEP_H], 10, (255, 255, 255), color)
        F.rrect(d, [x + 2, y + 2, x + 12, y + STEP_H - 2], 4, color)
        F.text(d, (x + 26, y + 22), main, main_sz, F.INK, bold=True)
        if sub:
            F.text(d, (x + 26, y + 56), sub, sub_sz, F.INK2, light=True)
        if tag:
            bx, by = x + STEP_W - tw - 14, y + 8
            F.rrect(d, [bx, by, bx + tw, by + 24], 12, color)
            F.text(d, (bx + tw / 2, by + 2), tag, 16, (255, 255, 255), bold=True, anchor='ma')
        if i < len(steps) - 1:
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


def _numbers(d, y, rows, color, cols=None):
    """关键数字：4 行（维度 / 两列口径）。"""
    cols = cols or ('论文口径', '本仓实测（3090）')
    F.table(d, F.M, y, F.W - 2 * F.M, rows,
            [(cols[0], 796, color), (cols[1], 796, F.INDIGO)],
            dim_w=200, row_h=57)


def takeaway(d, y, s, color, h=TAKE_H, label='一句话', size=FS_TAKE):
    """底部结论条：单行或两行（传 list）。"""
    lines = [s] if isinstance(s, str) else list(s)
    F.rrect(d, [F.M, y, F.W - F.M, y + h], 12, F.CARD_FILL, F.CARD_EDGE)
    F.rrect(d, [F.M + 2, y + 2, F.M + 12, y + h - 2], 5, color)
    F.text(d, (F.M + 30, y + 12), label, 18, F.INK2, bold=True)
    for i, ln in enumerate(lines[:2]):
        sz = size - (3 if i else 0)
        while sz > 17 and d.textlength(ln, font=F.font(sz, bold=True)) > F.W - 2 * F.M - 60:
            sz -= 1
        F.text(d, (F.M + 30, y + 40 + i * 32), ln, sz, F.INK, bold=True)


def page(d, meta, steps, cards, numbers, takeaway_text):
    """meta: name / sub / source / color；可选 secs（三个区标签）、num_cols（两列标题）。"""
    color = meta['color']
    title_size = _fit(d, meta['name'], F.FS_TITLE, 32, F.W - 2 * F.M - 480, bold=True)
    F.header(d, meta['name'], meta['sub'], size=title_size)
    if meta.get('source'):
        F.text(d, (F.W - F.M, 44), meta['source'], 19, F.INK2, light=True, anchor='ra')

    secs = meta.get('secs') or ('①  它怎么走', '②  三个关键设计', '③  关键数字：论文口径 vs 本仓实测')
    _sec(d, F.M, SEC_Y, secs[0])
    _steps(d, STEP_Y, steps, color)

    _sec(d, F.M, CARD_Y - 30, secs[1])
    _cards(d, CARD_Y, cards, color)

    _sec(d, F.M, NUM_Y - 30, secs[2])
    _numbers(d, NUM_Y, numbers, color, meta.get('num_cols'))

    takeaway(d, TAKE_Y, takeaway_text, color)
