# -*- coding: utf-8 -*-
"""极简折线 / 柱状绘制（PIL 手绘，避开 matplotlib 的中文字体坑）。

坐标系：box=(x0,y0,x1,y1) 为绘图区，y 轴向上（数值大在上）。
字号走 figlib 阶梯：刻度 23 / 轴标签 24 / 数据点标注 24。
"""
from figlib import INK2, font

FRAME = (160, 172, 186)
GRID = (238, 242, 247)
FS_TICK = 23
FS_AXIS = 24
FS_DATA = 24


class Axes:
    def __init__(self, d, box, xlim, ylim, xticks, yticks, ylabel='', xlabel='',
                 yfmt='{:.0f}', ygrid=True):
        self.d, self.box = d, box
        self.x0, self.y0, self.x1, self.y1 = box
        self.xlim, self.ylim = xlim, ylim
        self.xlabel, self.ylabel = xlabel, ylabel
        self.xticks, self.yticks = xticks, yticks
        self.yfmt = yfmt
        self._frame()
        if ygrid:
            for t in self.yticks:
                y = self.py(t)
                d.line([self.x0, y, self.x1, y], fill=GRID, width=1)
        self._ticks()
        if ylabel:
            d.text(((self.x0 + self.x1) / 2, self.y1 + 66), ylabel,
                   font=font(FS_AXIS), fill=INK2, anchor='ma')

    def px(self, v):
        a, b = self.xlim
        return self.x0 + (v - a) / (b - a) * (self.x1 - self.x0)

    def py(self, v):
        a, b = self.ylim
        return self.y1 - (v - a) / (b - a) * (self.y1 - self.y0)

    def _frame(self):
        d = self.d
        d.line([self.x0, self.y0, self.x0, self.y1], fill=FRAME, width=2)
        d.line([self.x0, self.y1, self.x1, self.y1], fill=FRAME, width=2)

    def _ticks(self):
        d = self.d
        for t in self.yticks:
            y = self.py(t)
            d.line([self.x0 - 8, y, self.x0, y], fill=FRAME, width=2)
            d.text((self.x0 - 16, y), self.yfmt.format(t), font=font(FS_TICK),
                   fill=INK2, anchor='rm')
        for t, lab in self.xticks:
            x = self.px(t)
            d.line([x, self.y1, x, self.y1 + 8], fill=FRAME, width=2)
            d.text((x, self.y1 + 16), lab, font=font(FS_AXIS), fill=(60, 72, 88),
                   anchor='ma')

    def line(self, xs, ys, color, width=5, dot=11, dash=None,
             label_dy=-34, label_color=None, labels=None):
        d = self.d
        pts = [(self.px(x), self.py(y)) for x, y in zip(xs, ys)]
        if dash:
            for i in range(len(pts) - 1):
                if i % 2 == 0:
                    d.line([pts[i], pts[i + 1]], fill=color, width=width)
        else:
            d.line(pts, fill=color, width=width, joint='curve')
        r = dot // 2
        for i, (x, y) in enumerate(pts):
            d.ellipse([x - r, y - r, x + r, y + r], fill=color,
                      outline=(255, 255, 255), width=2)
            if labels:
                d.text((x, y + label_dy), labels[i], font=font(FS_DATA, True),
                       fill=label_color or color, anchor='ma')
        return pts

    def vband(self, xa, xb, color=(244, 247, 251)):
        self.d.rectangle([self.px(xa), self.y0, self.px(xb), self.y1], fill=color)

    def legend(self, x, y, items, gap=54, size=FS_AXIS):
        """items = [(name, color, dashed)]"""
        d = self.d
        for i, (name, color, dashed) in enumerate(items):
            yy = y + i * gap
            d.line([x, yy, x + 52, yy], fill=color, width=6)
            if dashed:
                d.rectangle([x + 18, yy - 4, x + 34, yy + 4], fill=(255, 255, 255))
            d.text((x + 68, yy), name, font=font(size), fill=(50, 62, 78), anchor='lm')

    def note(self, x, y, s, size=22, color=INK2, anchor='la'):
        self.d.text((x, y), s, font=font(size), fill=color, anchor=anchor)


def bar(axes, centers, heights, width, color, labels=None, label_fmt='{:.2f}',
        label_color=None, alpha_fill=None):
    d = axes.d
    half = width / 2
    for i, (c, h) in enumerate(zip(centers, heights)):
        x = axes.px(c)
        ytop = axes.py(h)
        d.rectangle([x - half, ytop, x + half, axes.y1], fill=alpha_fill or color,
                    outline=color, width=3)
        if labels:
            d.text((x, ytop - 14), label_fmt.format(labels[i]), font=font(FS_DATA, True),
                   fill=label_color or color, anchor='mb')
