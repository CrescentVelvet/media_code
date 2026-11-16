"""Reusable algorithm diagrams and local-texture illustrations for figkit.

Content and page-specific layouts belong in the external figure build directory.
Text overflow raises an error instead of silently truncating scientific labels.
Uses SPEC.md section 5.9 and the existing figlib font/color conventions.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

import figlib as F


def tint(color, amount=0.07):
    return tuple(round(255 * (1 - amount) + c * amount) for c in color)


class Diagram:
    """1920x1080 slide figure with checked text, nodes and routed arrows."""

    def __init__(self, title, subtitle, footer):
        self.image, self.draw = F.canvas()
        self.footer = footer
        self.records = []
        self.text((96, 32, 1856, 89), title, 42, bold=True)
        self.text((98, 90, 1856, 125), subtitle, 22, F.INK2)
        self.draw.rectangle((64, 40, 72, 92), fill=F.INDIGO)
        self.draw.line((64, 132, 1856, 132), fill=F.RULE, width=2)

    def text(self, box, value, size=23, color=F.INK, bold=False, align="left"):
        x0, y0, x1, y1 = box
        lines = str(value).split("\n")
        line_h = math.ceil(size * 1.42)
        glyph_h = max(self.draw.textbbox((0, 0), line or ' ',
                      font=F.font(size, bold), anchor="lt")[3] for line in lines)
        if line_h * (len(lines)-1) + glyph_h > y1 - y0:
            raise ValueError(f"Text too tall: {value!r} in {box}")
        for i, line in enumerate(lines):
            width = self.draw.textlength(line, font=F.font(size, bold))
            if width > x1 - x0:
                raise ValueError(f"Text too wide ({width:.1f}): {line!r} in {box}")
            x = x0 if align == "left" else (x0 + x1 - width) / 2
            self.draw.text((x, y0 + i * line_h), line,
                           font=F.font(size, bold), fill=color, anchor="lt")
        self.records.append(dict(box=list(box), text=str(value), size=size))

    def panel(self, box, color=F.INDIGO, fill=True):
        F.rrect(self.draw, box, 16, tint(color) if fill else F.BG,
                tint(color, 0.28), 2)

    def node(self, box, title, body="", color=F.CYAN, tag=None):
        x0, y0, x1, y1 = box
        self.panel(box, color)
        self.draw.rounded_rectangle((x0 + 2, y0 + 2, x0 + 10, y1 - 2),
                                    radius=4, fill=color)
        top = y0 + 23
        if tag:
            self.text((x0 + 26, top, x1 - 22, top + 30), tag, 19, color, True)
            top += 38
        self.text((x0 + 26, top, x1 - 22, top + 46), title, 23, bold=True)
        if body:
            self.text((x0 + 26, top + 46, x1 - 22, y1 - 15), body, 22, F.INK2)

    def arrow(self, points, color=F.INK2, width=3, dashed=False):
        points = list(points)
        if len(points) < 2:
            raise ValueError("An arrow requires at least two points")
        for start, end in zip(points, points[1:]):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if dashed and length:
                for t in range(0, math.ceil(length), 18):
                    a, b = t / length, min(t + 10, length) / length
                    self.draw.line((start[0]+a*dx, start[1]+a*dy,
                                    start[0]+b*dx, start[1]+b*dy), fill=color, width=width)
            else:
                self.draw.line((start, end), fill=color, width=width)
        x, y = points[-1]
        dx, dy = x - points[-2][0], y - points[-2][1]
        length = math.hypot(dx, dy)
        if not length:
            raise ValueError("Arrow's final segment must have nonzero length")
        ux, uy = dx/length, dy/length
        self.draw.polygon([(x, y), (x-14*ux+6*uy, y-14*uy-6*ux),
                           (x-14*ux-6*uy, y-14*uy+6*ux)], fill=color)

    def strip(self, y, title, body, color=F.AMBER):
        self.panel((64, y, 1856, y+105), color)
        self.text((90, y+18, 1828, y+55), title, 23, color, True)
        self.text((90, y+60, 1828, y+96), body, 22)

    def save(self, path):
        self.text((64, 1044, 1856, 1070), self.footer, 17, F.INK2)
        for item in self.records:
            x0, y0, x1, y1 = item["box"]
            if not (0 <= x0 < x1 <= F.W and 0 <= y0 < y1 <= F.H):
                raise ValueError(f"Text outside canvas: {item}")
        self.image.save(path, optimize=True)
        return self.records


def texture_surfels(size, rgb, alpha=None, grid=True):
    """Illustrate one affine tangent plane with bilinear RGB/alpha texture.

    This is a schematic, not a physical Gaussian rasterizer. rgb has shape
    [rows, cols, 3] in [0,1]; alpha, when supplied, has shape [rows, cols].
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("rgb must have shape [rows, cols, 3]")
    if not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("rgb values must be finite and in [0, 1]")
    rows, cols = rgb.shape[:2]
    alpha = np.ones((rows, cols), np.float32) if alpha is None else np.asarray(alpha, np.float32)
    if alpha.shape != (rows, cols) or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError("alpha must match the texture grid and lie in [0, 1]")
    w, h = size
    yy, xx = np.mgrid[:h, :w]
    center = np.array([w * .5, h * .5])
    basis = np.array([[w*.37, w*.11], [-h*.12, h*.36]])
    uv = np.stack((xx-center[0], yy-center[1]), -1) @ np.linalg.inv(basis).T
    u, v = uv[..., 0], uv[..., 1]
    tx = np.clip((u+1)*.5*(cols-1), 0, cols-1)
    ty = np.clip((v+1)*.5*(rows-1), 0, rows-1)
    x0, y0 = tx.astype(int), ty.astype(int)
    x1, y1 = np.minimum(x0+1, cols-1), np.minimum(y0+1, rows-1)
    dx, dy = tx-x0, ty-y0

    def sample(t):
        weights = ((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy)
        values = (t[y0,x0], t[y0,x1], t[y1,x0], t[y1,x1])
        return sum(weight[..., None]*value if t.ndim == 3 else weight*value
                   for weight, value in zip(weights, values))

    support = (u*u+v*v <= 1)
    opacity = np.exp(-1.6*(u*u+v*v)) * sample(alpha) * support
    checker = np.where((xx//24+yy//24)%2, .94, .985)
    pixels = checker[..., None]*(1-opacity[..., None])+sample(rgb)*opacity[..., None]
    im = Image.fromarray(np.uint8(np.clip(pixels, 0, 1)*255))
    overlay = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(overlay)

    def project(a, b):
        return tuple(center + basis @ np.array([a, b]))

    if grid:
        for i in range(1, cols):
            a = -1+2*i/cols
            d.line((project(a,-1), project(a,1)), fill=F.RULE, width=2)
        for i in range(1, rows):
            a = -1+2*i/rows
            d.line((project(-1,a), project(1,a)), fill=F.RULE, width=2)
    mask = np.asarray(overlay).min(axis=2) < 255
    im.paste(overlay, mask=Image.fromarray(np.uint8(mask & support)*255))
    d = ImageDraw.Draw(im)
    boundary = [project(math.cos(t), math.sin(t)) for t in np.linspace(0, 2*math.pi, 160)]
    d.line(boundary, fill=F.CYAN, width=3)
    return im
