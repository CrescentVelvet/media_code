#!/usr/bin/env python3
"""AUDIT-21b · 人脸窗标定用的"网格 + 方框"检查图。

为什么要有这个脚本：`audit_21` 发现固定人脸框在 view 8 落在书架上，把人脸锐度虚高 +180、
把 I-03 的锐度差从 −5 夸大成 −69（ERRATA E12）。定框这件事不能靠"看着差不多"，
必须**画出来 + 带像素刻度 + 目视核对**。

用法：
    python audit_21b_face_box_grid.py            # 默认三个视角 + 当前生效的框
    BOXES="0:0.42,0.14,0.58,0.30" python ...     # 只画一个视角、试一个候选框

输出：`<OUT>/box_<view>.png`，红色方框 + 50 px 网格 + 刻度。
只读渲染图，不写产物目录以外的任何东西。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SC = os.environ["SC"]
SRC = os.environ.get("SRC", os.path.join(SC, "i03b_oracle"))
DST = os.environ.get("OUT", os.path.join(SC, "face_box_grid"))
PREFIX = os.environ.get("PREFIX", "3dgs")

# 与 audit_17 / audit_21 保持同一套框（改这里要同步那两处）
DEFAULT = {0: (0.415, 0.140, 0.585, 0.300),
           8: (0.515, 0.145, 0.665, 0.310),
           16: (0.430, 0.155, 0.600, 0.310)}


def parse_boxes(s):
    out = {}
    for item in s.split(";"):
        item = item.strip()
        if not item:
            continue
        v, rest = item.split(":")
        out[int(v)] = tuple(float(x) for x in rest.split(","))
    return out


BOXES = parse_boxes(os.environ["BOXES"]) if os.environ.get("BOXES") else DEFAULT
os.makedirs(DST, exist_ok=True)

for v, b in BOXES.items():
    p = os.path.join(SRC, "%s_view%03d.png" % (PREFIX, v))
    if not os.path.exists(p):
        print("跳过（缺图）:", p)
        continue
    im = np.asarray(Image.open(p))
    H, W = im.shape[:2]
    x0, y0, x1, y1 = int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)
    fig, ax = plt.subplots(figsize=(5.2, 6.8), dpi=150)
    ax.imshow(im)
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=1.6))
    ax.set_xticks(range(0, W, 50))
    ax.set_yticks(range(0, H, 50))
    ax.tick_params(labelsize=7)
    ax.grid(color="cyan", lw=0.4, alpha=0.4)
    ax.set_title("view %d  box=(%d,%d,%d,%d)" % (v, x0, y0, x1, y1), fontsize=9)
    fig.tight_layout()
    q = os.path.join(DST, "box_%03d.png" % v)
    fig.savefig(q)
    plt.close(fig)
    print("💾", q)
