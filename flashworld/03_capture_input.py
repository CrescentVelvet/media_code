#!/usr/bin/env python3
"""Remy 采集包（transforms.json + image/*.heic）→ FlashWorld 输入 json。

与 02_make_input.py 的区别：02 是给「单张裸图」自造环绕轨迹；本脚本直接复用
采集包自己的真实相机轨迹（82 帧里均匀抽 24），内参也用 transforms.json 的真实值。

这样的好处：
- 生成的 3DGS 就在采集包的坐标系里，能和 vggt_human/som_human 的 COLMAP 重建直接对比；
- 内参是真的（fl_x/fl_y/cx/cy），不用猜 FOV。

坐标系约定（和 02_make_input.py 保持一致）：
  - transforms.json 的 transform_matrix 是 OpenGL 约定 c2w（看 -Z），FlashWorld 要的也是 OpenGL c2w；
  - FlashWorld json 里四元数是 [w, x, y, z]；
  - app.py 会对输入图做 center crop 到 image_height×image_width，并相应缩放 fx/cx——
    所以这里填的必须是「原始 1440×1920 图」的内参，不能填 480×704 的。

输出 json 的 cameras 数固定 24（FlashWorld 训练/推理的帧数）。

Env vars:
  PKG          采集包目录（含 transforms.json + image/*.heic），必填
  OUTPUT       json 输出目录
  FRAME        输入图在 82 帧里的序号（默认 0，000 是正面全身）
  TEXT         提示词（默认空）
  N_FRAME      输出相机数（默认 24）
  HEIGHT/WIDTH 输出分辨率（默认 704/480 → 竖屏 480x704；采集包是竖屏 1440x1920）
  PNG_DIR      已转好的 png 目录（默认 <OUT>/../../flashworld_input/<包名>_png 由外部负责转）
  IMAGE        直接给一张已转好的 png/jpg 路径（覆盖 FRAME）
"""
import json
import math
import os
import sys
from pathlib import Path

PKG = os.environ.get("PKG", "")
OUTPUT = os.environ.get("OUTPUT", "")
FRAME = int(os.environ.get("FRAME", "0"))
TEXT = os.environ.get("TEXT", "")
N_FRAME = int(os.environ.get("N_FRAME", "24"))
HEIGHT = int(os.environ.get("HEIGHT", "704"))
WIDTH = int(os.environ.get("WIDTH", "480"))
IMAGE = os.environ.get("IMAGE", "")

if not PKG:
    sys.exit("❌ PKG not set")
PKG = Path(PKG)
if not (PKG / "transforms.json").exists():
    sys.exit("❌ 不是 Remy 采集包（缺 transforms.json）: %s" % PKG)

meta = json.loads((PKG / "transforms.json").read_text(encoding="utf-8"))
frames = meta["frames"]
n_total = len(frames)

# 均匀抽 N_FRAME 个相机（纯 python，不依赖 numpy）
idxs = sorted(set(round(i * (n_total - 1) / (N_FRAME - 1)) for i in range(N_FRAME)))
# 保证 FRAME 在采样里
if FRAME not in idxs:
    nearest = min(range(len(idxs)), key=lambda k: abs(idxs[k] - FRAME))
    idxs[nearest] = FRAME
    idxs = sorted(set(idxs))
# 若 set 去重导致不足，向两侧补
step = max(1, n_total // N_FRAME)
i = 0
while len(idxs) < N_FRAME and i < n_total:
    if i not in idxs:
        idxs.append(i)
        idxs.sort()
    i += step
idxs = idxs[:N_FRAME]
image_index = idxs.index(FRAME)


def mat_to_quat_wxyz(m):
    """旋转矩阵（3x3 list）→ [w,x,y,z] 四元数。"""
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return [w, x, y, z]

cams = []
for i in idxs:
    f = frames[i]
    M = f["transform_matrix"]
    R = [[M[r][c] for c in range(3)] for r in range(3)]
    t = [M[r][3] for r in range(3)]
    q = mat_to_quat_wxyz(R)
    cams.append({
        "quaternion": [round(float(v), 6) for v in q],
        "position": [round(float(v), 6) for v in t],
        # 原始 1440x1920 内参，app.py 会按 crop/scale 自动换算
        "fx": float(f["fl_x"]), "fy": float(f["fl_y"]),
        "cx": float(f["cx"]), "cy": float(f["cy"]),
    })

# 输入图路径：优先 IMAGE 环境变量（已转好的 png），否则从 PKG 里取 FRAME 对应的 heic
if IMAGE:
    img_path = Path(IMAGE)
else:
    rel = frames[FRAME]["file_path"].lstrip("./")
    img_path = PKG / rel
    print("⚠️  IMAGE 未设，用原始 HEIC：%s（PIL 需要 pillow-heif）" % img_path.name)

if not img_path.exists():
    sys.exit("❌ 输入图不存在: %s" % img_path)

out_dir = Path(OUTPUT) if OUTPUT else Path("flashworld_input") / PKG.name
out_dir.mkdir(parents=True, exist_ok=True)
out_json = out_dir / ("%s_frame%03d.json" % (PKG.name, FRAME))

payload = {
    "image_prompt": str(img_path).replace("\\", "/"),
    "text_prompt": TEXT,
    "resolution": [N_FRAME, HEIGHT, WIDTH],
    "image_index": image_index,
    "cameras": cams,
}
with open(out_json, "w", encoding="utf-8", newline="\n") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print("✅ %s" % out_json)
print("   包: %s（共 %d 帧）" % (PKG.name, n_total))
print("   输入帧: %d -> image_index=%d" % (FRAME, image_index))
print("   采样帧: %s" % list(idxs))
print("   分辨率: %d 帧 x %d x %d" % (N_FRAME, HEIGHT, WIDTH))
