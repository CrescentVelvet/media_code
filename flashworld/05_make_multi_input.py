#!/usr/bin/env python3
"""多图条件实验的输入 json 生成器。

在已有单图 json 基础上，把 image_prompt/image_index 换成 image_prompts 列表。
相机轨迹不变；条件帧索引直接对应 json cameras 的序号（窄轨迹 json 的相机 i
= 采集包帧 i，已验证 max|dpos|=5e-7）。

用法：
  BASE=<已有 json> PNG_DIR=<png 目录> COND=0,11,23 OUT=<输出 json> \
      python flashworld/05_make_multi_input.py

可选 FRAME_IDXS：逗号分隔的「槽位→采集包帧号」映射（长度须等于 cameras 数）。
用于长轨迹实验（如 41 槽均匀采自 82 帧），槽位号与 png 帧号不一致时指定；
不指定则恒等映射（槽位号=帧号）。
"""
import json
import os
import sys
from pathlib import Path

BASE = os.environ.get("BASE", "")
PNG_DIR = os.environ.get("PNG_DIR", "")
COND = os.environ.get("COND", "")
OUT = os.environ.get("OUT", "")
FRAME_IDXS = os.environ.get("FRAME_IDXS", "")
if not (BASE and PNG_DIR and COND and OUT):
    sys.exit("❌ 需要 BASE/PNG_DIR/COND/OUT 四个环境变量")

d = json.loads(Path(BASE).read_text(encoding="utf-8"))
n_cam = len(d["cameras"])
indices = [int(x) for x in COND.split(",")]
for i in indices:
    if not (0 <= i < n_cam):
        sys.exit("❌ 条件帧 %d 越界（共 %d 帧）" % (i, n_cam))

if FRAME_IDXS:
    slot2frame = [int(x) for x in FRAME_IDXS.split(",")]
    if len(slot2frame) != n_cam:
        sys.exit("❌ FRAME_IDXS 长度 %d != cameras 数 %d" % (len(slot2frame), n_cam))
else:
    slot2frame = list(range(n_cam))

pngs = sorted(Path(PNG_DIR).glob("*.png"))
by_idx = {}
for p in pngs:
    try:
        by_idx[int(p.name.split("_")[0])] = p
    except ValueError:
        continue

prompts = []
for i in indices:
    fi = slot2frame[i]
    if fi not in by_idx:
        sys.exit("❌ 找不到采集帧 %03d 的 png（槽位 %d，%s）" % (fi, i, PNG_DIR))
    # 写 /mnt/d 风格路径，推理在 WSL 里跑
    path = str(by_idx[fi]).replace("\\", "/")
    if path.startswith("D:/"):
        path = "/mnt/d/" + path[3:]
    prompts.append({"path": path, "index": i})

d.pop("image_prompt", None)
d.pop("image_index", None)
d["image_prompts"] = prompts

out = Path(OUT)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(d, f, ensure_ascii=False, indent=2)
print("✅ %s（条件帧 %s）" % (out, indices))
