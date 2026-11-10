#!/usr/bin/env python3
"""生成仿真数据集逐视角曝光补偿（view_ev.csv），供后期在 v0 干净图上施加 v1 曝光差异。

规格见 24_ue_render_spec.md §6：
  MODE=sparse：视角间 EV 平滑变化 ±0.3 EV（模拟不同方位 AE 不同）；
  MODE=triple：三摄固定偏移（广角 +0.15 / 主摄 0 / 长焦 -0.15）+ 场景级 ±0.2 抖动。
seed 由 seq_id 派生（crc32），同一 seq_id 永远生成同一文件，可复现。

Env vars:
  SEQ_ID    序列名（如 ada_env1），决定 seed 与输出文件名
  MODE      sparse（13 视角）| triple（3 摄像头）
  OUT_DIR   输出目录，默认当前目录
"""
import os
import sys
import zlib

import numpy as np

SPARSE_RANGE = 0.3        # 稀疏环视角间 EV 幅度
TRIPLE_OFFSETS = {"wide": 0.15, "main": 0.0, "tele": -0.15}   # 三摄固定 EV 偏移
TRIPLE_SCENE_JITTER = 0.2  # 场景级 EV 抖动幅度


def sparse_curve(n, rng):
    """低频随机游走归一化到 [-1, 1]：白噪声 cumsum + 高斯平滑。"""
    x = np.cumsum(rng.standard_normal(n))
    sigma = 2.0
    k = int(sigma * 6) | 1
    t = np.arange(k) - k // 2
    g = np.exp(-0.5 * (t / sigma) ** 2)
    g /= g.sum()
    x = np.convolve(x, g, mode="same")
    return x / (np.abs(x).max() + 1e-12)


def main():
    seq_id = os.environ.get("SEQ_ID", "")
    mode = os.environ.get("MODE", "")
    if not seq_id:
        sys.exit("❌ SEQ_ID not set（例: SEQ_ID=ada_env1）")
    if mode not in ("sparse", "triple"):
        sys.exit("❌ MODE must be sparse|triple")
    out_dir = os.environ.get("OUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)

    seed = zlib.crc32(f"{seq_id}_{mode}".encode("utf-8"))
    rng = np.random.default_rng(seed)

    if mode == "sparse":
        views = [f"s{i:02d}" for i in range(13)]
        ev = sparse_curve(len(views), rng) * SPARSE_RANGE
    else:
        views = ["wide", "main", "tele"]
        scene = (rng.random() * 2.0 - 1.0) * TRIPLE_SCENE_JITTER
        ev = [TRIPLE_OFFSETS[v] + scene for v in views]

    path = os.path.join(out_dir, f"{seq_id}_{mode}_view_ev.csv")
    with open(path, "w", encoding="utf-8", newline="\n") as f:   # 仓规：防 CRLF
        f.write("view,ev\n")
        for v, e in zip(views, ev):
            f.write(f"{v},{e:+.5f}\n")

    print(f"🚀 [{seq_id}/{mode}] 逐视角 EV: {len(views)} 个视角, seed={seed}")
    print(f"   值域 [{min(ev):+.3f}, {max(ev):+.3f}] EV")
    print(f"✅ {path}")


if __name__ == "__main__":
    main()
