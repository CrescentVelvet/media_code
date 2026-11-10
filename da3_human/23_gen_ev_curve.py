#!/usr/bin/env python3
"""生成 UE 渲染用的逐帧曝光补偿曲线（ev_curve.csv）——walk180 视频版（暂缓）。

⚠️ 现行规格以 24_ue_render_spec.md 为准（稀疏视角 + 三摄同位，静帧曝光走
25_gen_view_ev.py）。本脚本保留供视频方案重启时使用。
seed 由 seq_id 派生（crc32），同一 seq_id 永远生成同一条曲线——
同事导入 UE 渲 v3，我们后期在 v0 上施加 v1，两边逐帧一致。

Env vars:
  SEQ_ID    序列名（如 anna_walk180_cw），决定 seed 与输出文件名
  N_FRAMES  帧数，默认 180
  OUT_DIR   输出目录，默认当前目录
"""
import os
import sys
import zlib

import numpy as np

EV_RANGE = 0.3        # 随机游走幅度（EV）
EV_MAX_STEP = 0.08    # 相邻帧最大变化（EV）
BUMP_EV = 0.5         # 中段 ramp 幅度（EV）
BUMP_WIDTH = 30       # ramp 帧数（升余弦单瓣）
BUMP_LO, BUMP_HI = 0.4, 0.6   # ramp 中心所在帧区间（占 N 比例）


def smooth_walk(n, rng, sigma):
    """白噪声 cumsum + 高斯平滑，归一化到 [-1, 1]。"""
    x = np.cumsum(rng.standard_normal(n))
    k = int(sigma * 6) | 1          # 奇数长度的平滑核窗口
    t = np.arange(k) - k // 2
    g = np.exp(-0.5 * (t / sigma) ** 2)
    g /= g.sum()
    x = np.convolve(x, g, mode="same")
    x /= np.abs(x).max() + 1e-12
    return x


def main():
    seq_id = os.environ.get("SEQ_ID", "")
    if not seq_id:
        sys.exit("❌ SEQ_ID not set（例: SEQ_ID=anna_walk180_cw）")
    n = int(os.environ.get("N_FRAMES", "180"))
    out_dir = os.environ.get("OUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)

    seed = zlib.crc32(seq_id.encode("utf-8"))
    rng = np.random.default_rng(seed)

    base = smooth_walk(n, rng, sigma=5.0) * EV_RANGE
    step = np.abs(np.diff(base)).max()
    if step > EV_MAX_STEP:                       # 理论上平滑后极小，保险钳一下
        base *= EV_MAX_STEP / step

    # 中段 +EV ramp：升余弦单瓣，能量集中在 BUMP_WIDTH 帧内
    c0, c1 = int(n * BUMP_LO), int(n * BUMP_HI)
    center = int(rng.integers(c0 + BUMP_WIDTH, max(c1 - BUMP_WIDTH, c0 + BUMP_WIDTH + 1)))
    half = BUMP_WIDTH // 2
    lo, hi = max(center - half, 0), min(center + half, n)
    t = np.arange(lo, hi)
    bump = np.zeros(n)
    bump[lo:hi] = BUMP_EV * 0.5 * (1.0 - np.cos(2.0 * np.pi * (t - lo) / max(hi - lo - 1, 1)))

    ev = np.clip(base + bump, -EV_RANGE - BUMP_EV, EV_RANGE + BUMP_EV)

    path = os.path.join(out_dir, f"{seq_id}_ev_curve.csv")
    with open(path, "w", encoding="utf-8", newline="\n") as f:   # 仓规：防 CRLF
        f.write("frame,ev\n")
        for i, v in enumerate(ev):
            f.write(f"{i},{v:.5f}\n")

    print(f"🚀 [{seq_id}] EV 曲线: {n} 帧, seed={seed}")
    print(f"   随机游走 ±{EV_RANGE} EV | ramp +{BUMP_EV} EV @帧 {lo}~{hi} | "
          f"相邻帧 max|Δ|={np.abs(np.diff(ev)).max():.4f} EV | 值域 [{ev.min():+.3f}, {ev.max():+.3f}]")
    print(f"✅ {path}")


if __name__ == "__main__":
    main()
