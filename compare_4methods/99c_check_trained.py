#!/usr/bin/env python3
"""B1 闸门（R-3 专用变体）：训练是否真的发生了。

历史事故：3DGS 的 restore() 直接赋值、不包 nn.Parameter，训练会正常打印完成、
loss 正常"波动"，但存出的 PLY 与初值逐字节相同。当时我是靠 md5 才发现的。
R-3 的 init 只留了 .pth（没存 iteration_0 的 PLY），所以这里直接读 ckpt 的 _xyz
与训练后 PLY 的 xyz 逐点比 —— 比 md5 多一个好处：能量出位置实际移动了多少。

用法（querysplat env）：
  python _res_gate.py <chkpnt.pth> <point_cloud.ply>
退出码 0 = 通过（参数确实变了），1 = 未通过（训练没生效）。
"""
import sys

import numpy as np
import torch
from plyfile import PlyData


def main():
    ckpt, ply = sys.argv[1], sys.argv[2]
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    args = obj[0] if isinstance(obj, (tuple, list)) else obj
    xyz0 = args[1].detach().cpu().numpy().astype(np.float64)

    v = PlyData.read(ply)["vertex"]
    xyz1 = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    print(f"init 点数 {len(xyz0):,}  训练后点数 {len(xyz1):,}")
    if len(xyz0) != len(xyz1):
        print(f"ℹ️ 点数变化 {len(xyz1) - len(xyz0):+,}（densify 或 prune 生效）")
        n = min(len(xyz0), len(xyz1))
        xyz0, xyz1 = xyz0[:n], xyz1[:n]
        print(f"   仅比较前 {n:,} 点")

    d = np.linalg.norm(xyz1 - xyz0, axis=1)
    med = float(np.median(d))
    p90 = float(np.percentile(d, 90))
    mx = float(d.max())
    print(f"位置位移：median {med * 1000:.3f} mm  p90 {p90 * 1000:.3f} mm  max {mx * 1000:.3f} mm")

    extent = float(np.linalg.norm(xyz1.max(0) - xyz1.min(0)))
    print(f"场景 extent {extent:.3f} m，median 位移相当于 {med / extent:.6f} × extent")

    if med <= 1e-9:
        print("❌ B1 未通过：位置一个都没动，参数没被训练")
        return 1
    print("✅ B1 通过：参数确实被优化了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
