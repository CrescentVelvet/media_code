#!/usr/bin/env python3
"""检查 02 产物是否满足 SplatAnything 的尺度契约（GSN）。

三件事：
  1. 高斯 scale 中位数应与 spacing_target（1.3228e-3）同量级 —— 网络学的是
     "相对 spacing 的 scale"，scale 若离 1e-3 太远说明 GSN 没生效或尺度错位；
  2. 相邻高斯中心间距中位数应接近 voxel_size（0.002），因为合并就是这个网格；
  3. 相机中心分布应落在"高斯包围盒"量级内（同一坐标系）。

用法: python 05_check_scale.py --result <02 输出目录>
"""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True)
    ap.add_argument('--sample', type=int, default=200000, help='算间距时最多抽多少点')
    args = ap.parse_args()

    res = Path(args.result)
    ply = PlyData.read(str(res / 'gaussians.ply'))['vertex']
    means = np.stack([ply[n] for n in ('x', 'y', 'z')], 1).astype(np.float32)
    scales = np.exp(np.stack([ply[f'scale_{i}'] for i in range(3)], 1).astype(np.float32))
    opac = 1.0 / (1.0 + np.exp(-ply['opacity'].astype(np.float32)))

    cams = json.loads((res / 'cameras.json').read_text())
    C = np.array([e['matrix'] for e in cams['extrinsics']])[:, :3, 3]

    print(f'高斯 {len(means):,}')
    print(f'  means 包围盒  min {means.min(0).round(4)}  max {means.max(0).round(4)}')
    print(f'  means 范围    {np.ptp(means, 0).round(4)}')
    print(f'  scale 中位数  {np.median(scales, 0).round(6)}  → spacing 量级 {np.median(scales.max(1)):.6f}')
    print(f'  opacity 均值  {opac.mean():.4f}')
    print(f'相机 {len(C)}')
    print(f'  c2w 平移范围  min {C.min(0).round(3)}  max {C.max(0).round(3)}')
    print(f'  相机到原点距离 p50 {np.median(np.linalg.norm(C, axis=1)):.3f}  '
          f'min {np.linalg.norm(C, axis=1).min():.3f}  max {np.linalg.norm(C, axis=1).max():.3f}')

    # 最近邻间距：合并后的高斯中心应该落在 voxel_size 网格上，所以 1-NN 距离 ≈ voxel。
    # ⚠️ 不要用"包围盒体积/点数"估密度：物体是表面薄壳，包围盒体积会把密度算小 20 倍。
    n = len(means)
    idx = np.random.default_rng(0).choice(n, min(args.sample, n), replace=False)
    sub = means[idx]
    from scipy.spatial import cKDTree
    d, _ = cKDTree(sub).query(sub, k=2)          # k=1 是自己
    nn = d[:, 1]
    print(f'  1-NN 间距（{len(sub):,} 点抽样）p10 {np.percentile(nn,10):.5f}  '
          f'p50 {np.median(nn):.5f}  p90 {np.percentile(nn,90):.5f}')
    print(f'  voxel_size=0.002 的预期：p50 应 ≈ 0.002（同一格内被合并成一个）')

    target = 0.0013227513
    smed = float(np.median(scales.max(1)))
    print(f'\n判据: scale/spacing_target = {smed/target:.3f}（应在 0.3~3 量级）')


if __name__ == '__main__':
    main()
