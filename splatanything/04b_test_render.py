#!/usr/bin/env python3
"""临时：造一个最小 result 目录，验证 03_render_eval.py 的渲染链路能跑通。

不验证画质，只验证：gsplat 调用 / PLY 字段解析 / cameras.json 解析 / 指标计算 / 出图。
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

ROOT = Path('/tmp/sa_render_test')
RES = 56
F = 50.0


def build_gaussians():
    """z=2 平面上 20×20 个高斯，颜色棋盘格。"""
    xs = np.linspace(-0.5, 0.5, 20)
    ys = np.linspace(-0.5, 0.5, 20)
    means, colors = [], []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            means.append([x, y, 2.0])
            c = 0.9 if (i + j) % 2 == 0 else 0.1
            colors.append([c, c, c])
    means = np.array(means, np.float32)
    colors = np.array(colors, np.float32)
    n = len(means)

    attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)] + \
            ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
    el = np.empty(n, dtype=[(a, 'f4') for a in attrs])
    el['x'], el['y'], el['z'] = means.T
    el['nx'] = el['ny'] = el['nz'] = 0
    # f_dc 存 SH 直流：rgb = 0.5 + C0 * f_dc
    dc = (colors - 0.5) / 0.28209479177387814
    for i in range(3):
        el[f'f_dc_{i}'] = dc[:, i]
    el['opacity'] = np.log(0.99 / 0.01)                       # logit
    for i in range(3):
        el[f'scale_{i}'] = np.log(0.03)                        # log scale
    el['rot_0'] = 1.0
    el['rot_1'] = el['rot_2'] = el['rot_3'] = 0.0
    return el, n


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'rgb').mkdir(exist_ok=True)

    el, n = build_gaussians()
    PlyData([PlyElement.describe(el, 'vertex')]).write(str(ROOT / 'gaussians.ply'))

    cams = {'extrinsics': [], 'intrinsics': []}
    for i, dz in enumerate([0.0, 0.05]):
        c2w = np.eye(4)
        c2w[0, 3] = dz
        cams['extrinsics'].append({'camera_id': f'{i:03d}', 'matrix': c2w.tolist()})
        cams['intrinsics'].append({'camera_id': f'{i:03d}',
                                   'matrix': [[F, 0, RES / 2], [0, F, RES / 2], [0, 0, 1]]})
    (ROOT / 'cameras.json').write_text(json.dumps(cams))

    # GT 先用纯灰占位：只验流程，PSNR 数值本身无意义
    for i in range(2):
        Image.fromarray(np.full((RES, RES, 3), 128, np.uint8)).save(ROOT / 'rgb' / f'{i:03d}.png')

    r = subprocess.run([sys.executable, str(Path(__file__).with_name('03_render_eval.py')),
                        '--result', str(ROOT), '--out', str(ROOT / 'eval')])
    if r.returncode != 0:
        raise SystemExit(1)

    s = json.loads((ROOT / 'eval' / 'eval_summary.json').read_text())
    print('summary:', {k: s[k] for k in ('views', 'gaussians', 'resolution', 'psnr_mean', 'alpha_mean')})
    # 灰色 GT vs 棋盘渲染：alpha 覆盖要显著 > 0，说明高斯真的被光栅化了
    assert s['gaussians'] == n, s['gaussians']
    assert s['alpha_mean'] > 0.05, f'alpha 太低（{s["alpha_mean"]:.3f}），可能没渲到东西'
    assert (ROOT / 'eval' / 'render' / '000.png').exists()
    assert (ROOT / 'eval' / 'contact_sheet.png').exists()
    print('✅ 渲染链路通过')


if __name__ == '__main__':
    main()
