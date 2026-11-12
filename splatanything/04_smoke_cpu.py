#!/usr/bin/env python3
"""CPU 冒烟：不加载真权重，验证 inference.py 全链路（相机 → 模型 → GSN → 导出）。

用随机初始化的两个模型跑 2 张 56×56 图、pose 模式。目的是提前暴露接口错误
（config 白名单、相机 JSON 校验、GSN、体素合并、PLY 导出），不验证画质。
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

REPO = Path.home() / 'repos/SplatAnything'
sys.path.insert(0, str(REPO))

from inference import main  # noqa: E402

RES = 56
K = [[40.0, 0.0, 28.0], [0.0, 40.0, 28.0], [0.0, 0.0, 1.0]]


def make_scene(root: Path):
    scene = root / 'scene'
    (scene / 'images').mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(2):
        arr = (rng.random((RES, RES, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(scene / 'images' / f'{i:03d}.png')
    cams = {'extrinsics': [], 'intrinsics': []}
    for i in range(2):
        c2w = np.eye(4)
        c2w[0, 3] = 0.2 * i          # 给一点基线，避免 da3_pose_align_factor 退化
        cams['extrinsics'].append({'camera_id': f'{i:03d}', 'matrix': c2w.tolist()})
        cams['intrinsics'].append({'camera_id': f'{i:03d}', 'matrix': K})
    (scene / 'prior_camera.json').write_text(json.dumps(cams))
    # 权重路径只做存在性检查，内容由 patch 跳过
    ckpt = root / 'fake.ckpt'
    ckpt.write_bytes(b'0')
    return scene, ckpt


def main_smoke():
    tmp = Path(tempfile.mkdtemp(prefix='sa_smoke_'))
    try:
        scene, ckpt = make_scene(tmp)
        out = tmp / 'out'
        argv = ['--input_path', str(scene), '--mode', 'pose', '--device', 'cpu',
                '--resolution', str(RES), str(RES), '--output_path', str(out),
                '--ckpt_path', str(ckpt), '--geometry_ckpt_path', str(ckpt),
                '--no_depth_rgb']
        with patch('src.utils.checkpoints.load_model_weights'):
            main(argv)
        d = out / 'scene'
        files = sorted(p.name for p in d.iterdir())
        print('\n[smoke] 产物:', files)
        for name in ('gaussians.ply', 'points.ply', 'cameras.json', 'config.yaml', 'run.json'):
            assert (d / name).exists(), f'缺 {name}'
        cam = json.loads((d / 'cameras.json').read_text())
        assert len(cam['extrinsics']) == 2, cam
        print(f'[smoke] ✅ 全链路通过（{len(cam["extrinsics"])} 相机）')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main_smoke()
