#!/usr/bin/env python3
"""使用实际官方GaussianAdapter验证捕获相机必须包含其尺度；需DA3环境。"""
import importlib.util
from pathlib import Path

import numpy as np
import torch
from types import SimpleNamespace


def main():
    spec = importlib.util.spec_from_file_location('cam', Path(__file__).with_name('36_run_hisilicon_cameras.py'))
    cam = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cam)
    from depth_anything_3.model.gs_adapter import GaussianAdapter
    centers = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    E = torch.eye(4).repeat(1, 4, 1, 1)
    E[0, :, :3, 3] = -centers
    target = E.clone()
    target[..., :3, 3] *= 2
    K = torch.tensor([[4., 0., 2.], [0., 4., 2.], [0., 0., 1.]]).repeat(1, 4, 1, 1)
    adapter = GaussianAdapter(sh_degree=0, pred_offset_xy=False)
    raw = torch.zeros(1, 4, 4, 4, adapter.d_in)
    raw[..., 6] = 1
    kwargs = dict(extrinsics=E, intrinsics=K, depths=torch.ones(1, 4, 4, 4) * 3,
                  opacities=torch.ones(1, 4, 4, 4) * .5, raw_gaussians=raw, image_shape=(4, 4))
    with torch.no_grad():
        plain = adapter(**kwargs)
        conditioned = adapter(**kwargs, gt_extrinsics=target)
    np.testing.assert_allclose(conditioned.means.numpy(), plain.means.numpy() * 2, atol=2e-6)
    np.testing.assert_allclose(conditioned.scales.numpy(), plain.scales.numpy() * 2, atol=2e-6)
    fixed = cam.adapter_camera_extrinsics(E.numpy()[0], 2.)
    np.testing.assert_allclose(np.linalg.inv(fixed)[:, :3, 3], centers.numpy() * 2, atol=1e-6)
    p = plain.means.numpy().reshape(4, 16, 3)
    q = conditioned.means.numpy().reshape(4, 16, 3)
    p_cam = p + E.numpy()[0, :, None, :3, 3]
    q_cam = q + fixed[:, None, :3, 3]
    np.testing.assert_allclose(p_cam[..., :2] / p_cam[..., 2:], q_cam[..., :2] / q_cam[..., 2:], atol=1e-6)
    wrong_cam = q + E.numpy()[0, :, None, :3, 3]
    assert np.max(np.abs(p_cam[..., :2] / p_cam[..., 2:] - wrong_cam[..., :2] / wrong_cam[..., 2:])) > .1
    prediction = SimpleNamespace(gaussians=conditioned, gs_extrinsics=fixed, adapter_scale=2.)
    alignment = cam.transform_gltf_gaussians(prediction, E.numpy()[0])
    np.testing.assert_allclose(alignment['total_gaussian_scale'], .5, atol=1e-6)
    np.testing.assert_allclose(prediction.gaussians.means.numpy(), plain.means.numpy(), atol=2e-6)
    np.testing.assert_allclose(prediction.gaussians.scales.numpy(), plain.scales.numpy(), atol=2e-6)
    print('✅ 官方adapter尺度2回归通过：高斯中心/尺度、相机中心及投影一致；旧相机配对确实失败。')


if __name__ == '__main__':
    main()
