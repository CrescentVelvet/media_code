#!/usr/bin/env python3
"""独立核验 glTF 相机的图像极线一致性及 adapter 高斯的相机尺度。

Env vars: CASE_DIR, RESULTS_DIR, AUDIT_SOURCE (可选，已保存原生 PLY 的诊断目录)
不推理、不修改输入；SIFT 匹配用图像估计 F 筛选，不用待测相机筛选。
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData


def homogeneous(ext):
    out = np.broadcast_to(np.eye(4), (len(ext), 4, 4)).copy()
    out[:, :3] = ext[:, :3]
    return out


def sampson(F, p, q):
    x = np.column_stack((p, np.ones(len(p))))
    y = np.column_stack((q, np.ones(len(q))))
    fx, fy = x @ F.T, y @ F
    return np.abs(np.sum(y * fx, axis=1)) / np.sqrt(
        np.sum(fx[:, :2] ** 2 + fy[:, :2] ** 2, axis=1).clip(1e-20))


def fundamental(ext, K, i, j):
    relative = ext[j] @ np.linalg.inv(ext[i])
    x, y, z = relative[:3, 3]
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.linalg.inv(K[j]).T @ skew @ relative[:3, :3] @ np.linalg.inv(K[i])


def stats(x):
    return dict(zip(('median', 'p90', 'p99'), np.quantile(x, [.5, .9, .99]).tolist()))


def main():
    case = Path(os.environ['CASE_DIR'])
    out = Path(os.environ['RESULTS_DIR'])
    out.mkdir(parents=True, exist_ok=False)
    data = json.loads((case / f'{case.name}_camera.json').read_text(encoding='utf-8'))
    path = data['extrinsics']['camera_path']
    c2w = np.asarray(path['c2w_keyframes'])
    ext = np.linalg.inv(c2w @ np.diag([1, -1, -1, 1]))
    n = len(ext)
    width = data['intrinsics']['reference_image_size']['width']
    height = data['intrinsics']['reference_image_size']['height']
    top = np.broadcast_to(np.asarray(data['intrinsics']['K']), (n, 3, 3)).copy()
    animated = top.copy()
    animated[:, 0, 0] = animated[:, 1, 1] = height / (2 * np.tan(np.asarray(path['yfov']).reshape(-1) / 2))
    sift = cv2.SIFT_create(nfeatures=6000)
    features = [sift.detectAndCompute(cv2.imread(str(case / 'images' / f'frame_{i+1:04d}.png'), 0), None) for i in range(n)]
    matcher = cv2.BFMatcher()
    variants = {'top_level_K': (ext, top), 'animated_K': (ext, animated),
                'reversed_order': (ext[::-1], animated[::-1]),
                'no_axis_flip': (np.linalg.inv(c2w), animated),
                'no_axis_flip_top_K': (np.linalg.inv(c2w), top),
                'flip_xz': (np.linalg.inv(c2w @ np.diag([-1, 1, -1, 1])), animated),
                'flip_xy': (np.linalg.inv(c2w @ np.diag([-1, -1, 1, 1])), animated),
                'as_w2c': (c2w, animated)}
    negative_centers = c2w.copy()
    negative_centers[:, :3, 3] *= -1
    variants['no_flip_negative_centers'] = (np.linalg.inv(negative_centers), animated)
    for label, diagonal in [('x', [-1, 1, 1]), ('y', [1, -1, 1]), ('z', [1, 1, -1]), ('yz', [1, -1, -1])]:
        basis = np.diag(diagonal)
        candidate = c2w.copy()
        candidate[:, :3, :3] = basis @ c2w[:, :3, :3] @ basis
        variants[f'conjugate_{label}'] = (np.linalg.inv(candidate), animated)
    errors = {k: [] for k in (*variants, 'image_fit_F')}
    front = {k: [0, 0] for k in variants}
    pairs = []
    cv2.setRNGSeed(42)
    for i in range(n):
        for j in range(i + 1, n):
            ki, di = features[i]
            kj, dj = features[j]
            good = [a for a, b in matcher.knnMatch(di, dj, k=2) if a.distance < .7 * b.distance]
            if len(good) < 20:
                continue
            p = np.asarray([ki[m.queryIdx].pt for m in good])
            q = np.asarray([kj[m.trainIdx].pt for m in good])
            F, mask = cv2.findFundamentalMat(p, q, cv2.USAC_MAGSAC, 1., .999, 10000)
            if F is None or mask is None or mask.sum() < 20:
                continue
            p, q = p[mask.ravel() != 0], q[mask.ravel() != 0]
            matrices = {k: fundamental(E, K, i, j) for k, (E, K) in variants.items()}
            matrices['image_fit_F'] = F
            row = {'views': [i, j], 'matches': len(p)}
            for label, f in matrices.items():
                e = sampson(f, p, q)
                errors[label].extend(e)
                row[label] = stats(e)
                if label in variants:
                    E, K = variants[label]
                    xyz = cv2.triangulatePoints(K[i] @ E[i, :3], K[j] @ E[j, :3], p.T, q.T)
                    xyz = xyz[:3] / xyz[3:]
                    positive = ((E[i, 2, :3] @ xyz + E[i, 2, 3]) > 0) & ((E[j, 2, :3] @ xyz + E[j, 2, 3]) > 0)
                    front[label][0] += int(positive.sum())
                    front[label][1] += len(p)
            pairs.append(row)
    if not pairs:
        raise RuntimeError('没有足够图像匹配，不能判定相机约定')
    report = {'case': str(case), 'image_size': [width, height],
              'focal_top': top[:, 0, 0].tolist(), 'focal_animation': animated[:, 0, 0].tolist(),
              'rotation_orthogonality_max': float(np.abs(c2w[:, :3, :3] @ c2w[:, :3, :3].transpose(0, 2, 1) - np.eye(3)).max()),
              'epipolar_pixel_error': {k: stats(v) for k, v in errors.items()},
              'triangulated_positive_depth_fraction': {k: a / b for k, (a, b) in front.items()}, 'pairs': pairs,
              'limits': '图像匹配极线检验约束相对位姿/内参，不能证明绝对尺度或独立深度真值。'}
    if os.environ.get('AUDIT_SOURCE'):
        source = Path(os.environ['AUDIT_SOURCE'])
        if (source / 'raw_cameras.npz').is_file():
            with np.load(source / 'raw_cameras.npz') as raw:
                if 'adapter_extrinsics_before_scale' in raw.files:
                    raise ValueError('该目录已经使用修正后的相机捕获；AUDIT_SOURCE只用于旧版未计adapter尺度的诊断产物')
        summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
        a = float(summary['adapter_pose_scale'])
        z = np.load(source / 'native_cameras.npz')
        cameras = {k: z[k] for k in z.files}
        before = homogeneous(cameras['extrinsics'])
        after = before.copy()
        after[:, :3, 3] *= a
        cameras['extrinsics'] = after[:, :3]
        cameras['camera_frame'] = 'adapter高斯世界；相机中心与深度同乘adapter_pose_scale'
        np.savez_compressed(out / 'corrected_native_cameras.npz', **cameras)
        v = PlyData.read(str(source / 'native.ply'))['vertex'].data
        h, w = cameras['image_shape']
        if len(v) != n * h * w:
            raise ValueError('需要未裁点且保持view/pixel顺序的原生PLY')
        indices = np.arange(0, int(h*w), 83)
        pixels = np.column_stack((indices % w + .5, indices // w + .5))
        pixel_errors = {'before': [], 'after': []}
        for i in range(n):
            sampled = v[i*h*w + indices]
            xyz = np.column_stack([sampled[k] for k in ('x', 'y', 'z')])
            for label, E in (('before', before), ('after', after)):
                xyz_cam = xyz @ E[i, :3, :3].T + E[i, :3, 3]
                uv = xyz_cam @ cameras['intrinsics'][i].T
                uv = uv[:, :2] / uv[:, 2:]
                pixel_errors[label].extend(np.linalg.norm(uv - pixels, axis=1))
        report['adapter_scale_audit'] = {'adapter_scale': a, 'sampled_pixel_projection_error': {k: stats(v) for k, v in pixel_errors.items()},
                                         'expected': '修正后投影误差含高斯头学习的offset_xy及相机计算误差；不是独立几何误差，也不保证offset为亚像素。'}
    (out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'pairs'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
