#!/usr/bin/env python3
"""同输入/分辨率对比 RGB、声明的 glTF 约定和图像极线支持的相机解释。

Env vars: CASE_DIR, RESULTS_DIR, DA3_MODEL_DIR, PROCESS_RES=704,
          BRANCHES=rgb,declared_gltf,negative_centers, SOURCE_INDICES (可选，0基)
negative_centers 仅为经图像检验的诊断坐标，不是通用 glTF 转换规则。
输出是 adapter 世界，配套相机包含 adapter 尺度；不覆盖正式 PLY。
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch


def main():
    script = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('camera', script / '36_run_hisilicon_cameras.py')
    cam = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cam)
    case = Path(os.environ['CASE_DIR'])
    out = Path(os.environ['RESULTS_DIR'])
    out.mkdir(parents=True, exist_ok=False)
    if os.environ.get('GLTF_CAMERA_CONVENTION', 'gltf') != 'gltf':
        raise ValueError('对照入口须保持GLTF_CAMERA_CONVENTION=gltf，特殊约定由negative_centers分支单独应用')
    files, ext, K, camera_path, manifest = cam.load_gltf_inputs(case)
    data = json.loads(camera_path.read_text(encoding='utf-8'))
    corrected_c2w = np.asarray(data['extrinsics']['camera_path']['c2w_keyframes'])
    corrected_c2w[:, :3, 3] *= -1
    corrected = np.linalg.inv(corrected_c2w)
    all_files = files
    indices = np.asarray([int(x) for x in os.environ.get('SOURCE_INDICES', ','.join(map(str, range(len(files))))).split(',')])
    if len(indices) < 3 or len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= len(files):
        raise ValueError('SOURCE_INDICES需要至少三个不重复的有效索引')
    targets = np.asarray([i for i in range(len(files)) if i not in indices], dtype=int)
    all_K = K.copy()
    files = [all_files[i] for i in indices]
    ext, K = ext[indices], K[indices]
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils import gsply_helpers as gh
    from depth_anything_3.utils.pose_align import batch_align_poses_umeyama, align_poses_umeyama
    gh.export_ply = cam._io.fast_export_ply
    gh.inverse_sigmoid = lambda x: torch.logit(x.clamp(torch.finfo(x.dtype).tiny, 1 - torch.finfo(x.dtype).eps))
    model_path = os.environ['DA3_MODEL_DIR']
    model = DepthAnything3.from_pretrained(model_path).to('cuda').eval()
    adapter = model.model.da3.gs_adapter
    original = adapter.forward
    captured = {}

    def capture(*args, **kwargs):
        E = kwargs['extrinsics'].detach().float().clone()
        a = 1.
        if kwargs.get('gt_extrinsics') is not None and not torch.equal(kwargs['extrinsics'], kwargs['gt_extrinsics']):
            _, _, scale = batch_align_poses_umeyama(kwargs['gt_extrinsics'].detach().float(), E)
            a = float(scale.clamp(1 / 3, 3)[0])
        # adapter 对中心和深度同时乘 a；w2c 平移也乘 a。
        E[..., :3, 3] *= a
        captured.update(extrinsics=E.cpu().numpy()[0],
                        intrinsics=kwargs['intrinsics'].detach().float().cpu().numpy()[0],
                        adapter_scale=a)
        result = original(*args, **kwargs)
        return result

    variants = {'rgb': None, 'declared_gltf': ext, 'negative_centers': corrected[indices]}
    report = {'case': str(case), 'model': model_path, 'model_sha256': cam._io.sha256(Path(model_path) / 'model.safetensors'),
              'process_res': int(os.environ.get('PROCESS_RES', '704')), 'seed': 42,
              'camera_json_sha256': cam._io.sha256(camera_path),
              'input_sha256': [cam._io.sha256(p) for p in files], 'loader_manifest': manifest, 'branches': {}}
    report.update(source_indices=indices.tolist(), target_indices=targets.tolist())
    for label in os.environ.get('BRANCHES', 'rgb,declared_gltf,negative_centers').split(','):
        ex = variants[label]
        branch = out / label
        branch.mkdir()
        torch.manual_seed(42)
        np.random.seed(42)
        captured.clear()
        print(f'🚀 {label}', flush=True)
        adapter.forward = capture
        try:
            pred = model.inference([str(p) for p in files], extrinsics=ex, intrinsics=K if ex is not None else None,
                                   process_res=report['process_res'], infer_gs=True,
                                   ref_view_strategy='saddle_balanced', export_dir=None, export_format='')
        finally:
            adapter.forward = original
        np.savez_compressed(branch / 'cameras.npz', extrinsics=captured['extrinsics'], intrinsics=captured['intrinsics'],
                            image_shape=np.asarray(pred.depth.shape[-2:]), image_names=np.asarray([p.name for p in files]),
                            camera_frame='adapter输出世界，已计入adapter尺度')
        # 仅使用source拟合一个Sim(3)；相同物理target相机映射到各自模型世界。
        R, t, scale = align_poses_umeyama(corrected[indices], captured['extrinsics'], ransac=False)
        evaluation_c2w = corrected_c2w.copy()
        evaluation_c2w[:, :3, :3] = R.T @ corrected_c2w[:, :3, :3]
        evaluation_c2w[:, :3, 3] = (corrected_c2w[:, :3, 3] - t) @ R / scale
        for suffix, selected in [('common_source', indices), ('heldout', targets)]:
            if not len(selected):
                continue
            np.savez_compressed(branch / f'{suffix}_cameras.npz', extrinsics=np.linalg.inv(evaluation_c2w[selected]),
                                intrinsics=all_K[selected], image_shape=np.asarray([
                                    data['intrinsics']['reference_image_size']['height'],
                                    data['intrinsics']['reference_image_size']['width']]),
                                image_names=np.asarray([all_files[i].name for i in selected]),
                                camera_frame='图像验证相机，仅source拟合Sim(3)后映射到adapter世界')
        gh.save_gaussian_ply(gaussians=pred.gaussians, save_path=str(branch / 'native.ply'),
                            ctx_depth=torch.from_numpy(pred.depth).unsqueeze(-1).to(pred.gaussians.means),
                            shift_and_scale=False, save_sh_dc_only=False, gs_views_interval=1,
                            inv_opacity=True, prune_by_depth_percent=1., prune_border_gs=False, match_3dgs_mcmc_dev=False)
        report['branches'][label] = {**cam._io.inspect_ply(branch / 'native.ply'), 'adapter_scale': captured['adapter_scale'],
                                     'scale_factor': float(pred.scale_factor), 'depth_shape': list(pred.depth.shape),
                                     'source_alignment': {'R': R.tolist(), 't': t.tolist(), 's': float(scale)}}
        cam._io.dump_json(out / 'summary.json', report)
        del pred
        torch.cuda.empty_cache()
        print(f'✅ {label}', flush=True)


if __name__ == '__main__':
    main()
