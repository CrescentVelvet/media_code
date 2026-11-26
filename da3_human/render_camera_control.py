#!/usr/bin/env python3
"""用明确的PLY/相机对回放源或留出视角，保存逐帧图、PSNR及评测口径。

Env vars: PLY_FILE, CAMERAS_FILE, IMAGE_DIR, RESULTS_DIR, GSPLAT_CACHED_SO,
          RENDER_SIZE=504, LABEL=control, EVAL_ROLE=source|heldout
"""
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch


def main():
    if os.environ.get('GSPLAT_CACHED_SO'):
        spec = importlib.util.spec_from_file_location('gsplat_cuda', os.environ['GSPLAT_CACHED_SO'])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules['gsplat_cuda'] = mod
        sys.modules['gsplat.csrc'] = mod
        import gsplat
        gsplat.csrc = mod
    from gsplat import rasterization
    spec = importlib.util.spec_from_file_location('render', Path(__file__).with_name('35_render_hisilicon.py'))
    render = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render)
    out = Path(os.environ['RESULTS_DIR'])
    out.mkdir(parents=True, exist_ok=False)
    means, scales, quats, dc, sh, opacity = render.load_ply(Path(os.environ['PLY_FILE']))
    z = np.load(os.environ['CAMERAS_FILE'])
    cams = {k: z[k] for k in z.files}
    size = int(os.environ.get('RENDER_SIZE', '504'))
    K, base = render.load_intrinsics(cams, size, size)
    rows = []
    metrics = []
    role = os.environ.get('EVAL_ROLE', 'source')
    if role not in ('source', 'heldout'):
        raise ValueError(role)
    for i, name in enumerate(cams['image_names']):
        E = torch.eye(4, device='cuda')
        E[:3] = torch.tensor(cams['extrinsics'][i, :3], device='cuda', dtype=torch.float32)
        color, alpha, _ = rasterization(means=means, scales=scales.exp(),
            quats=torch.nn.functional.normalize(quats, dim=-1), opacities=opacity.sigmoid(), colors=sh,
            viewmats=E[None], Ks=torch.tensor(K[i:i+1], device='cuda'),
            width=size, height=size, packed=False, sh_degree=int(sh.shape[1] ** .5) - 1,
            eps2d=.01, backgrounds=torch.ones(1, 3, device='cuda'), render_mode='RGB')
        rgb = color[0].clamp(0, 1).cpu().numpy()
        source = Image.open(Path(os.environ['IMAGE_DIR']) / str(name)).convert('RGB').resize((size, size))
        gt = np.asarray(source) / 255.
        psnr = float(-10 * np.log10(np.mean((rgb - gt) ** 2).clip(1e-12)))
        frame = Image.fromarray((rgb * 255).astype('uint8'))
        frame.save(out / f'{i:02d}.png')
        row = Image.new('RGB', (size * 2, size + 26), 'white')
        row.paste(source, (0, 26))
        row.paste(frame, (size, 26))
        ImageDraw.Draw(row).text((5, 5), f"{os.environ.get('LABEL', 'control')} | {i:02d} input/render | {psnr:.2f} dB", fill='black')
        rows.append(row)
        metrics.append({'view': i, 'image_name': str(name), 'psnr_db': psnr, 'alpha_mean': float(alpha.mean())})
    sheet = Image.new('RGB', (size * 4, (size + 26) * ((len(rows) + 1)//2)), 'white')
    for i, row in enumerate(rows):
        sheet.paste(row, ((i % 2) * size * 2, (i // 2) * (size + 26)))
    sheet.save(out / 'contact.jpg', quality=92)
    report = {'ply': os.environ['PLY_FILE'], 'cameras': os.environ['CAMERAS_FILE'],
              'intrinsics_base_size': list(base), 'render_size': size,
              'mean_psnr_db': float(np.mean([m['psnr_db'] for m in metrics])), 'eval_role': role,
              'views': metrics, 'limits': ('各分支使用自身配套相机；源视角重放，不是独立新视角或几何真值指标。'
                                           if role == 'source' else '留出RGB不参与前向；仅source相机拟合Sim(3)，不代表独立几何真值。')}
    (out / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
