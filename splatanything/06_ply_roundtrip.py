#!/usr/bin/env python3
"""06_ply_roundtrip.py — 验证"导出的 gaussians.ply 就是模型吐的那批高斯"（WB-1）。

为什么这么做：03 的自洽 PSNR 评的是 **PLY 重载后** 的那批高斯。如果导出链（opacity
过滤 / logit 编码 / fp32 落盘 / scale log 往返）有错，后面所有指标都在评另一批数据，
而且错得越隐晦越难发现。

取数方式：**不复制官方 main()**，而是 monkeypatch `src.utils.scene_io.export_gaussians`
抓一份 preds 引用，再原样调 `inference.main()`。
  inference.py:130 在 main() **函数体内** `from src.utils.scene_io import export_gaussians`，
  所以 patch 模块属性在 main() 之前生效，代码路径与正式跑完全一致（无重复实现、无漂移）。

三组对比：
  A 内存全量（体素合并后、未按 opacity 过滤）—— 参考组，PLY 必然比它少点
  B 内存按官方同一条件过滤              —— 应当与 PLY **一致到 fp32 精度**
  C PLY 重载                            —— 03 实际评的对象

判据：B vs C 的
  * 属性 MAE（means / log scales / opacity / quats / f_dc）
  * 渲染 RGB MAE ≤ 1e-4、alpha MAE ≤ 1e-4（同相机、同背景）

Args:
  --input       场景目录（含 images/；pose 模式还需 prior_camera.json）
  --mode        rgb | pose（默认 pose）
  --resolution  W H（默认 504 504）
  --out         输出根目录（默认 $SPLATANYTHING_RESULTS_DIR/_roundtrip）
  --eps2d       渲染屏幕膨胀，与 03 保持一致（默认 0.01）
  --chunk       每次渲染几个视角（默认 4，控显存）
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get('SPLATANYTHING_DIR', Path.home() / 'repos/SplatAnything'))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def _load_render_lib():
    """03_render_eval.py 的模块名以数字开头，不能直接 import，走 spec 加载复用。"""
    spec = importlib.util.spec_from_file_location('sa_render_eval', HERE / '03_render_eval.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def capture_hook_factory(store):
    """替身 export_gaussians / merge_voxels：抓快照后原样调官方实现。

    preds['splats'] 是**合并之后**的，体素内样本的离散度只能用合并前的输入算，
    所以 merge_voxels 也必须包一层（在它入口拿 splats + keep）。
    注意 merge_voxels 挂在 `GaussianSplatRenderer`（model.gs_renderer）上，
    不在 GaussianWorldMirror，patch 要打到正确的类。
    """
    from src.utils import scene_io
    from src.models.models.rasterization import GaussianSplatRenderer

    original_export = scene_io.export_gaussians
    original_merge = GaussianSplatRenderer.merge_voxels

    def merge_hook(self, splats, keep):
        try:
            pre = splats['means'][0][keep[0]].detach().float().cpu().numpy()
            store['merge_stats'] = voxel_merge_stats(pre, float(self.voxel_size))
            store['voxel_size'] = float(self.voxel_size)
            store['spacing_target'] = float(self.spacing_target)
        except Exception as exc:                    # 诊断失败不能影响主链路
            store['merge_stats'] = dict(error=repr(exc))
        return original_merge(self, splats, keep)

    def export_hook(preds, outdir, opacity_thresh, scale_quantile):
        splats = preds['splats']
        # 只留对比需要的字段，立刻搬 CPU，避免和推理峰值显存叠加。
        store['splats'] = {
            k: splats[k][0].detach().float().cpu().numpy().copy()
            for k in ('means', 'quats', 'scales', 'opacities', 'sh')
        }
        store['c2w'] = preds['export_camera_poses'][0].detach().float().cpu().numpy().copy()
        store['K'] = preds['export_camera_intrs'][0].detach().float().cpu().numpy().copy()
        store['opacity_thresh'] = opacity_thresh
        store['scale_quantile'] = scale_quantile
        return original_export(preds, outdir, opacity_thresh, scale_quantile)

    scene_io.export_gaussians = export_hook
    GaussianSplatRenderer.merge_voxels = merge_hook
    return (original_export, original_merge)


def restore_hooks(originals):
    from src.utils import scene_io
    from src.models.models.rasterization import GaussianSplatRenderer
    original_export, original_merge = originals
    scene_io.export_gaussians = original_export
    GaussianSplatRenderer.merge_voxels = original_merge


def apply_export_filters(splats, opacity_thresh, scale_quantile):
    """复刻 scene_io.export_gaussians + _build_gs_ply_data 的两级过滤。

    ⚠️ 顺序与官方一致：先按 opacity 过滤（`if opacity_thresh` 为假值时**完全不过滤**，
    注意不是 `>=0`），再交给 save_gs_ply 做 scale 分位裁剪（quantile=None 时不过滤）。
    """
    dc = splats['sh'][:, 0, :]
    means, scales = splats['means'], splats['scales']
    quats, opac = splats['quats'], splats['opacities'].reshape(-1)
    kept = dict(n_before=len(means), opacity_filtered=False, scale_filtered=False)

    if opacity_thresh:
        keep = opac >= opacity_thresh
        if keep.any():
            means, scales, quats = means[keep], scales[keep], quats[keep]
            dc, opac = dc[keep], opac[keep]
            kept['opacity_filtered'] = True
    if scale_quantile is not None:
        thr = np.quantile(scales.max(axis=-1), scale_quantile)
        keep = scales.max(axis=-1) <= thr
        means, scales, quats = means[keep], scales[keep], quats[keep]
        dc, opac = dc[keep], opac[keep]
        kept['scale_filtered'] = True

    kept.update(n_after=len(means))
    return dict(means=means, scales=scales, quats=quats, opacities=opac, sh=dc), kept


def attr_mae(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(mae=float(d.mean()), max=float(d.max()))


def voxel_merge_stats(means, voxel_size=0.0018):
    """复刻 merge_voxels 的分组，量"一个体素里样本位置有多散"。

    这是"跨视角几何不一致"的直接读数：体素内 3D RMS 离散度 / voxel_size 越大，
    说明不同视角对同一块表面的预测越不一致，加权平均后越容易把细节摊平。
    （权重没参与统计，只衡量位置本身的分散程度。）
    """
    coords = np.floor(means / voxel_size).astype(np.int64)
    coords -= coords.min(axis=0)
    dims = coords.max(axis=0) + 1
    keys = (coords[:, 0] * dims[1] * dims[2] + coords[:, 1] * dims[2] + coords[:, 2]).astype(np.int64)
    order = np.argsort(keys, kind='stable')
    ks, ms = keys[order], means[order].astype(np.float64)
    _, start, counts = np.unique(ks, return_index=True, return_counts=True)

    n = counts.astype(np.float64)
    s1 = np.add.reduceat(ms, start, axis=0)
    s2 = np.add.reduceat(ms ** 2, start, axis=0)
    var = np.maximum(s2 / n[:, None] - (s1 / n[:, None]) ** 2, 0.0)
    rms = np.sqrt(var.sum(axis=1))                     # 每体素的位置 RMS 离散度

    multi = counts >= 2
    out = dict(
        n_samples=int(len(means)), n_voxels=int(len(counts)),
        sample_retention=float(len(counts) / len(means)),
        frac_voxels_multi=float(multi.mean()) if len(counts) else 0.0,
        max_samples_per_voxel=int(counts.max()) if len(counts) else 0,
        mean_samples_per_voxel=float(counts.mean()) if len(counts) else 0.0,
        spread_rms_mean_multi=float(rms[multi].mean()) if multi.any() else 0.0,
        spread_rms_median_multi=float(np.median(rms[multi])) if multi.any() else 0.0,
        spread_rms_over_voxel=float(np.median(rms[multi]) / voxel_size) if multi.any() else 0.0,
    )
    return out


def render_chunked(lib, g, c2w, K, w, h, bg, eps2d, chunk, tag):
    """分块渲染，返回 (rgb[C,H,W,3], alpha[C,H,W])。"""
    rgbs, alphas = [], []
    t0 = time.perf_counter()
    for s in range(0, len(c2w), chunk):
        sl = slice(s, min(s + chunk, len(c2w)))
        r, a = lib.render(g, c2w[sl], K[sl], w, h, eps2d=eps2d, background=bg)
        rgbs.append(r)
        alphas.append(a)
    dt = time.perf_counter() - t0
    print(f'    渲染 {tag}: {len(c2w)} 视角 {dt:.1f}s')
    return np.concatenate(rgbs, 0), np.concatenate(alphas, 0)


def io_audit(res_dir, scene):
    """逐文件核对 02 的产物：数量、形状、与输入视角数是否自洽。

    只查"能不能对上账"，不评画质。缺文件记 None 而不是抛错（NO_DEPTH_RGB=1 时
    depth/rgb 本来就没有）。
    """
    from plyfile import PlyData

    audit = {'scene': scene, 'files': {}}

    def rec(name, path, info):
        audit['files'][name] = dict(
            exists=path.exists(),
            bytes=path.stat().st_size if path.exists() else None, **info)

    gs = res_dir / 'gaussians.ply'
    if gs.exists():
        v = PlyData.read(str(gs))['vertex']
        rec('gaussians.ply', gs, dict(n=int(v.count),
                                      fields=sorted(v.data.dtype.names or ())))
    pts = res_dir / 'points.ply'
    if pts.exists():
        v = PlyData.read(str(pts))['vertex']
        rec('points.ply', pts, dict(n=int(v.count), fields=sorted(v.data.dtype.names or ())))
    cam = res_dir / 'cameras.json'
    if cam.exists():
        c = json.loads(cam.read_text())
        K = np.asarray(c['intrinsics'][0]['matrix'], np.float64) if c.get('intrinsics') else None
        rec('cameras.json', cam, dict(
            n_extrinsics=len(c.get('extrinsics', [])),
            n_intrinsics=len(c.get('intrinsics', [])),
            fx=float(K[0, 0]) if K is not None else None,
            fy=float(K[1, 1]) if K is not None else None,
            cx=float(K[0, 2]) if K is not None else None,
            cy=float(K[1, 2]) if K is not None else None))
    for sub in ('rgb', 'depth'):
        d = res_dir / sub
        rec(f'{sub}/', d, dict(n=len(list(d.glob('*.png'))) if d.is_dir() else 0))
    for name in ('config.yaml', 'run.json'):
        rec(name, res_dir / name, {})
    return audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--mode', default='pose', choices=['rgb', 'pose'])
    ap.add_argument('--resolution', type=int, nargs=2, default=[504, 504])
    ap.add_argument('--out', default=None)
    ap.add_argument('--eps2d', type=float, default=0.01)
    ap.add_argument('--chunk', type=int, default=4)
    ap.add_argument('--background', type=float, default=None)
    args = ap.parse_args()

    scene_dir = Path(args.input).expanduser().resolve()
    scene = scene_dir.name
    out_root = Path(args.out or os.environ.get(
        'SPLATANYTHING_RESULTS_DIR', Path.home() / 'output/splatanything_results') + '/_roundtrip')
    out_root.mkdir(parents=True, exist_ok=True)

    from inference import main as infer_main

    store = {}
    originals = capture_hook_factory(store)
    argv = ['--input_path', str(scene_dir), '--mode', args.mode,
            '--resolution', str(args.resolution[0]), str(args.resolution[1]),
            '--output_path', str(out_root)]
    t0 = time.perf_counter()
    try:
        infer_main(argv)
    finally:
        restore_hooks(originals)
    infer_dt = time.perf_counter() - t0

    if 'splats' not in store:
        raise SystemExit('❌ 没抓到 preds（export_gaussians 未被调用？检查 --no_depth_rgb / 过滤后是否为空）')

    res_dir = out_root / scene
    print(f'\n📦 导出现场: {res_dir}')

    # ---- 属性级对比 ----
    lib = _load_render_lib()
    mem = store['splats']
    mem_f, kept = apply_export_filters(mem, store['opacity_thresh'], store['scale_quantile'])
    ply = lib.load_ply(res_dir / 'gaussians.ply')
    print(f'  高斯数：内存全量 {kept["n_before"]:,} → 过滤后 {kept["n_after"]:,} → PLY {len(ply["means"]):,}')

    ms = store.get('merge_stats') or {}
    if 'error' not in ms and ms:
        print(f'  体素合并（pre-merge {ms["n_samples"]:,} 样本 → {ms["n_voxels"]:,} 体素，'
              f'voxel={store.get("voxel_size")}）：')
        print(f'    保留率 {100 * ms["sample_retention"]:.1f}%  '
              f'多样本体素占比 {100 * ms["frac_voxels_multi"]:.1f}%  '
              f'每体素均 {ms["mean_samples_per_voxel"]:.2f} / 最多 {ms["max_samples_per_voxel"]}')
        print(f'    体位素内位置 RMS 离散度：中位 {ms["spread_rms_median_multi"]:.3e} '
              f'（= {ms["spread_rms_over_voxel"]:.3f} × voxel_size）')
    elif ms:
        print(f'  ⚠️ 体素合并统计失败：{ms.get("error")}')

    if kept['n_after'] != len(ply['means']):
        raise SystemExit(f'❌ 数量不一致：复刻过滤得 {kept["n_after"]:,}，PLY 有 {len(ply["means"]):,}。'
                         '过滤条件没复刻对，后面的逐点对比无意义。')

    # opacity 在 PLY 里是 logit(clamp(op,1e-6,1-1e-6))，往返必然有损，单独统计饱和点数
    op_clamped = np.clip(mem_f['opacities'], 1e-6, 1 - 1e-6)
    saturated = int(np.sum((mem_f['opacities'] < 1e-6) | (mem_f['opacities'] > 1 - 1e-6)))
    attrs = {
        'means': attr_mae(mem_f['means'], ply['means']),
        'scales_rel': attr_mae(np.log(mem_f['scales']), np.log(ply['scales'])),
        'opacities': attr_mae(op_clamped, ply['opacities']),
        'quats': attr_mae(mem_f['quats'], ply['quats']),
        'sh_dc': attr_mae(mem_f['sh'], ply['sh']),
    }
    print('  属性 MAE（B 内存过滤后 vs C PLY）：')
    for k, v in attrs.items():
        print(f'    {k:<12s} mae {v["mae"]:.3e}  max {v["max"]:.3e}')
    print(f'    （opacity 触顶饱和 {saturated:,} / {kept["n_after"]:,}）')

    # ---- 相机一致性 ----
    c2w_ply, K_ply, ids = lib.load_cameras(res_dir / 'cameras.json')
    cam_drift = float(np.abs(c2w_ply - store['c2w'].astype(np.float64)).max())
    intr_drift = float(np.abs(K_ply - store['K']).max())
    print(f'  相机漂移：extrinsics {cam_drift:.3e}  intrinsics {intr_drift:.3e}')

    # ---- 渲染级对比 ----
    from PIL import Image
    rgb_dir = res_dir / 'rgb'
    w, h = args.resolution
    if (rgb_dir / f'{ids[0]}.png').exists():
        with Image.open(rgb_dir / f'{ids[0]}.png') as im:
            w, h = im.size
        first = np.asarray(Image.open(rgb_dir / f'{ids[0]}.png').convert('RGB'), np.float32) / 255.0
        edge = np.concatenate([first[0], first[-1], first[:, 0], first[:, -1]], 0)
        bg = float(np.median(edge)) if args.background is None else float(args.background)
    else:
        bg = 0.0 if args.background is None else float(args.background)
    print(f'  渲染 {w}×{h}，背景 {bg:.3f}，eps2d {args.eps2d}')

    g_mem_all = dict(means=mem['means'], quats=mem['quats'], scales=mem['scales'],
                     opacities=mem['opacities'].reshape(-1), sh=mem['sh'][:, 0, :])
    g_mem_f = dict(means=mem_f['means'], quats=mem_f['quats'], scales=mem_f['scales'],
                   opacities=mem_f['opacities'], sh=mem_f['sh'])

    print('  渲染 A（内存全量）…')
    rgb_a, al_a = render_chunked(lib, g_mem_all, c2w_ply, K_ply, w, h, bg, args.eps2d, args.chunk, 'A')
    print('  渲染 B（内存过滤后）…')
    rgb_b, al_b = render_chunked(lib, g_mem_f, c2w_ply, K_ply, w, h, bg, args.eps2d, args.chunk, 'B')
    print('  渲染 C（PLY 重载）…')
    rgb_c, al_c = render_chunked(lib, ply, c2w_ply, K_ply, w, h, bg, args.eps2d, args.chunk, 'C')

    metrics = {}
    for tag, (r, a) in (('A_full', (rgb_a, al_a)), ('B_filtered', (rgb_b, al_b))):
        diff = np.abs(r - rgb_c)
        p = float(lib.psnr(rgb_c, r))
        metrics[tag] = dict(
            rgb_mae=float(diff.mean()), rgb_max=float(diff.max()),
            # RGB 逐位相同时 PSNR 是 inf，json 里写成 null（Infinity 不是合法 JSON）
            psnr=None if not np.isfinite(p) else p,
            psnr_inf=not np.isfinite(p),
            alpha_mae=float(np.abs(a - al_c).mean()),
            alpha_max=float(np.abs(a - al_c).max()),
        )
    m_b = metrics['B_filtered']
    psnr_txt = 'inf（逐位相同）' if m_b['psnr_inf'] else f'{m_b["psnr"]:.2f}'
    print('\n  B vs C（等价性判据）: '
          f'RGB MAE {m_b["rgb_mae"]:.3e}  alpha MAE {m_b["alpha_mae"]:.3e}  PSNR {psnr_txt}')
    print('  A vs C（过滤影响，参考）: '
          f'RGB MAE {metrics["A_full"]["rgb_mae"]:.3e}  alpha MAE {metrics["A_full"]["alpha_mae"]:.3e}')

    ok = metrics['B_filtered']['rgb_mae'] <= 1e-4 and metrics['B_filtered']['alpha_mae'] <= 1e-4

    # ---- 差异图 ----
    def amp(x):
        return (np.clip(x * 1000.0, 0, 1) * 255).astype(np.uint8)

    tiles = []
    for i in range(min(3, len(rgb_c))):
        # gsplat 的 alpha 是 [H,W,1]，先压成 [H,W] 再拼，否则维度对不上
        ad = np.abs(al_b[i].astype(np.float64) - al_c[i].astype(np.float64)).reshape(al_c.shape[1], al_c.shape[2])
        tiles.append(np.concatenate([
            (rgb_c[i] * 255).astype(np.uint8),
            amp(np.abs(rgb_b[i] - rgb_c[i])),
            amp(np.repeat(ad[..., None], 3, axis=2)),
        ], axis=1))
    if tiles:
        Image.fromarray(np.concatenate(tiles, 0)).save(res_dir / 'roundtrip_diff.png')

    summary = dict(
        scene=scene, input=str(scene_dir), result_dir=str(res_dir),
        resolution=[w, h], eps2d=args.eps2d, background=bg,
        io_audit=io_audit(res_dir, scene),
        gaussians=dict(in_memory=kept['n_before'], after_filter=kept['n_after'],
                       in_ply=len(ply['means']), **{k: v for k, v in kept.items()
                                                    if k.startswith(('opacity', 'scale'))}),
        opacity_thresh=store['opacity_thresh'], scale_quantile=store['scale_quantile'],
        opacity_saturated=saturated,
        voxel_merge=ms, voxel_size=store.get('voxel_size'),
        spacing_target=store.get('spacing_target'),
        attributes=attrs,
        camera_drift=dict(extrinsics=cam_drift, intrinsics=intr_drift),
        render=metrics, pass_1e4=ok, infer_seconds=round(infer_dt, 1),
    )
    (res_dir / 'ply_roundtrip.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print('\n  io_audit（产物逐文件核对）:')
    for name, info in summary['io_audit']['files'].items():
        detail = '  '.join(f'{k}={v}' for k, v in info.items()
                           if k not in ('exists', 'bytes', 'fields'))
        print(f'    {"✓" if info["exists"] else "✗"} {name:<16s} {detail}')

    print(f'\n{"🎉 PASS" if ok else "❌ FAIL"}：B vs C RGB MAE {m_b["rgb_mae"]:.3e} '
          f'≤ 1e-4 → {ok}')
    print(f'  台账: {res_dir / "ply_roundtrip.json"}')
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
