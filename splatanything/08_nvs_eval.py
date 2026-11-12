#!/usr/bin/env python3
"""08_nvs_eval.py — held-out NVS 评测（WB-2）：source 视角建云，留出视角只用于评分。

为什么必须做：E-08 已经证明自洽 PSNR 随输入视角数反相关（同一视角 22.4 → 19.0），
所以自洽分数连"哪一档更好"都答不了。唯一能给出质量结论的口径是留出视角。

坐标系怎么对上（这一步最容易错）：
  导出的云住在"source 首个相机的规范系 + GSN 缩放"里；留出视角的位姿在原始采集系里。
  官方 `build_views` 只做两件事：
    ① cam_align：`C' = inv(C_0) @ C`            → R_A = R_0ᵀ, t_A = -R_0ᵀ t_0
    ② GSN：`translation *= s`（只缩放平移，不动旋转）
  合起来 `C_exp = [R_A·R_o | s·(R_A·t_o + t_A)]`。等价地：
    R_exp = M·R_o ,  t_exp = s·(M·t_o) + u      （M = R_A，u = s·t_A）
  反过来用导出的 `cameras.json` 与原始 `prior_camera.json` **拟合** M / s / u（不假设实现细节），
  再用 source 视角反算残差自证；残差不为 0 就说明公式或数据配错了，直接报错退出。

GT 也必须过同一套预处理：`preprocess_view`（以主点为中心裁正方形 → rescale 到
resolution → 再精调 bbox）是纯函数，这里直接调官方的，不自己复刻。

Args:
  --result        02/06 的输出目录（含 gaussians.ply / cameras.json / rgb/）
  --source-scene  推理时真正喂进去的场景目录（含 prior_camera.json，视角数=source）
  --all-scene     含全部视角的场景目录（images/ + prior_camera.json），留出 = 全部 − source
  --resolution    W H，必须与推理时一致（默认 504 504）
  --out           输出目录，默认 RESULT/nvs
  --max-held-out  最多评几个留出视角，0 = 全部
  --eps2d         屏幕膨胀，与 03 一致（默认 0.01）
  --chunk         每次渲染几个视角（默认 4）
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
    spec = importlib.util.spec_from_file_location('sa_render_eval', HERE / '03_render_eval.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_cams_json(path):
    """{camera_id: (c2w 4x4, K 3x3)}，保序。"""
    data = json.loads(Path(path).read_text())
    ext = {e['camera_id']: np.asarray(e['matrix'], np.float64) for e in data['extrinsics']}
    itr = {e['camera_id']: np.asarray(e['matrix'], np.float64) for e in data['intrinsics']}
    ids = [e['camera_id'] for e in data['extrinsics'] if e['camera_id'] in itr]
    return ids, np.stack([ext[i] for i in ids]), np.stack([itr[i] for i in ids])


def fit_frame_transform(ids, c2w_exp, c2w_orig):
    """拟合 R_exp = M·R_o，t_exp = s·(M·t_o) + u；返回 (M, s, u) 与残差。

    不假设 `M = R_0ᵀ / u = -s·M·t_0` 这一具体形式，而是从数据解出来，再用残差自证。
    s 另有独立估计（成对相机距离比），两者必须一致 —— 这是最省事的"公式写错了"探针。
    """
    R_exp, t_exp = c2w_exp[:, :3, :3], c2w_exp[:, :3, 3]
    R_o, t_o = c2w_orig[:, :3, :3], c2w_orig[:, :3, 3]
    n = len(ids)

    # 旋转：每个视角都能独立解出一个 M，全部必须一致（否则不是同一个刚体变换）
    M_list = np.stack([R_exp[i] @ R_o[i].T for i in range(n)])
    M = M_list.mean(axis=0)
    # 投影回最近的旋转矩阵（平均后的矩阵可能轻微非正交）
    U, _, Vt = np.linalg.svd(M)
    M = U @ Vt
    if np.linalg.det(M) < 0:
        M = U @ np.diag([1, 1, -1]) @ Vt
    rot_spread = float(np.abs(M_list - M).max())
    rot_orth = float(np.abs(M.T @ M - np.eye(3)).max())

    # 平移：t_exp = s·(M t_o) + u，未知量 (s, u) 共 4 个，方程 3N 个
    # b = t_exp.reshape(-1) 按视角展平（view0 的 xyz、view1 的 xyz…），
    # 所以 u 的系数是 N 个 I3 竖着摞（[3N, 3]），不是 kron(I_N, I_3)（那是 [3N, 3N]，
    # 会把每个方程都放开，最小二乘退化成"每个点各自最优"，s 会算错）。
    a = (M @ t_o.T).T                                   # [N,3] = M t_o
    A = np.concatenate([a.reshape(-1, 1), np.tile(np.eye(3), (n, 1))], axis=1)   # [3N, 4]
    sol, *_ = np.linalg.lstsq(A, t_exp.reshape(-1), rcond=None)
    s, u = float(sol[0]), sol[1:4]
    t_recon = s * a + u
    trans_res = float(np.abs(t_recon - t_exp).max())

    # 独立估计：成对距离比只依赖尺度，与 u 无关
    pair_s = []
    for i in range(n):
        for j in range(i + 1, n):
            num = np.linalg.norm(t_exp[i] - t_exp[j])
            den = np.linalg.norm(M @ (t_o[i] - t_o[j]))
            if den > 1e-6:
                pair_s.append(num / den)
    pair_s = np.asarray(pair_s) if pair_s else np.asarray([np.nan])

    return dict(M=M, s=s, u=u, rot_spread=rot_spread, rot_orth=rot_orth,
                trans_res=trans_res, s_pair_median=float(np.nanmedian(pair_s)),
                s_pair_spread=float(np.nanmax(pair_s) - np.nanmin(pair_s)) if len(pair_s) else 0.0,
                n_views=n)


def apply_frame_transform(fit, c2w_orig):
    """把留出视角的原始 c2w 映射到导出系。"""
    M, s, u = fit['M'], fit['s'], fit['u']
    out = np.tile(np.eye(4), (len(c2w_orig), 1, 1))
    out[:, :3, :3] = M @ c2w_orig[:, :3, :3]
    out[:, :3, 3] = s * (M @ c2w_orig[:, :3, 3].T).T + u
    return out


def preprocess_gt(image_path, K, resolution):
    """调官方 preprocess_view，保证与推理时完全同一套裁剪/缩放。"""
    from PIL import Image
    from src.utils.scene_io import preprocess_view
    img_t, K2 = preprocess_view(Image.open(image_path), K, tuple(resolution))
    return img_t.permute(1, 2, 0).numpy().astype(np.float32), K2.numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True)
    ap.add_argument('--source-scene', required=True)
    ap.add_argument('--all-scene', required=True)
    ap.add_argument('--resolution', type=int, nargs=2, default=[504, 504])
    ap.add_argument('--out', default=None)
    ap.add_argument('--max-held-out', type=int, default=0)
    ap.add_argument('--eps2d', type=float, default=0.01)
    ap.add_argument('--chunk', type=int, default=4)
    ap.add_argument('--max-residual', type=float, default=1e-3,
                    help='source 反算残差阈值（导出位姿是 fp32 存的，留 1e-3 余量）')
    args = ap.parse_args()

    res_dir = Path(args.result).expanduser().resolve()
    src_scene = Path(args.source_scene).expanduser().resolve()
    all_scene = Path(args.all_scene).expanduser().resolve()
    out = Path(args.out) if args.out else res_dir / 'nvs'
    out.mkdir(parents=True, exist_ok=True)
    for p in (res_dir / 'gaussians.ply', res_dir / 'cameras.json'):
        if not p.exists():
            raise SystemExit(f'❌ 缺 {p}（先跑 02/06）')

    lib = _load_render_lib()

    # ---- 1. 位姿对齐 ----
    ids_src, c2w_exp, K_exp = load_cams_json(res_dir / 'cameras.json')
    ids_src_o, c2w_src_o, _ = load_cams_json(src_scene / 'prior_camera.json')
    if ids_src_o != ids_src:
        # 顺序可以不同，但集合必须一致
        if set(ids_src_o) != set(ids_src):
            raise SystemExit(f'❌ source 视角集合不一致：结果 {len(ids_src)} / 场景 {len(ids_src_o)}')
        idx = [ids_src_o.index(i) for i in ids_src]
        c2w_src_o = c2w_src_o[idx]

    fit = fit_frame_transform(ids_src, c2w_exp, c2w_src_o)
    print(f'🔗 坐标系对齐（{fit["n_views"]} 个 source 视角）')
    print(f'  M 的行列式 {np.linalg.det(fit["M"]):.6f}  正交误差 {fit["rot_orth"]:.2e}  '
          f'各视角 M 离散 {fit["rot_spread"]:.2e}')
    print(f'  尺度 s = {fit["s"]:.6f}；成对距离比中位 {fit["s_pair_median"]:.6f} '
          f'（离散 {fit["s_pair_spread"]:.2e}）')
    print(f'  平移残差 max {fit["trans_res"]:.3e}')
    if max(fit['rot_spread'], fit['trans_res']) > args.max_residual:
        raise SystemExit(f'❌ 对齐残差超阈值 {args.max_residual}：公式或相机配对有问题，'
                         '继续渲只会得到错位结果。')
    if abs(fit['s'] - fit['s_pair_median']) > 1e-3:
        raise SystemExit(f'❌ 两种尺度估计不一致（{fit["s"]:.6f} vs {fit["s_pair_median"]:.6f}），'
                         '说明平移模型形式不对')
    print('  ✅ 反算 source 位姿自检通过')

    # ---- 2. 留出视角 ----
    ids_all, c2w_all, K_all = load_cams_json(all_scene / 'prior_camera.json')
    src_set = set(ids_src)
    held = [(i, cid) for i, cid in enumerate(ids_all) if cid not in src_set]
    # 官方 examples/object 有相机条目但缺图（frame_0023），留出集合要按实际图片过滤
    no_img = [cid for _, cid in held if not (all_scene / 'images' / f'{cid}.png').exists()]
    if no_img:
        print(f'⚠️  留出集合里有 {len(no_img)} 个视角缺图片，已剔除：{no_img[:4]}')
        held = [(i, cid) for i, cid in held if cid not in set(no_img)]
    if not held:
        raise SystemExit('❌ 没有留出视角（source == all？）')
    if args.max_held_out > 0:
        held = held[:args.max_held_out]
    print(f'🎯 留出 {len(held)} / {len(ids_all)} 个视角')

    idx_h = [i for i, _ in held]
    c2w_h = apply_frame_transform(fit, c2w_all[idx_h])

    # ---- 3. GT 过官方预处理 ----
    gts, Ks = [], []
    for i in idx_h:
        img, K2 = preprocess_gt(all_scene / 'images' / f'{ids_all[i]}.png',
                                K_all[i], args.resolution)
        gts.append(img)
        Ks.append(K2)
    Ks = np.stack(Ks)
    h, w = gts[0].shape[:2]
    edge = np.concatenate([gts[0][0], gts[0][-1], gts[0][:, 0], gts[0][:, -1]], 0)
    bg = float(np.median(edge))
    print(f'  渲染 {w}×{h}，背景 {bg:.3f}，eps2d {args.eps2d}')

    # ---- 4. 渲染 + 打分 ----
    g = lib.load_ply(res_dir / 'gaussians.ply')
    print(f'  高斯 {len(g["means"]):,}')
    (out / 'render').mkdir(exist_ok=True)

    rows, pairs = [], []
    t0 = time.perf_counter()
    for start in range(0, len(idx_h), args.chunk):
        sl = slice(start, min(start + args.chunk, len(idx_h)))
        rgb, alpha = lib.render(g, c2w_h[sl], Ks[sl], w, h, eps2d=args.eps2d, background=bg)
        for j, k in enumerate(range(sl.start, sl.stop)):
            cid = ids_all[idx_h[k]]
            gt, rd = gts[k], rgb[j]
            al = alpha[j].reshape(h, w)
            fg = al > 0.5
            sh_gt, sh_rd = lib.sharpness(gt, fg), lib.sharpness(rd, fg)
            row = dict(id=cid, psnr=lib.psnr(gt, rd), ssim=lib.ssim(gt, rd),
                       psnr_fg=lib.psnr_region(gt, rd, fg),
                       psnr_bg=lib.psnr_region(gt, rd, al < 0.01),
                       alpha_mean=float(al.mean()),
                       sharp_ratio=(sh_rd / sh_gt) if sh_gt > 0 else None)
            rows.append(row)
            pairs.append(((gt * 255).astype(np.uint8), (rd * 255).astype(np.uint8)))
            from PIL import Image
            Image.fromarray((rd * 255).astype(np.uint8)).save(out / 'render' / f'{cid}.png')
            fg_txt = f'{row["psnr_fg"]:6.2f}' if row['psnr_fg'] is not None else '   n/a'
            print(f'  {cid:<20s} PSNR {row["psnr"]:6.2f} (前景 {fg_txt})  SSIM {row["ssim"]:.4f}  '
                  f'alpha {row["alpha_mean"]:.3f}  锐度 {100 * (row["sharp_ratio"] or 0):.0f}%')
    render_dt = time.perf_counter() - t0

    lib.contact_sheet(pairs, out / 'contact_sheet.png')

    def agg(key):
        vals = [r[key] for r in rows if r[key] is not None and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else None

    summary = dict(
        result=str(res_dir), source_scene=str(src_scene), all_scene=str(all_scene),
        n_source=len(ids_src), n_held_out=len(rows),
        resolution=[w, h], eps2d=args.eps2d, background=bg,
        gaussians=int(len(g['means'])),
        align=dict(s=fit['s'], s_pair_median=fit['s_pair_median'],
                   s_pair_spread=fit['s_pair_spread'],
                   rot_spread=fit['rot_spread'], rot_orth=fit['rot_orth'],
                   trans_res=fit['trans_res'],
                   M=fit['M'].tolist(), u=fit['u'].tolist()),
        metrics=dict(psnr=agg('psnr'), psnr_fg=agg('psnr_fg'), psnr_bg=agg('psnr_bg'),
                     ssim=agg('ssim'), alpha=agg('alpha_mean'),
                     sharp_ratio=agg('sharp_ratio'), render_seconds=round(render_dt, 1)),
        views_detail=rows,
    )
    (out / 'nvs_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    m = summary['metrics']
    print(f'\n📊 held-out NVS（{len(rows)} 视角，source {len(ids_src)}）：'
          f'PSNR {m["psnr"]:.2f}  SSIM {m["ssim"]:.4f}  alpha {m["alpha"]:.3f}  '
          f'锐度 {100 * (m["sharp_ratio"] or 0):.0f}%')
    print(f'  台账: {out / "nvs_summary.json"}')
    print(f'  拼图: {out / "contact_sheet.png"}')


if __name__ == '__main__':
    main()
