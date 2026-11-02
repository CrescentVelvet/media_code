#!/usr/bin/env python3
"""Step 1：扫体素合并参数（voxel_multiplier / opacity_threshold），看糊和重影能救回多少。

为什么值得扫：导出链路里唯一没被训练死的两个数就是它们（`gaussians.py:merge` 硬编码
1.4 与 0.01）。体素边长 = median_scale × multiplier，倍数越大 → 相邻不同深度的点被平均
成一个更大的高斯 → 边缘发散、看起来像重影。这是零成本、不用重训的唯一杠杆。

做法上刻意只跑**一次前向**：把 `infinisplat.model.merge` 换成捕获版，拿到合并前的 raw
splats（每像素一个），然后在本地按不同参数反复合并 + 渲染。前向 30 s，每次重新合并
几秒，比反复跑整条 pipeline 便宜一个量级。⚠️ 前提是本地合并与官方实现逐位一致——
脚本启动时用默认参数做一次回归校验（final_count 必须相等），不一致直接退出。

Env vars: 无（全走命令行）。

用法:
  python 10_sweep_merge.py --model ~/model/infinisplat_v2/da3 \
    --images-dir ~/output/infinisplat_v2_results/_captures/jiajia/images \
    --cameras    ~/output/infinisplat_v2_results/_captures/jiajia/cameras.npz \
    --out ~/output/infinisplat_v2_results/_sweep --tag jiajia \
    --voxels 1.4,1.0,0.7,0.5
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import ghost_metrics as gm


@torch.no_grad()
def merge_param(splats, voxel_multiplier=1.4, opacity_threshold=0.01):
    """官方 `gaussians.merge` 的参数化版本（逻辑逐行对齐，只把两个常数提出来）。

    改动点仅两处：`voxel = median * voxel_multiplier`、`opacities >= opacity_threshold`。
    其余（有限性过滤、floor 网格、scatter_add 求均值、四元数重归一）保持一致，
    否则扫出来的差异可能来自实现而非参数。
    """
    keep = torch.ones(len(splats["means"]), dtype=torch.bool, device=splats["means"].device)
    for key in ("means", "quats", "scales", "opacities", "sh"):
        v = splats[key]
        keep &= torch.isfinite(v.flatten(1) if v.ndim > 1 else v[:, None]).all(-1)
    keep &= (splats["scales"] > 0).all(-1) & (splats["quats"].norm(dim=-1) > 1e-8)
    if not keep.any():
        raise ValueError("No valid Gaussians")
    median = splats["scales"].float().log().mean(-1).exp()[keep].median()
    voxel = median * float(voxel_multiplier)
    s = {k: v[keep].float() for k, v in splats.items() if k != "base_rgb"}
    grid = (s["means"] / float(voxel)).floor().long()
    grid -= grid.min(0).values
    dims = grid.max(0).values + 1
    flat = grid[:, 0] * dims[1] * dims[2] + grid[:, 1] * dims[2] + grid[:, 2]
    unique, inverse = torch.unique(flat, return_inverse=True)
    counts = torch.zeros(len(unique), device=grid.device)
    counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))
    merged = {}
    for key, value in s.items():
        shape = value.shape[1:]
        columns = value.reshape(len(value), -1)
        result = columns.new_zeros((len(unique), columns.shape[1]))
        for column in range(columns.shape[1]):
            result[:, column].scatter_add_(0, inverse, columns[:, column])
        merged[key] = (result / counts[:, None]).reshape(len(unique), *shape)
    merged["quats"] /= merged["quats"].norm(dim=-1, keepdim=True).clamp_min(1e-8)
    visible = merged["opacities"] >= float(opacity_threshold)
    if not visible.any():
        raise ValueError("No Gaussians remain after opacity pruning")
    result = {k: v[visible] for k, v in merged.items()}
    stats = dict(raw_count=len(keep), valid_count=int(keep.sum()), merged_count=len(unique),
                 final_count=int(visible.sum()), median_scale=float(median),
                 voxel_size=float(voxel), voxel_multiplier=float(voxel_multiplier),
                 opacity_threshold=float(opacity_threshold))
    return result, stats


def to_device(splats, device):
    """一次性搬上卡 —— render_check.render 每帧都重传一遍，32 帧会白等十几秒。"""
    return {k: splats[k].detach().float().to(device)
            for k in ("means", "quats", "scales", "opacities", "sh")}


def render_view(g, c2w, K, w, h, device, eps2d=0.01, background=1.0):
    from gsplat import rasterization
    w2c = torch.from_numpy(np.linalg.inv(np.asarray(c2w, np.float64)).astype(np.float32))[None].to(device)
    Ks = torch.from_numpy(np.asarray(K, np.float32))[None].to(device)
    colors = g["sh"][None] if g["sh"].ndim == 3 else g["sh"][:, None, :][None]  # [N,1,3] → [1,N,1,3]
    bg = torch.full((1, 3), float(background), device=device)
    out, _, _ = rasterization(means=g["means"], quats=g["quats"], scales=g["scales"],
                              opacities=g["opacities"], colors=colors,
                              viewmats=w2c, Ks=Ks, width=w, height=h, sh_degree=0,
                              eps2d=eps2d, near_plane=0.01, far_plane=200.0,
                              backgrounds=bg, render_mode="RGB")
    return out[0].clamp(0, 1).cpu().numpy()


def save_ply(path, splats):
    """标准 3DGS PLY（字段顺序与官方 export.export_ply 一致），便于外部查看器/后续 finetune。"""
    from plyfile import PlyData, PlyElement
    n = len(splats["means"])
    fields = {}
    for i, name in enumerate(("x", "y", "z")):
        fields[name] = splats["means"][:, i].detach().float().cpu().numpy()
    for name in ("nx", "ny", "nz"):
        fields[name] = None
    for i in range(3):
        fields[f"f_dc_{i}"] = splats["sh"][:, 0, i].detach().float().cpu().numpy()
        fields[f"scale_{i}"] = splats["scales"][:, i].detach().float().cpu().numpy()
    op = splats["opacities"].detach().float().clamp(1e-6, 1 - 1e-6)
    fields["opacity"] = torch.logit(op).cpu().numpy()
    for i in range(3):
        fields[f"scale_{i}"] = np.log(np.maximum(fields[f"scale_{i}"], 1e-12))
    for i in range(4):
        fields[f"rot_{i}"] = splats["quats"][:, i].detach().float().cpu().numpy()
    data = np.empty(n, dtype=[(k, "<f4") for k in fields])
    for k, v in fields.items():
        data[k] = 0.0 if v is None else v
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--cameras", help="NPZ: c2w + intrinsics（给了走 provided）")
    ap.add_argument("--long-side", type=int, default=518)
    ap.add_argument("--max-views", type=int, default=32)
    ap.add_argument("--voxels", default="1.4,1.0,0.7,0.5", help="逗号分隔的 voxel_multiplier 候选")
    ap.add_argument("--opacity", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--save-ply", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in Path(args.images_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not imgs:
        raise SystemExit("❌ 没有图像")

    import infinisplat.model as M
    from infinisplat.pipeline import Pipeline

    pipe = Pipeline.from_pretrained(args.model, args.device)
    cap = {}
    orig_merge = M.merge

    def capture(raw):
        cap["raw"] = raw          # 只存引用，不复制（raw 在 merge 后即被丢弃）
        return orig_merge(raw)

    cams = None
    if args.cameras:
        with np.load(args.cameras, allow_pickle=False) as z:
            cams = {k: z[k] for k in ("c2w", "intrinsics")}
    M.merge = capture
    t0 = time.time()
    scene = pipe.reconstruct(imgs, cameras=cams,
                            pose_mode=None if cams is not None else "da3-pose",
                            long_side=args.long_side, max_views=args.max_views)
    M.merge = orig_merge
    print(f"⏱️  单次前向 {time.time() - t0:.1f}s | raw {scene['stats']['raw_count']:,}")

    raw = cap.get("raw")
    if raw is None:
        raise SystemExit("❌ 没捕获到 merge 的输入（官方实现可能改了调用路径）")
    # 回归校验：本地实现必须能逐位复现官方的默认结果
    with torch.inference_mode():
        _, s_default = merge_param(raw, 1.4, 0.01)
    if s_default["final_count"] != scene["stats"]["final_count"]:
        raise SystemExit(f"❌ 本地 merge 与官方不一致：{s_default['final_count']} vs "
                         f"{scene['stats']['final_count']}（先修实现，否则扫描结果不可信）")
    print(f"✅ 本地 merge 回归通过（final {s_default['final_count']:,} 与官方一致）")

    c2w = scene["c2w"].cpu().numpy()
    K = scene["intrinsics"].cpu().numpy()
    records = scene["metadata"]["inputs"]
    n = min(len(imgs), len(c2w), args.max_views)
    gts, sizes, base = [], [], []
    for i in range(n):
        w, h = (records[i].get("processed_wh") if i < len(records) else None) or (0, 0)
        gt = gm.load_gt_processed(imgs[i], int(w), int(h))
        gts.append(gt)
        sizes.append((int(w), int(h)))
        # 零假设底噪（GT 与自身比）与合并参数无关，扫之前算一次即可
        base.append(gm.edge_stats(gt, gt)[1])

    rows = []
    for vm in [float(x) for x in args.voxels.split(",") if x.strip()]:
        with torch.inference_mode():
            splats, st = merge_param(raw, vm, args.opacity)
        g = to_device(splats, args.device)
        del splats
        torch.cuda.empty_cache()

        sig, ps, spreads, dbl, rets, t0 = [], [], [], [], [], time.time()
        for i in range(n):
            w, h = sizes[i]
            rd = render_view(g, c2w[i], K[i], w, h, args.device)
            gt = gts[i]
            se, _, _ = gm.equivalent_blur_sigma(gt, rd)
            sp, dp, _, _ = gm.edge_stats(gt, rd)
            sig.append(se)
            ps.append(gm.psnr(gt, rd))
            spreads.append(sp)
            dbl.append(dp)
            rets.append(float(np.nanmean(gm.tile_retention(gt, rd))))
        dt = time.time() - t0
        row = dict(voxel_multiplier=vm, opacity_threshold=args.opacity,
                   gaussians=int(st["final_count"]), merged=int(st["merged_count"]),
                   voxel_size=st["voxel_size"],
                   psnr=float(np.mean(ps)), sigma_eff=float(np.median(sig)),
                   tile_ret=float(np.mean(rets)), spread_ratio=float(np.median(spreads)),
                   double_peak=float(np.mean(dbl)), double_peak_baseline=float(np.mean(base)),
                   render_s=round(dt, 2))
        rows.append(row)
        print(f"  voxel×{vm:<4} 高斯 {row['gaussians']:>9,}  PSNR {row['psnr']:6.3f}  "
              f"σ_eff {row['sigma_eff']:.1f}  锐度 {row['tile_ret']:.3f}  "
              f"扩散 {row['spread_ratio']:.2f}  双峰 {row['double_peak'] * 100:5.1f}%  ({dt:.0f}s)")
        if args.save_ply:
            save_ply(out / f"{args.tag}_vox{vm}.ply", g)
        del g
        torch.cuda.empty_cache()

    payload = dict(tag=args.tag, views=n, long_side=args.long_side,
                   raw_count=int(scene["stats"]["raw_count"]), rows=rows)
    with open(out / f"sweep_{args.tag}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ 输出: {out}/sweep_{args.tag}.json")


if __name__ == "__main__":
    main()
