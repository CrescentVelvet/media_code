#!/usr/bin/env python3
"""Step 3：冻结位置的高斯后处理精调（3DGS photometric finetune）。

只回答一个问题：**给定 InfiniSplat 已经导出的这堆高斯，纯光度优化还能榨出多少？**

三条纪律（都是前几轮踩出来的，别改）：
  1) **位置冻结**。means 全程 requires_grad=False，既不做 densify 也不做 prune。
     Step 1 已证明点密度就是一像素一个，加密度只会更糊；位置一动还会把
     「多视角深度不一致」的错位放大成真实几何误差。
  2) **位姿绝不联合优化**。相机是 GT（provided 模式），动了只会把弱纹理区带崩
     （som_human B 案：COLMAP 好初值上联合优化 −8.29 dB）。
  3) **必须同时报自洽与留出**。自洽 PSNR 会骗人（E-10：物件自洽 21.5 / 真 NVS 13.1）。
     这里 `--holdout-every K` 能把视角切成训练/留出两组，留出那组只用于报数。

可精调的量（默认全开，可单独关掉做消融）：
  sh(N,3,DC) / scale(N,3,log) / opacity(N,logit) / quat(N,4)

指标口径与 render_check.py 完全同源（直接 import 它的 model_preprocess / psnr），
避免又出现「渲染放大去凑 GT」那类注入式偏差（E-26）。

用法:
  python 14_finetune_gs.py --result <result_dir> --images-dir <imgs> \
      --out <out_dir> --iters 3000
  python 14_finetune_gs.py ... --iters 0        # 干跑，验证本脚本渲染与 render_check 等价

输出:
  <out>/scene.ply          精调后的高斯（字段与官方 export 一致）
  <out>/result.json        result.json 副本（供 render_check / 09 直接复用）
  <out>/cameras.npz        cameras.npz 副本
  <out>/finetune.json      超参、逐次 eval 曲线、耗时、峰值显存
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement
from gsplat import rasterization

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_check import load_ply, model_preprocess, psnr, render as rc_render  # noqa: E402


def ssim_fn(x, y, win=11, sigma=1.5):
    """可微 SSIM 的窗口核（与 render_check.ssim 同参数，只换成 batched tensor）。"""
    ch = x.shape[1]
    c = torch.arange(win, device=x.device, dtype=x.dtype) - win // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(-1, 1)
    k = (g @ g.t()).expand(ch, 1, win, win).contiguous()
    pad = win // 2

    def conv(t):
        return F.conv2d(t, k, padding=pad, groups=ch)

    mu_x, mu_y = conv(x), conv(y)
    sxx = conv(x * x) - mu_x ** 2
    syy = conv(y * y) - mu_y ** 2
    sxy = conv(x * y) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.mean()


def write_ply(path, means, scales, opacities, quats, sh):
    """字段与官方 export_ply 对齐，保证下游 render_check / 09 / 3DGS 工具能直接读。

    ⚠️ PLY 的 `opacity` 字段存的是 **logit**、`scale_i` 存的是 **log**（3DGS 约定）。
    这里入参都是物理量（概率 / 长度），转换必须在本函数里做——早期版本漏了 opacity 的
    logit，回读时被再 sigmoid 一次（0.283→0.568），渲染 PSNR 直接掉 8.6 dB。
    """
    n = len(means)
    op_logit = np.log(np.clip(opacities, 1e-6, 1 - 1e-6) /
                      (1.0 - np.clip(opacities, 1e-6, 1 - 1e-6)))
    fields = {}
    for i, name in enumerate(("x", "y", "z")):
        fields[name] = means[:, i]
    for name in ("nx", "ny", "nz"):
        fields[name] = np.zeros(n, np.float32)
    for i in range(3):
        fields[f"f_dc_{i}"] = sh[:, i]
        fields[f"scale_{i}"] = scales[:, i]
    fields["opacity"] = op_logit
    for i in range(4):
        fields[f"rot_{i}"] = quats[:, i]
    data = np.empty(n, dtype=[(k, "<f4") for k in fields])
    for k, v in fields.items():
        data[k] = v.astype(np.float32)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="含 scene.ply / cameras.npz / result.json")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=3000, help="0 = 干跑，只做 eval")
    ap.add_argument("--max-views", type=int, default=64)
    ap.add_argument("--holdout-every", type=int, default=0,
                    help="K>0 时把 idx%%K==K-1 的视角留出（只报数，不参与训练）")
    ap.add_argument("--opt", default="sh,scale,opacity,quat",
                    help="逗号分隔，可只开部分做消融")
    ap.add_argument("--lr-sh", type=float, default=2.5e-3)
    ap.add_argument("--lr-scale", type=float, default=5e-3)
    ap.add_argument("--lr-opacity", type=float, default=5e-2)
    ap.add_argument("--lr-quat", type=float, default=1e-3)
    ap.add_argument("--ssim-weight", type=float, default=0.2)
    ap.add_argument("--lr-final", type=float, default=0.05, help="末期学习率相对初始的倍率")
    ap.add_argument("--scale-init-mult", type=float, default=1.0, help="精调前先把半径乘这个系数")
    ap.add_argument("--eps2d", type=float, default=0.01, help="论文值 1e-2")
    ap.add_argument("--background", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    res, out = Path(args.result), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((res / "result.json").read_text())
    g = load_ply(res / "scene.ply")
    cams = np.load(res / "cameras.npz", allow_pickle=False)
    c2w_all, K_all = np.array(cams["c2w"]), np.array(cams["intrinsics"]).astype(np.float32)
    records = meta.get("inputs", [])
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    imgs = sorted(p for p in Path(args.images_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in exts)
    n = min(len(c2w_all), args.max_views, len(imgs))
    print(f"🚀 [14] 精调 | 高斯 {len(g['means']):,} | 视角 {n} | iters {args.iters}")

    # ---- 逐视角 GT（与 render_check 同一预处理，避免裁剪偏移）----
    gts, whs = [], []
    for i in range(n):
        rec = records[i] if i < len(records) else {}
        w, h = rec.get("processed_wh", [0, 0])
        if not (w and h):
            with Image.open(imgs[i]) as im:
                w, h = im.size
        w, h = int(w), int(h)
        arr, _ = model_preprocess(imgs[i], w, h)
        gts.append(arr)
        whs.append((w, h))
    w, h = whs[0]
    if len(set(whs)) != 1:
        raise SystemExit(f"❌ 视角尺寸不一致 {set(whs)}")
    gt_t = torch.from_numpy(np.stack(gts).astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(dev)

    # ---- 相机（w2c / K）全部搬到 GPU，训练期间只读 ----
    viewmats = torch.from_numpy(np.linalg.inv(c2w_all[:n]).astype(np.float32)).to(dev)
    Ks = torch.from_numpy(K_all[:n]).to(dev)
    bg = torch.full((1, 3), float(args.background), device=dev)

    # ---- 参数（全部在 PLY 的原始空间里优化，渲染时再变换）----
    means = torch.from_numpy(g["means"]).to(dev)
    scales0 = np.log(np.maximum(g["scales"] * args.scale_init_mult, 1e-8)).astype(np.float32)
    opac0 = np.log(np.clip(g["opacities"], 1e-6, 1 - 1e-6) /
                   (1 - np.clip(g["opacities"], 1e-6, 1 - 1e-6))).astype(np.float32)
    quat0 = g["quats"].astype(np.float32)
    sh0 = g["sh"].astype(np.float32)

    p = {k: torch.nn.Parameter(torch.from_numpy(v).to(dev)) for k, v in
         dict(sh=sh0, scale=scales0, opacity=opac0, quat=quat0).items()}
    opt_set = {s.strip() for s in args.opt.split(",") if s.strip()}
    for k in p:
        p[k].requires_grad_(k in opt_set)
    print(f"   可优化组: {sorted(k for k in p if p[k].requires_grad)}")
    sc_sorted = np.percentile(g["scales"], [1, 50, 99])
    print(f"   初始半径分位 1/50/99%: {sc_sorted.round(5).tolist()}（世界单位，RDNorm 半径=1）")

    groups = [{"params": [p[k]], "lr": {"sh": args.lr_sh, "scale": args.lr_scale,
                                        "opacity": args.lr_opacity, "quat": args.lr_quat}[k]}
              for k in p if p[k].requires_grad]
    opt = torch.optim.Adam(groups, eps=1e-15)
    base_lrs = [gp["lr"] for gp in opt.param_groups]

    # 防爆：尺度别跨 3 个数量级，不透明度别到硬 0/1
    sc_lo, sc_hi = np.percentile(scales0, 0.1) - 3.0, np.percentile(scales0, 99.9) + 2.0

    def render(idx):
        C = len(idx)
        colors = p["sh"][None, :, None, :].expand(C, -1, -1, -1).contiguous()
        out_rgb, alpha, _ = rasterization(
            means=means, quats=F.normalize(p["quat"], dim=-1),
            scales=torch.exp(p["scale"]), opacities=torch.sigmoid(p["opacity"]),
            colors=colors, viewmats=viewmats[idx], Ks=Ks[idx], width=w, height=h,
            sh_degree=0, eps2d=args.eps2d, near_plane=0.01, far_plane=200.0,
            backgrounds=bg.expand(C, 3), render_mode="RGB")
        return out_rgb.clamp(0, 1), alpha

    def evaluate(idx, tag="", chunk=8):
        """分块渲染：32 视角一次性渲染会吃掉 20 GB（实测），分块后只有 1/4。"""
        if len(idx) == 0:
            return None
        ps, als = [], []
        with torch.no_grad():
            for s in range(0, len(idx), chunk):
                part = idx[s:s + chunk]
                im, al = render(part)          # gsplat 返回 [c,H,W,3]
                als.append(float(al.mean()))
                ps += [psnr(gt_t[i].permute(1, 2, 0).cpu().numpy(),
                            im[k].cpu().numpy()) for k, i in enumerate(part)]
        if tag:
            print(f"   [{tag}] PSNR {np.mean(ps):6.3f}  alpha {np.mean(als):.3f}")
        return dict(psnr=float(np.mean(ps)), alpha=float(np.mean(als)))

    hold = [i for i in range(n) if args.holdout_every and i % args.holdout_every == args.holdout_every - 1]
    train = [i for i in range(n) if i not in set(hold)]
    print(f"   训练 {len(train)} 视角 / 留出 {len(hold)} 视角")

    history = []
    base_train = evaluate(train, "init/train")
    base_hold = evaluate(hold, "init/hold") if hold else None
    history.append(dict(iter=0, train=base_train, hold=base_hold))
    t0 = time.time()

    for it in range(1, args.iters + 1):
        frac = it / max(1, args.iters)
        for gp, lr0 in zip(opt.param_groups, base_lrs):
            gp["lr"] = lr0 * (args.lr_final ** frac)
        idx = [train[np.random.randint(len(train))]]
        im, _ = render(idx)              # [1,H,W,3]
        im = im.permute(0, 3, 1, 2)      # → [1,3,H,W]，与 gt_t 同布局
        gt = gt_t[idx]
        l1 = (im - gt).abs().mean()
        loss = (1 - args.ssim_weight) * l1 + args.ssim_weight * (1 - ssim_fn(im, gt))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            p["scale"].clamp_(sc_lo, sc_hi)
            p["opacity"].clamp_(-12.0, 12.0)
            p["quat"].clamp_(-4.0, 4.0)
        if it % 100 == 0 or it == 1:
            print(f"  it {it:5d}/{args.iters}  loss {loss.item():.5f}  l1 {l1.item():.5f}"
                  f"  {(time.time() - t0) / it:.3f}s/it")
        if args.eval_every and it % args.eval_every == 0:
            history.append(dict(iter=it, train=evaluate(train, f"it{it}/train"),
                                hold=evaluate(hold, f"it{it}/hold") if hold else None))

    dt = time.time() - t0
    if args.iters:
        final_train = evaluate(train, "final/train")
        final_hold = evaluate(hold, "final/hold") if hold else None
        history.append(dict(iter=args.iters, train=final_train, hold=final_hold))
    else:
        final_train, final_hold = base_train, base_hold

    # ---- 落盘 ----
    with torch.no_grad():
        sc = torch.exp(p["scale"]).cpu().numpy()
        op = torch.sigmoid(p["opacity"]).cpu().numpy()
        qt = F.normalize(p["quat"], dim=-1).cpu().numpy()
        sh = p["sh"].cpu().numpy()
    write_ply(out / "scene.ply", g["means"], np.log(np.maximum(sc, 1e-8)), op, qt, sh)
    shutil.copy2(res / "cameras.npz", out / "cameras.npz")
    shutil.copy2(res / "result.json", out / "result.json")

    # ---- 回读自检：写盘 → 用 render_check 的读取+渲染路径重渲一个视角 ----
    # PY 的 opacity/scale 字段是 logit/log 分位，写错过一次就是 8.6 dB 的静默损失，
    # 所以这里必须对拍内存值，不能只看"文件写成功了"。
    i0 = train[0]
    gt0 = gt_t[i0].permute(1, 2, 0).cpu().numpy()
    with torch.no_grad():
        im0, _ = render([i0])
    ps_mem = psnr(gt0, im0[0].cpu().numpy())
    gl = load_ply(out / "scene.ply")
    rd0, _ = rc_render(gl, c2w_all[i0], K_all[i0], w, h, eps2d=args.eps2d,
                       background=args.background, device=dev)
    ps_ply = psnr(gt0, rd0[0])
    rt_delta = ps_ply - ps_mem
    print(f"\n🔁 回读写盘自检: 内存 {ps_mem:.3f} dB / PLY 回读 {ps_ply:.3f} dB  Δ{rt_delta:+.3f}")
    if abs(rt_delta) > 0.05:
        print("   ⚠️ 量级不符 —— 大概率是 opacity(logit)/scale(log) 字段写错，结论不可信")

    delta_sc = float(np.mean(np.abs(np.log(np.maximum(sc, 1e-9)) - np.log(g["scales"]))))
    delta_sh = float(np.mean(np.abs(sh - g["sh"])))
    rep = dict(result=str(res), images_dir=str(args.images_dir), iters=args.iters,
               opt=sorted(opt_set), holdout_every=args.holdout_every,
               n_train=len(train), n_hold=len(hold), wh=[w, h],
               gaussians=int(len(g["means"])),
               init=dict(train=base_train, hold=base_hold),
               final=dict(train=final_train, hold=final_hold),
               history=history,
               scale_log_delta_mean=round(delta_sc, 4),
               sh_abs_delta_mean=round(delta_sh, 5),
               opacity_mean_init=round(float(g["opacities"].mean()), 4),
               opacity_mean=round(float(op.mean()), 4),
               ply_roundtrip=dict(mem=round(ps_mem, 3), ply=round(ps_ply, 3),
                                  delta=round(rt_delta, 3)),
               seconds=round(dt, 1),
               sec_per_iter=round(dt / max(1, args.iters), 4),
               peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    (out / "finetune.json").write_text(json.dumps(rep, indent=2))
    print(f"\n📊 init → final  训练集 PSNR {base_train['psnr']:.3f} → {final_train['psnr']:.3f}"
          f"  ({(final_train['psnr'] - base_train['psnr']):+.3f} dB)")
    if base_hold:
        print(f"              留出集 PSNR {base_hold['psnr']:.3f} → {final_hold['psnr']:.3f}"
              f"  ({(final_hold['psnr'] - base_hold['psnr']):+.3f} dB)")
    print(f"   半径对数均值变化 {delta_sc:.4f}（负=整体变小）  不透明度均值 {op.mean():.4f}")
    print(f"   {dt:.1f}s / {dt / max(1, args.iters) * 1000:.0f}ms per it  峰值显存 {rep['peak_vram_gb']:.2f} GB")
    print(f"✅ 输出: {out}/scene.ply, finetune.json")


if __name__ == "__main__":
    main()
