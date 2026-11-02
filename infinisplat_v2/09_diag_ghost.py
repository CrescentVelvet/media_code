#!/usr/bin/env python3
"""Step 0：糊 / 重影归因（纯 CPU，读已有渲染，不占 GPU）。

对每个视角算三组指标（定义见 ghost_metrics.py）：
  - sigma_eff   ：渲染等价于把 GT 模糊几像素 —— 糊的绝对量
  - tile_ret    ：4×4 分块高频保留率 —— 糊在哪
  - width_ratio / double_peak_ratio —— 边缘是变宽（发散）还是裂成两条（真重影）

判读（写进 EXPERIMENTS，别凭直觉）：
  - spread_ratio >> 1 且 double_peak_ratio ≈ 0  → 单向发散，糊来自大高斯/合并平均
  - double_peak_ratio 显著 > 0                 → 边缘分裂，重影来自两态几何冲突
  - 两者都小但 sigma_eff 大                     → 问题不在边缘，是整体颜色/低频偏

⚠️ 指标口径不确定时先跑 `--selftest`：用「已知答案」的合成对照（模糊 / 人工鬼影）
验证灵敏度与特异性。第一版 edge_stats 只数局部极大，在真数据上误报 73% 双峰，
就是没做自检的后果。

Env vars: 无（全走命令行）。

用法:
  python 09_diag_ghost.py --result ~/output/.../jiajia \
      --images-dir ~/output/.../_captures/jiajia/images --out /tmp/diag --tag jiajia
  python 09_diag_ghost.py --result ... --images-dir ... --out ... --selftest
"""
import argparse
import json
from pathlib import Path

import numpy as np

import ghost_metrics as gm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="含 result.json 与 eval/render_*.png")
    ap.add_argument("--images-dir", required=True, help="输入原图目录（与 result.json inputs 同序）")
    ap.add_argument("--render-dir", help="默认 <result>/eval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="ghost")
    ap.add_argument("--grid", type=int, default=4, help="分块数（越大越细，块内样本越少）")
    ap.add_argument("--max-views", type=int, default=32)
    ap.add_argument("--selftest", action="store_true", help="先在视角 0 上跑合成对照，验证指标灵敏度")
    args = ap.parse_args()

    res, out = Path(args.result), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((res / "result.json").read_text())
    records = meta.get("inputs", [])
    rdir = Path(args.render_dir) if args.render_dir else res / "eval"

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    imgs = sorted(p for p in Path(args.images_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in exts)
    renders = sorted(rdir.glob("render_*.png"))
    n = min(len(imgs), len(renders), args.max_views)
    if n == 0:
        raise SystemExit(f"❌ 找不到配对：图 {len(imgs)} / 渲染 {len(renders)}（目录 {rdir}）")
    print(f"🚀 [09] 糊/重影归因 | {n} 视角 | 渲染 {rdir}")

    from PIL import Image

    # 指标自检：真数据上的「假阳性底噪」= GT 与自身比；再验两种已知答案的对照
    if args.selftest:
        w0, h0 = records[0].get("processed_wh", [0, 0])
        gt0 = gm.load_gt_processed(imgs[0], int(w0), int(h0))
        print("\n🔬 指标自检（对照的已知答案）")
        print("   期望：blur 只抬 spread、ghost 明显抬双峰、GT 自身给出假阳性底噪")
        for name, tgt in [("GT自身", gt0)] + list(gm.synthetic_controls(gt0).items()):
            sp, dp, ne, _ = gm.edge_stats(gt0, tgt)
            sig, _, _ = gm.equivalent_blur_sigma(gt0, tgt)
            print(f"   {name:6s} σ_eff {sig:4.1f}  spread×{sp:4.2f}  双峰 {dp * 100:5.1f}%  n={ne}")
        print()

    rows, mats = [], []
    for i in range(n):
        w, h = records[i].get("processed_wh", [0, 0]) if i < len(records) else (0, 0)
        if not (w and h):
            with Image.open(imgs[i]) as im:
                w, h = im.size
        w, h = int(w), int(h)
        gt = gm.load_gt_processed(imgs[i], w, h)
        rd = np.asarray(Image.open(renders[i]).convert("RGB"), np.float32) / 255.0
        if rd.shape != gt.shape:
            raise SystemExit(f"❌ 视角 {i} 尺寸不一致 GT{gt.shape} vs render{rd.shape}")

        sig, p_at_best, curve = gm.equivalent_blur_sigma(gt, rd)
        mat = gm.tile_retention(gt, rd, args.grid)
        wr, dpr, ne, ex = gm.edge_stats(gt, rd)
        # 零假设：GT 与自身比。干净单峰筛选后应≈0，实测值即该视角的假阳性底噪
        _, dp_base, _, _ = gm.edge_stats(gt, gt)
        rows.append(dict(view=i, image=imgs[i].name, psnr=gm.psnr(gt, rd),
                         sigma_eff=sig, psnr_at_sigma=p_at_best,
                         psnr_gain_from_blur=p_at_best - gm.psnr(gt, rd),
                         tile_ret=[[float(x) for x in r] for r in mat],
                         tile_ret_mean=float(np.nanmean(mat)),
                         spread_ratio=wr, double_peak_ratio=dpr,
                         double_peak_baseline=dp_base,
                         double_peak_net=dpr - dp_base, n_edges=ne,
                         double_peak_examples=ex))
        mats.append((gt, rd, mat))
        print(f"  [{i:02d}] PSNR {gm.psnr(gt, rd):6.3f}  σ_eff {sig:4.1f}px  "
              f"锐度保留 {np.nanmean(mat):.3f}  扩散×{wr:.2f}  "
              f"双峰 {dpr * 100:5.1f}%（底噪 {dp_base * 100:4.1f}%）  n={ne}")

    def agg(k):
        vals = [r[k] for r in rows if np.isfinite(r[k])]
        return float(np.median(vals)) if vals else float("nan")

    summary = dict(tag=args.tag, views=n, gaussians=int(meta["stats"]["final_count"]),
                   grid=args.grid,
                   psnr_mean=float(np.mean([r["psnr"] for r in rows])),
                   sigma_eff_median=agg("sigma_eff"),
                   psnr_gain_from_blur_median=agg("psnr_gain_from_blur"),
                   tile_ret_mean=agg("tile_ret_mean"),
                   spread_ratio_median=agg("spread_ratio"),
                   double_peak_ratio_mean=float(np.mean([r["double_peak_ratio"] for r in rows])),
                   double_peak_baseline_mean=float(np.mean([r["double_peak_baseline"] for r in rows])),
                   double_peak_net_mean=float(np.mean([r["double_peak_net"] for r in rows])),
                   per_view=rows)
    (out / f"ghost_{args.tag}.json").write_text(json.dumps(summary, indent=2))

    gm.retention_map_png(*mats[0], out / f"retention_{args.tag}.png", grid=args.grid)
    print("\n📊 汇总")
    print(f"  PSNR            {summary['psnr_mean']:.3f}")
    print(f"  σ_eff（中位）    {summary['sigma_eff_median']:.2f} px")
    print(f"  模糊换来的 PSNR  +{summary['psnr_gain_from_blur_median']:.2f} dB（越高说明误差主要是糊）")
    print(f"  锐度保留（中位）  {summary['tile_ret_mean']:.3f}")
    print(f"  边缘扩散倍数      {summary['spread_ratio_median']:.2f}×")
    print(f"  双峰比例          {summary['double_peak_ratio_mean'] * 100:.1f}%"
          f"（底噪 {summary['double_peak_baseline_mean'] * 100:.1f}%，"
          f"净 {summary['double_peak_net_mean'] * 100:+.1f}pp）")
    print(f"✅ 输出: {out}/ghost_{args.tag}.json, retention_{args.tag}.png")


if __name__ == "__main__":
    main()
