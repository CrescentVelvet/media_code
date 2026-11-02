#!/usr/bin/env python3
"""Step 2：预算重分配对照 —— 同 raw 高斯量下「加分辨率」vs「加视角」。

为什么需要单独一个脚本：两组的渲染分辨率不同（777 → 588×784，518 → 392×518），
PSNR 与 σ_eff 都是**像素单位**，直接跨表比会被分辨率污染。这里把两组渲染都
重采样到同一目标尺寸（默认 518 长边 = 短边 392），GT 也统一从原图重算一次，
于是：

  native 口径 : 各自原生分辨率 —— 回答「画质上限」
  shared 口径 : 统一到目标尺寸 —— 回答「在目标显示尺寸下，谁真的更准」

判读（Step 2 的核心问题）：
  * native σ_eff 在 777 下**仍约 1.5 px** → 绝对模糊没变，分辨率真换来了细节（锐 1.5×）
  * native σ_eff 涨到 ~2.2 px（等比放大）→ 模糊跟着放大，加分辨率白搭
  * shared PSNR 21v@777 ≈ 48v@518 → 高频优势在目标尺寸上体现不出来；> 才说明是真细节

用法:
  python 11_budget_compare.py \
      --a-result ~/output/.../jiajia_s2_v21 --a-images ~/output/.../_captures/jiajia_s2_v21/images --a-tag 21v@777 \
      --b-result ~/output/.../jiajia_s2_v48 --b-images ~/output/.../_captures/jiajia_s2_v48/images --b-tag 48v@518 \
      --out /mnt/d/output/.../_diag
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import ghost_metrics as gm


def load_render(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def resize_np(img, w, h):
    """统一缩放：缩小时用 BOX（面积平均，不产生振铃），放大时用 BICUBIC。"""
    sw, sh = img.size
    method = Image.Resampling.BOX if (w <= sw and h <= sh) else Image.Resampling.BICUBIC
    return np.asarray(img.resize((w, h), method), np.float32) / 255.0


def collect(tag, result, images_dir, target_wh, max_views):
    res = Path(result)
    meta = json.loads((res / "result.json").read_text())
    records = meta.get("inputs", [])
    rdir = res / "eval"
    imgs = sorted(p for p in Path(images_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    renders = sorted(rdir.glob("render_*.png"))
    n = min(len(imgs), len(renders), len(records), max_views)
    TW, TH = target_wh

    rows = []
    for i in range(n):
        pw, ph = (int(v) for v in records[i]["processed_wh"])
        gt_n = gm.load_gt_processed(imgs[i], pw, ph)
        rd_n = np.asarray(load_render(renders[i]), np.float32) / 255.0
        if rd_n.shape != gt_n.shape:
            raise SystemExit(f"❌ {tag} 视角 {i}: GT{gt_n.shape} vs render{rd_n.shape}")

        # shared：GT 从原图直接重算到目标尺寸（两组完全同源，与各自 processed 无关）
        gt_s = gm.load_gt_processed(imgs[i], TW, TH)
        rd_s = resize_np(load_render(renders[i]), TW, TH)

        sig_n, _, _ = gm.equivalent_blur_sigma(gt_n, rd_n)
        sig_s, _, _ = gm.equivalent_blur_sigma(gt_s, rd_s)
        mat_s = gm.tile_retention(gt_s, rd_s, 4)
        sp_s, dp_s, ne_s, _ = gm.edge_stats(gt_s, rd_s)
        _, dp_base, _, _ = gm.edge_stats(gt_s, gt_s)
        rows.append(dict(
            view=i, image=imgs[i].name, native_wh=[pw, ph],
            psnr_native=gm.psnr(gt_n, rd_n), sigma_native=sig_n,
            psnr_shared=gm.psnr(gt_s, rd_s), sigma_shared=sig_s,
            tile_ret_native=float(np.nanmean(gm.tile_retention(gt_n, rd_n, 4))),
            tile_ret_shared=float(np.nanmean(mat_s)),
            spread_shared=sp_s, double_peak_shared=dp_s, double_peak_base=dp_base,
            n_edges=ne_s,
        ))
        print(f"  [{i:02d}] {tag} native {pw}x{ph}  PSNR {rows[-1]['psnr_native']:6.3f}"
              f" σ {sig_n:4.1f}px  |  shared {TW}x{TH}  PSNR {rows[-1]['psnr_shared']:6.3f}"
              f" σ {sig_s:4.1f}px  锐度 {rows[-1]['tile_ret_shared']:.3f}")

    def mean(k):
        return float(np.mean([r[k] for r in rows]))

    def med(k):
        return float(np.median([r[k] for r in rows]))

    return dict(tag=tag, views=n, native_wh=rows[0]["native_wh"],
                gaussians=int(meta["stats"]["final_count"]),
                raw_count=int(meta["stats"]["raw_count"]),
                psnr_native_mean=mean("psnr_native"), sigma_native_med=med("sigma_native"),
                psnr_shared_mean=mean("psnr_shared"), sigma_shared_med=med("sigma_shared"),
                tile_ret_native_med=med("tile_ret_native"),
                tile_ret_shared_med=med("tile_ret_shared"),
                spread_shared_med=med("spread_shared"),
                double_peak_shared_mean=mean("double_peak_shared"),
                double_peak_base_mean=mean("double_peak_base"),
                per_view=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-result", required=True)
    ap.add_argument("--a-images", required=True)
    ap.add_argument("--a-tag", default="A")
    ap.add_argument("--b-result", required=True)
    ap.add_argument("--b-images", required=True)
    ap.add_argument("--b-tag", default="B")
    ap.add_argument("--target-wh", help="统一口径宽高，如 392,518；默认取 B 组首帧的 processed_wh")
    ap.add_argument("--max-views", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.target_wh:
        TW, TH = (int(v) for v in args.target_wh.split(","))
    else:
        m = json.loads((Path(args.b_result) / "result.json").read_text())
        TW, TH = (int(v) for v in m["inputs"][0]["processed_wh"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"🎯 统一口径 {TW}x{TH}")

    A = collect(args.a_tag, args.a_result, args.a_images, (TW, TH), args.max_views)
    B = collect(args.b_tag, args.b_result, args.b_images, (TW, TH), args.max_views)

    both = dict(target_wh=[TW, TH], a=A, b=B)
    (out / "budget_compare.json").write_text(json.dumps(both, indent=2))

    print("\n📊 同 raw 预算对照")
    hdr = f"{'配置':<12}{'原生':<12}{'raw':>10}{'PSNR原生':>10}{'σ原生':>7}" \
          f"{'PSNR@518':>10}{'σ@518':>7}{'锐度@518':>9}{'双峰@518':>9}"
    print(hdr)
    for r in (A, B):
        nw = f"{r['native_wh'][0]}x{r['native_wh'][1]}"
        print(f"{r['tag']:<12}{nw:<12}{r['raw_count']:>10,d}{r['psnr_native_mean']:>10.3f}"
              f"{r['sigma_native_med']:>7.1f}{r['psnr_shared_mean']:>10.3f}"
              f"{r['sigma_shared_med']:>7.1f}{r['tile_ret_shared_med']:>9.3f}"
              f"{r['double_peak_shared_mean'] * 100:>8.1f}%")
    print(f"✅ 输出: {out}/budget_compare.json")


if __name__ == "__main__":
    main()
