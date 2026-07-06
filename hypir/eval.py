#!/usr/bin/env python3
"""Evaluate a trained HYPIR-SD2 LoRA on paired LQ/HQ images.

Loads the trained LoRA (WEIGHT_PATH), restores each LQ image, and — when
TEST_HQ_DIR is given — computes PSNR / SSIM / LPIPS of the result vs HQ, plus
the same metrics for a bicubic-upsampled LQ vs HQ as a no-model baseline (so
the model's restoration gain is visible). Saves restored PNGs, side-by-side
comparison montages (LQ | result | HQ), and a metrics CSV.

Mirrors run_inference.py's env-driven loading; adds metrics + comparison.
Metrics: PSNR/SSIM are numpy/torch (no extra deps); LPIPS uses the `lpips`
package (already a HYPIR dep) — if its VGG weights can't download, LPIPS is
skipped and PSNR/SSIM still print.

Env (set by 05_eval.sh):
  HYPIR_DIR, BASE_MODEL_PATH, WEIGHT_PATH, LORA_RANK, LORA_MODULES,
  MODEL_T, COEFF_T, TEST_LQ_DIR, TEST_HQ_DIR, EVAL_DIR,
  SCALE_BY, UPSCALE, PATCH_SIZE, STRIDE, SEED, DEVICE,
  EVAL_LIMIT, SAVE_COMPARE
"""
import csv
import os
import sys
import time
from pathlib import Path

HYPIR_DIR = os.environ.get("HYPIR_DIR", "../HYPIR")
sys.path.insert(0, HYPIR_DIR)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from accelerate.utils import set_seed
from torchvision import transforms

from HYPIR.enhancer.sd2 import SD2Enhancer

BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH")
WEIGHT_PATH = os.environ.get("WEIGHT_PATH")
LORA_RANK = int(os.environ.get("LORA_RANK", "256"))
LORA_MODULES = os.environ.get(
    "LORA_MODULES",
    "to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj",
).split(",")
MODEL_T = int(os.environ.get("MODEL_T", "200"))
COEFF_T = int(os.environ.get("COEFF_T", "200"))
TEST_LQ_DIR = os.environ.get("TEST_LQ_DIR")
TEST_HQ_DIR = os.environ.get("TEST_HQ_DIR") or None
EVAL_DIR = os.environ.get("EVAL_DIR")
SCALE_BY = os.environ.get("SCALE_BY", "factor")
UPSCALE = int(os.environ.get("UPSCALE", "1"))
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "512"))
STRIDE = int(os.environ.get("STRIDE", "256"))
SEED = int(os.environ.get("SEED", "231"))
DEVICE = os.environ.get("DEVICE", "cuda")
EVAL_LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))   # 0 = all
SAVE_COMPARE = os.environ.get("SAVE_COMPARE", "1") == "1"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
TO_TENSOR = transforms.ToTensor()


# ---------- metrics ----------
def psnr_rgb(a, b):
    """RGB PSNR (dB) on uint8 HxWx3 arrays. Higher = better."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 * 255.0 / mse))


def _gauss_window(size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


_WIN = _gauss_window()


def ssim_rgb(a, b):
    """RGB SSIM (Wang et al., 11x11 gaussian window) on uint8 HxWx3. Higher=better."""
    ta = torch.from_numpy(a).permute(2, 0, 1).float().div(255.0).to(_WIN.device)
    tb = torch.from_numpy(b).permute(2, 0, 1).float().div(255.0).to(_WIN.device)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for c in range(ta.shape[0]):
        x = ta[c:c + 1].unsqueeze(0)
        y = tb[c:c + 1].unsqueeze(0)
        mu_x = F.conv2d(x, _WIN, padding=5)
        mu_y = F.conv2d(y, _WIN, padding=5)
        sig_x = F.conv2d(x * x, _WIN, padding=5) - mu_x * mu_x
        sig_y = F.conv2d(y * y, _WIN, padding=5) - mu_y * mu_y
        sig_xy = F.conv2d(x * y, _WIN, padding=5) - mu_x * mu_y
        ss = ((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)) / \
             ((mu_x * mu_x + mu_y * mu_y + C1) * (sig_x + sig_y + C2))
        vals.append(ss.mean())
    return float(torch.stack(vals).mean().item())


def _lpips_dist(net, pil_a, pil_b):
    a = TO_TENSOR(pil_a.convert("RGB")).unsqueeze(0).to(DEVICE) * 2 - 1
    b = TO_TENSOR(pil_b.convert("RGB")).unsqueeze(0).to(DEVICE) * 2 - 1
    return float(net(a, b).item())


def _montage(triples, out_path):
    """Save [LQ | result | HQ] side-by-side with labels."""
    h = max(im.size[1] for im, _ in triples)
    panels = []
    for im, label in triples:
        w = int(im.size[0] * h / im.size[1])
        im2 = im.resize((w, h), Image.BICUBIC)
        panel = Image.new("RGB", (w, h + 22), (0, 0, 0))
        ImageDraw.Draw(panel).text((4, 2), label, (255, 255, 255))
        panel.paste(im2, (0, 22))
        panels.append(panel)
    canvas = Image.new("RGB", (sum(p.size[0] for p in panels), h + 22), (0, 0, 0))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.size[0]
    canvas.save(out_path)


def main():
    if not BASE_MODEL_PATH:
        sys.exit("ERROR: BASE_MODEL_PATH not set.")
    if not WEIGHT_PATH:
        sys.exit("ERROR: WEIGHT_PATH not set.")
    if not TEST_LQ_DIR:
        sys.exit("ERROR: TEST_LQ_DIR not set.")
    if not EVAL_DIR:
        sys.exit("ERROR: EVAL_DIR not set.")

    set_seed(SEED)

    model = SD2Enhancer(
        base_model_path=BASE_MODEL_PATH,
        weight_path=WEIGHT_PATH,
        lora_modules=LORA_MODULES,
        lora_rank=LORA_RANK,
        model_t=MODEL_T,
        coeff_t=COEFF_T,
        device=DEVICE,
    )
    print(f"[*] loading models (device={DEVICE}) ...")
    t0 = time.time()
    model.init_models()
    print(f"[*] 模型加载耗时: {time.time() - t0:.2f}s")

    # LPIPS is optional (VGG weights download may fail behind a proxy).
    lpips_net = None
    if TEST_HQ_DIR:
        try:
            import lpips
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lpips_net = lpips.LPIPS(net="vgg", verbose=False).to(DEVICE).eval()
            print("[*] LPIPS(net=vgg) ready")
        except Exception as e:
            print(f"[!] LPIPS unavailable ({e}); will skip LPIPS, PSNR/SSIM still computed.")

    input_dir = Path(TEST_LQ_DIR)
    eval_dir = Path(EVAL_DIR)
    result_dir = eval_dir / "result"
    compare_dir = eval_dir / "compare"
    result_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_COMPARE:
        compare_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                images.append(Path(root) / f)
    images.sort(key=lambda x: str(x.relative_to(input_dir)))
    if EVAL_LIMIT > 0:
        images = images[:EVAL_LIMIT]
    if not images:
        sys.exit(f"ERROR: no images in {input_dir}")
    print(f"[*] {len(images)} image(s) to eval: {input_dir} -> {result_dir}  "
          f"(scale_by={SCALE_BY} upscale={UPSCALE} patch={PATCH_SIZE} stride={STRIDE} seed={SEED})")
    if not TEST_HQ_DIR:
        print("[!] TEST_HQ_DIR not set — will only restore + save (no metrics).")

    rows = []
    infer_times = []
    ok = 0
    for i, fp in enumerate(images, 1):
        rel = fp.relative_to(input_dir)
        result_path = result_dir / rel.with_suffix(".png")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        lq_pil = Image.open(fp).convert("RGB")
        w0, h0 = lq_pil.size
        prompt = ""

        try:
            t1 = time.time()
            result = model.enhance(
                lq=TO_TENSOR(lq_pil).unsqueeze(0),
                prompt=prompt,
                scale_by=SCALE_BY,
                upscale=UPSCALE,
                patch_size=PATCH_SIZE,
                stride=STRIDE,
                return_type="pil",
            )[0]
            dt = time.time() - t1
            w1, h1 = result.size
            result.save(result_path)
            infer_times.append(dt)
            ok += 1
            line = (f"[{i}/{len(images)}] {fp.name}  | {w0}x{h0} -> {w1}x{h1} "
                    f"| 推理 {dt:.2f}s")
        except Exception as e:
            print(f"[{i}/{len(images)}] {fp.name}  ! failed: {e}", file=sys.stderr)
            continue

        # bicubic baseline at the output resolution (no model)
        lq_up = lq_pil.resize((w1, h1), Image.BICUBIC)

        if TEST_HQ_DIR:
            hq_path = Path(TEST_HQ_DIR) / rel
            if not hq_path.exists():
                line += "  [no HQ, skipped metrics]"
                print(line)
                continue
            hq_pil = Image.open(hq_path).convert("RGB")
            if hq_pil.size != result.size:
                hq_pil = hq_pil.resize((w1, h1), Image.BICUBIC)
            rarr = np.array(result)
            harr = np.array(hq_pil)
            larr = np.array(lq_up)
            p_m = psnr_rgb(rarr, harr)
            s_m = ssim_rgb(rarr, harr)
            p_b = psnr_rgb(larr, harr)
            s_b = ssim_rgb(larr, harr)
            row = {"file": rel.as_posix(), "lq": f"{w0}x{h0}", "out": f"{w1}x{h1}",
                   "bicubic_psnr": f"{p_b:.3f}", "bicubic_ssim": f"{s_b:.4f}",
                   "model_psnr": f"{p_m:.3f}", "model_ssim": f"{s_m:.4f}"}
            line += (f"  | bicubic PSNR {p_b:6.2f} SSIM {s_b:.3f}"
                     f" | model PSNR {p_m:6.2f} SSIM {s_m:.3f}"
                     f" | ΔPSNR {p_m - p_b:+.2f}")
            if lpips_net is not None:
                l_m = _lpips_dist(lpips_net, result, hq_pil)
                l_b = _lpips_dist(lpips_net, lq_up, hq_pil)
                row["bicubic_lpips"] = f"{l_b:.4f}"
                row["model_lpips"] = f"{l_m:.4f}"
                line += (f"  | bicubic LPIPS {l_b:.3f} model LPIPS {l_m:.3f}"
                         f" ΔLPIPS {l_m - l_b:+.3f}")
            rows.append(row)
            print(line)
            if SAVE_COMPARE:
                cmp_path = compare_dir / rel.with_suffix(".png")
                cmp_path.parent.mkdir(parents=True, exist_ok=True)
                _montage(
                    [(lq_up, "LQ(bicubic)"), (result, "result"), (hq_pil, "HQ")],
                    str(cmp_path),
                )
        else:
            print(line)

    # summary
    print(f"[*] done. {ok}/{len(images)} restored.")
    if infer_times:
        avg = sum(infer_times) / len(infer_times)
        print(f"[*] 单图推理耗时: avg {avg:.2f}s, min {min(infer_times):.2f}s, "
              f"max {max(infer_times):.2f}s")
    if rows:
        def avg(col):
            return sum(float(r[col]) for r in rows if col in r) / len(rows)
        print("[*] === 指标汇总 (model vs HQ; bicubic 为无模型基线) ===")
        print(f"    bicubic: PSNR {avg('bicubic_psnr'):.2f}  SSIM {avg('bicubic_ssim'):.4f}"
              + (f"  LPIPS {avg('bicubic_lpips'):.4f}" if "bicubic_lpips" in rows[0] else ""))
        print(f"    model  : PSNR {avg('model_psnr'):.2f}  SSIM {avg('model_ssim'):.4f}"
              + (f"  LPIPS {avg('model_lpips'):.4f}" if "model_lpips" in rows[0] else ""))
        print(f"    ΔPSNR  : {avg('model_psnr') - avg('bicubic_psnr'):+.2f} dB")
        # CSV
        csv_path = eval_dir / "metrics.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[*] metrics CSV: {csv_path}")
        if SAVE_COMPARE:
            print(f"[*] 对比图(LQ|result|HQ): {compare_dir}/")
        print(f"[*] 复原结果: {result_dir}/")


if __name__ == "__main__":
    _WIN = _WIN.to(DEVICE)
    main()
