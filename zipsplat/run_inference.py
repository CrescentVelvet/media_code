#!/usr/bin/env python3
"""任意图像目录 → 官方 ZipSplat 前馈 + TokenBA → PLY、相机、输入视角渲染。

Env vars:
  WEIGHT_PATH, ZIPSPLAT_REPO, OUT_DIR, IMAGES_DIR, MAX_VIEWS,
  BA (1/0), SAVE_FEEDFORWARD, RENDER_VIDEO, RENDER_SIZE, VIDEO_FRAMES
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

REPO = Path(os.environ.get("ZIPSPLAT_REPO", "~/repos/ZipSplat")).expanduser()
sys.path.insert(0, str(REPO))
from zipsplat import ZipSplat, load_image  # noqa: E402
from zipsplat import viz  # noqa: E402

WEIGHT_PATH = os.environ.get("WEIGHT_PATH", "zipsplat")
IMAGES_DIR = Path(os.environ["IMAGES_DIR"]).expanduser()
OUT_DIR = Path(os.environ["OUT_DIR"]).expanduser()
MAX_VIEWS = int(os.environ.get("MAX_VIEWS", "16"))
RUN_BA = os.environ.get("BA", "1") == "1"
SAVE_FF = os.environ.get("SAVE_FEEDFORWARD", "1") == "1"
RENDER_VIDEO = os.environ.get("RENDER_VIDEO", "0") == "1"
RENDER_SIZE = int(os.environ.get("RENDER_SIZE", "504"))
VIDEO_FRAMES = int(os.environ.get("VIDEO_FRAMES", "96"))
SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def gpu_stats():
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def main():
    files = sorted(p for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    if not files:
        sys.exit(f"❌ 没有支持的图像: {IMAGES_DIR}")
    if len(files) > MAX_VIEWS:
        idx = np.linspace(0, len(files) - 1, MAX_VIEWS).round().astype(int)
        files = [files[i] for i in sorted(set(idx.tolist()))]
    if not 1 <= len(files) <= 64:
        sys.exit(f"❌ 视角数 {len(files)} 超出本脚本安全范围 1..64")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🚀 ZipSplat: {len(files)} 视角 -> {OUT_DIR}", flush=True)
    print(f"🏋️ 权重: {WEIGHT_PATH}", flush=True)
    for p in files:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            print(f"  🖼️ {p.name}: {im.width}x{im.height}", flush=True)
    images = []
    for p in files:
        with Image.open(p) as im:
            images.append(load_image(p))
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    metrics = {
        "input_dir": str(IMAGES_DIR), "image_files": [p.name for p in files],
        "num_views": len(files), "weight_path": str(WEIGHT_PATH),
        "official_commit": commit, "token_ba": RUN_BA,
        "model_input_size": None, "timing_seconds": {}, "gpu": {},
    }
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = ZipSplat(weights=WEIGHT_PATH).cuda().eval()
    torch.cuda.synchronize()
    metrics["timing_seconds"]["load_model"] = time.perf_counter() - t0
    metrics["gpu"]["after_load"] = gpu_stats()
    print(f"✅ 模型加载 {metrics['timing_seconds']['load_model']:.2f}s", flush=True)

    t0 = time.perf_counter()
    with torch.no_grad():
        # 与官方 API 相同：先跑 ba=False，再按需调用 refine；默认 ba=True 会少一次可对照的前馈结果。
        pred = model(images, ba=False)
        torch.cuda.synchronize()
        metrics["timing_seconds"]["feedforward"] = time.perf_counter() - t0
        metrics["gpu"]["after_feedforward"] = gpu_stats()
        metrics["model_input_size"] = list(pred["model_images"].shape[-2:])[::-1]
        metrics["num_gaussians_feedforward"] = pred["gaussians"].num_gaussians
        metrics["gaussians_per_token"] = model.model.gaussian_head.gaussians_per_token
        print(f"✅ 前馈 {metrics['timing_seconds']['feedforward']:.2f}s / "
              f"{pred['gaussians'].num_gaussians:,} Gaussians", flush=True)
        if SAVE_FF or not RUN_BA:
            pred["gaussians"].save_ply(OUT_DIR / "gaussians_feedforward.ply")
        if RUN_BA:
            t0 = time.perf_counter()
            pred = model.refine(pred, verbose=True)
            torch.cuda.synchronize()
            metrics["timing_seconds"]["token_ba"] = time.perf_counter() - t0
            metrics["gpu"]["after_token_ba"] = gpu_stats()
            print(f"✅ TokenBA {metrics['timing_seconds']['token_ba']:.2f}s", flush=True)

    gaussians = pred["gaussians"]
    (OUT_DIR / "inputs").mkdir(exist_ok=True)
    if os.environ.get("DUMP_PREDICTION", "0") == "1":
        # 调试用缓存可能达数 GB，默认不保存；正式 PLY/相机/指标不受影响。
        torch.save({k: v for k, v in pred.items() if k != "gaussians"}, OUT_DIR / "prediction.pt")
        torch.save(gaussians.data_.cpu(), OUT_DIR / "gaussians.pt")
    metrics["num_gaussians"] = gaussians.num_gaussians
    ply_name = "gaussians.ply" if RUN_BA else "gaussians_feedforward.ply"
    gaussians.save_ply(OUT_DIR / ply_name)
    cameras = pred["cameras"].cpu().data_.numpy()
    poses = pred["poses"].cpu().Rt.numpy()
    np.savez(OUT_DIR / "predicted_cameras.npz", cameras=cameras, c2w=poses,
             names=np.array([p.name for p in files]))
    with open(OUT_DIR / "predicted_cameras.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"convention": "OpenCV c2w; camera=[w,h,fx,fy,cx,cy]",
                   "frames": [{"name": p.name, "c2w": T.tolist(), "camera": c.tolist()}
                              for p, T, c in zip(files, poses, cameras)]}, f,
                  ensure_ascii=False, indent=2)

    t0 = time.perf_counter()
    render_dir = OUT_DIR / "renders"
    render_dir.mkdir(exist_ok=True)
    render_records = []
    with torch.no_grad():
        for i, p in enumerate(files):
            rgb, info = gaussians.render(pred["model_cameras"][i], pred["poses"][i])
            rgb = rgb[0].float().clamp(0, 1).moveaxis(0, -1).cpu().numpy()
            alpha = info["alphas"][0, 0].float().cpu().numpy()
            if os.environ.get("DUMP_INPUT_METRICS", "0") == "1":
                torch.save({"rgb": rgb, "alpha": alpha}, OUT_DIR / "inputs" / f"{i:03d}.pt")
            Image.fromarray((rgb * 255).round().astype(np.uint8)).save(render_dir / f"{i:03d}.png")
            render_records.append({"index": i, "name": p.name,
                                   "alpha_gt_05_fraction": float((alpha > 0.5).mean())})
    torch.cuda.synchronize()
    metrics["timing_seconds"]["render_inputs"] = time.perf_counter() - t0
    metrics["render_alpha_coverage"] = render_records
    metrics["gpu"]["after_render"] = gpu_stats()

    if RENDER_VIDEO:
        # 官方 wiggle 轨迹是预测坐标系中的环绕，非 GT 轨迹；不能把它当 held-out 评测。
        t0 = time.perf_counter()
        viz.turntable(gaussians, OUT_DIR / "orbit_wiggle.mp4", render_size=RENDER_SIZE,
                      num_frames=VIDEO_FRAMES, fps=24, sweep_deg=None)
        torch.cuda.synchronize()
        metrics["timing_seconds"]["render_video"] = time.perf_counter() - t0
        metrics["gpu"]["after_video"] = gpu_stats()
    metrics["timing_seconds"]["total"] = sum(metrics["timing_seconds"].values())
    metrics["software"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                           "torch_cuda": torch.version.cuda}
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"🎉 完成: {OUT_DIR}\n{json.dumps(metrics['timing_seconds'], indent=2)}", flush=True)
    print(f"🎮 峰值 allocated: {metrics['gpu']['after_feedforward']['peak_allocated_mib']:.1f} MiB",
          flush=True)


if __name__ == "__main__":
    main()
