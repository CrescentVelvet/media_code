#!/usr/bin/env python3
"""拆解 InfiniSplat v2 单次推理的耗时构成：权重加载 / 图像预处理 / DA3 位姿 / 主前向 / 导出。

CLI 只打一个总时间，无法回答"20 秒里多少是加载"。这里复刻 cli.main() 的调用顺序分段计时，
并额外记录 torch 侧显存峰值（不含 CUDA context 约 300 MB 的固定开销）。

用法:
  python bench_stages.py --model ~/model/infinisplat_v2/da3 \
      --images-dir /mnt/d/dataset/物体三图数据/剃须刀 --out stages.json
"""
import argparse
import json
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--long-side", type=int, default=518)
    ap.add_argument("--max-views", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--repeat", type=int, default=1,
                    help=">1 时在首次加载后重复跑 reconstruct，测常驻复用（warm）单次成本")
    ap.add_argument("--out", default="stages.json")
    args = ap.parse_args()

    from infinisplat.pipeline import Pipeline, prepare_images
    from infinisplat.export import export_ply

    imgs = sorted(p for p in Path(args.images_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not imgs:
        raise SystemExit("❌ 没有图像")
    out_dir = Path(args.out).with_suffix("")
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)

    t = {}
    marks = {}

    def tick(name):
        torch.cuda.synchronize()
        now = time.time()
        marks[name] = round(now - last[0], 2)
        last[0] = now

    last = [time.time()]

    pipe = Pipeline.from_pretrained(args.model, args.device)   # 含 6.6 GB SHA256 校验
    tick("load_s")

    pixels, _, _ = prepare_images(imgs, long_side=args.long_side)
    tick("preprocess_s")

    cam = pipe.model.predict_cameras(pixels.to(args.device))
    tick("da3_pose_s")

    scene = pipe.reconstruct(imgs, cameras=None, pose_mode="da3-pose",
                             long_side=args.long_side, max_views=args.max_views)
    tick("reconstruct_s")   # 含重复的位姿预检，见下方说明

    export_ply(scene, str(out_dir), html=False)
    tick("export_s")

    # 常驻复用：模型已在卡上，重复调用测稳定态单次成本（不含加载）
    warm = []
    for i in range(max(0, args.repeat - 1)):
        torch.cuda.synchronize()
        t0 = time.time()
        scene = pipe.reconstruct(imgs, cameras=None, pose_mode="da3-pose",
                                 long_side=args.long_side, max_views=args.max_views)
        torch.cuda.synchronize()
        warm.append(round(time.time() - t0, 2))
        print(f"  warm#{i+1:<2d} {warm[-1]}s")

    t.update(marks)
    t["views"] = len(imgs)
    t["long_side"] = args.long_side
    if warm:
        t["warm_runs_s"] = warm
        t["warm_mean_s"] = round(sum(warm) / len(warm), 2)
        t["warm_min_s"] = min(warm)
    t["gaussians"] = scene["stats"].get("final_count") if isinstance(scene, dict) else None
    t["peak_torch_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    t["peak_torch_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(t, f, ensure_ascii=False, indent=2)
    for k, v in t.items():
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
