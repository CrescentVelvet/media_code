#!/usr/bin/env python3
"""把一个示例（图像 + cameras.npz）划成 输入集 / 留出目标集。

官方划分未公开，这里自建协议：目标视角在序列上等间隔抽取（默认 4 个，对齐论文
"每例 4 个 held-out 目标视角"），输入集保持视频序列顺序。等间隔而非随机，是为了
避免 target 与 input 相邻造成"几乎同一视角"的乐观偏差【本笔记自定，非论文协议】。

用法:
  python make_holdout.py --images-dir <dir> --cameras <npz> --out <dir> \
      [--n-input 8] [--n-target 4]
输出:
  <out>/input/{images/*, cameras.npz}
  <out>/target/{images/*, cameras.npz, manifest.json}
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-input", type=int, default=8)
    ap.add_argument("--n-target", type=int, default=4)
    args = ap.parse_args()

    src = Path(args.images_dir)
    imgs = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    cams = np.load(args.cameras, allow_pickle=False)
    c2w, K = cams["c2w"], cams["intrinsics"]
    if len(imgs) != len(c2w):
        raise SystemExit(f"❌ 图像数 {len(imgs)} 与相机数 {len(c2w)} 不一致")

    total = len(imgs)
    if args.n_input + args.n_target > total:
        raise SystemExit(f"❌ 输入 {args.n_input} + 目标 {args.n_target} > 总视角 {total}")

    # 等间隔选目标：把序列切成 n_target+1 段，每段中点取一个，避开首尾
    target_idx = [int(round((k + 0.5) * total / args.n_target)) for k in range(args.n_target)]
    target_idx = [min(i, total - 1) for i in target_idx]
    rest = [i for i in range(total) if i not in set(target_idx)]
    # 输入集在剩余帧里等间隔抽，保持序列顺序
    step = max(1, len(rest) // args.n_input)
    input_idx = sorted(rest[::step][: args.n_input])

    out = Path(args.out)
    (out / "input" / "images").mkdir(parents=True, exist_ok=True)
    (out / "target" / "images").mkdir(parents=True, exist_ok=True)

    for i in input_idx:
        shutil.copy2(imgs[i], out / "input" / "images" / imgs[i].name)
    for i in target_idx:
        shutil.copy2(imgs[i], out / "target" / "images" / imgs[i].name)

    np.savez(out / "input" / "cameras.npz", c2w=c2w[input_idx], intrinsics=K[input_idx])
    np.savez(out / "target" / "cameras.npz", c2w=c2w[target_idx], intrinsics=K[target_idx])

    manifest = dict(total_views=total, n_input=len(input_idx), n_target=len(target_idx),
                    input_index=input_idx, target_index=target_idx,
                    input_names=[imgs[i].name for i in input_idx],
                    target_names=[imgs[i].name for i in target_idx],
                    protocol="evenly-spaced target split (self-defined, not the paper's official split)")
    (out / "target" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"✅ 输入 {len(input_idx)} 视角 / 目标 {len(target_idx)} 视角（共 {total}）")
    print(f"   target: {[imgs[i].name for i in target_idx]}")


if __name__ == "__main__":
    main()
