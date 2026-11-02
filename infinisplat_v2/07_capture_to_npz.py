#!/usr/bin/env python3
"""华为 Remy 采集包（transforms.json + image/*.heic + pcd.ply）→ InfiniSplat v2 输入。

产出 OUT_DIR/{images/*.png, cameras.npz, meta.json}，可直接喂 02_run_inference.sh 的
IMAGES_DIR + CAMERAS（provided 模式）。

三个约定，错了就渲成白墙：
  1. transform_matrix 是 **c2w + OpenGL 相机系**（相机 -Z 指向 anchor，实测 cos=0.988）。
     InfiniSplat 要 OpenCV c2w → `c2w_cv = M @ diag(1,-1,-1,1)`。
  2. `--cameras` 的内参按**原图像素**解释（pipeline.prepare_images 自己乘 scale），
     所以在去畸变裁剪后的原图尺度上给，别再按 LONG_SIDE 缩一次。
  3. 源图带 k1..k3/p1/p2（k1≈0.09 的桶形畸变），而 InfiniSplat 只支持 pinhole。
     这里用 getOptimalNewCameraMatrix(alpha=0) + roi 裁剪 + 同步平移 cx/cy，
     不裁就带着 5% 的边角形变去反投影，几何会偏。

Env / args:
  --raw-dir   采集包路径（必填）
  --out-dir   输出目录（必填）
  --n-views   抽帧数，2..50（默认 32）。按序列等间隔，含首末帧
  --no-undistort  跳过去畸变（调试对比用）
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL c2w → OpenCV c2w


def pick_frames(n_frames, n_views):
    """序列等间隔抽帧，含首末。

    采集是环绕转圈，首末两帧朝向几乎重合（82 帧总转角 354°），所以"含首末"
    实际是给首尾各留一个近重复视角，不会浪费覆盖。
    """
    if n_views >= n_frames:
        return list(range(n_frames))
    idx = np.linspace(0, n_frames - 1, n_views).round().astype(int)
    return sorted(set(idx.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-views", type=int, default=32)
    ap.add_argument("--frames", default=None,
                    help="显式指定帧号（逗号分隔，覆盖 --n-views）。用于严格 train/holdout 切分："
                         "训练包与留出包共享同一 AR 世界坐标系，所以两包的 cameras.npz 可直接互为参照")
    ap.add_argument("--no-undistort", action="store_true")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    tj = raw / "transforms.json"
    if not tj.is_file():
        sys.exit(f"❌ 缺 transforms.json: {tj}")
    d = json.loads(tj.read_text(encoding="utf-8"))
    frames = d["frames"]
    if args.frames:
        picked = sorted({int(s) for s in args.frames.split(",") if s.strip()})
        bad = [i for i in picked if not 0 <= i < len(frames)]
        if bad:
            sys.exit(f"❌ 帧号越界 {bad}（总帧数 {len(frames)}）")
    else:
        if not 2 <= args.n_views <= 50:
            sys.exit("❌ n_views 必须在 2..50")
        picked = pick_frames(len(frames), args.n_views)
    (out / "images").mkdir(parents=True, exist_ok=True)

    c2ws, Ks, names, records = [], [], [], []
    crop_info = None
    for n, i in enumerate(picked):
        fr = frames[i]
        src = raw / fr["file_path"].replace("./", "")
        if not src.is_file():
            sys.exit(f"❌ 图片缺失: {src}")

        with Image.open(src) as im:
            arr = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        h, w = arr.shape[:2]

        K = np.array([[fr["fl_x"], 0.0, fr["cx"]],
                      [0.0, fr["fl_y"], fr["cy"]],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        if (w, h) != (fr["w"], fr["h"]):
            # EXIF 转置过或分辨率不符时按比例归一到实际像素
            K[0] *= w / fr["w"]
            K[1] *= h / fr["h"]

        if not args.no_undistort and "k1" in fr:
            dist = np.array([fr["k1"], fr["k2"], fr["p1"], fr["p2"], fr["k3"]], dtype=np.float64)
            newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
            arr = cv2.undistort(arr, K, dist, None, newK)
            x, y, rw, rh = roi
            if rw > 0 and rh > 0:
                arr = arr[y:y + rh, x:x + rw]
                newK[0, 2] -= x
                newK[1, 2] -= y
            K = newK
            if crop_info is None:
                crop_info = dict(src_wh=[w, h], out_wh=[int(arr.shape[1]), int(arr.shape[0])],
                                 roi=[int(v) for v in roi],
                                 dist=[float(v) for v in dist])

        M = np.array(fr["transform_matrix"], dtype=np.float64)  # c2w, OpenGL
        c2ws.append(M @ FLIP)

        # 内参跟着裁剪走：pipeline.prepare_images 只按图像尺寸乘 scale，
        # 所以这里必须是"裁剪后原图像素"尺度。
        Ks.append(K.astype(np.float32))

        name = f"{n:03d}_{src.stem}.png"
        Image.fromarray(arr).save(out / "images" / name, optimize=False)
        names.append(name)
        records.append(dict(idx=int(i), src=src.name, out=name, wh=[int(arr.shape[1]), int(arr.shape[0])]))

    c2w = np.stack(c2ws).astype(np.float32)
    K = np.stack(Ks).astype(np.float32)
    np.savez(out / "cameras.npz", c2w=c2w, intrinsics=K, names=np.array(names))

    # 自检：相机中心到场景原点距离 + 相邻视角夹角，退化会立刻暴露
    C = c2w[:, :3, 3]
    rad = np.linalg.norm(C, axis=1)
    Rs = c2w[:, :3, :3]

    def ang(a, b):
        return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))

    pairs = [ang(Rs[i], Rs[j]) for i in range(len(Rs)) for j in range(i + 1, len(Rs))]
    meta = dict(source=str(raw), n_frames_total=len(frames), n_views=len(picked),
                picked_idx=picked, undistort=not args.no_undistort, crop=crop_info,
                cam_radius=dict(min=float(rad.min()), max=float(rad.max()), mean=float(rad.mean())),
                pairwise_deg=dict(min=float(np.min(pairs)), median=float(np.median(pairs)),
                                  max=float(np.max(pairs))),
                frames=records)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ 输出 {out}")
    print(f"   🖼️  {len(picked)}/{len(frames)} 帧  K[0] fx={K[0,0,0]:.1f} cx={K[0,0,2]:.1f} cy={K[0,1,2]:.1f}")
    if crop_info:
        print(f"   🩹 去畸变: {crop_info['src_wh']} → {crop_info['out_wh']} (roi={crop_info['roi']})")
    print(f"   📷 相机半径 {rad.min():.3f}~{rad.max():.3f} m (mean {rad.mean():.3f})")
    print(f"   🔭 视角两两夹角 min {np.min(pairs):.1f}° / median {np.median(pairs):.1f}° / max {np.max(pairs):.1f}°")
    if np.median(pairs) < 2.0:
        print("   ⚠️ 夹角中位数 <2°，位姿疑似退化（全部相机几乎同向）", file=sys.stderr)


if __name__ == "__main__":
    main()
