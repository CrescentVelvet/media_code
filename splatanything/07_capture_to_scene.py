#!/usr/bin/env python3
"""华为 Remy 采集包（transforms.json + image/*.heic + pcd.ply）→ SplatAnything 场景目录。

产出 OUT/{images/*.png, prior_camera.json, meta.json}，直接喂 02_run_inference.sh 的 INPUT。

三个约定，错了就渲成白墙（与 infinisplat_v2/07_capture_to_npz.py 同源，实测判定）：
  1. transform_matrix 是 **c2w + OpenGL 相机系**（相机 -Z 指向 anchor，实测 cos=0.988）。
     SplatAnything 要 OpenCV c2w（X 右 / Y 下 / Z 前）→ `c2w_cv = M @ diag(1,-1,-1,1)`。
  2. JSON 内参按**原图像素**解释，且 loader 会用 K 的 (cx,cy) 决定裁剪中心。所以这里
     在去畸变裁剪后的原图尺度上给，不要按 --resolution 再缩一次。
  3. 源图带 k1..k3/p1/p2（k1≈0.09 桶形畸变），模型只支持 pinhole。这里用
     getOptimalNewCameraMatrix(alpha=0) + roi 裁剪 + 同步平移 cx/cy，不裁就带着 5%
     的边角形变去反投影，跨视角几何会偏。

Args:
  --raw-dir   采集包路径（必填）
  --out-dir   输出场景目录（必填）
  --n-views   抽帧数，2..50（默认 24，官方示例同量级）
  --frames    显式帧号（逗号分隔），优先级高于 --n-views
  --no-undistort  跳过去畸变（调试对比用）
  --max-side  长边上限，默认不缩（loader 自己裁/缩到 --resolution）
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
    ap.add_argument("--n-views", type=int, default=24)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--no-undistort", action="store_true")
    ap.add_argument("--max-side", type=int, default=0, help="0 = 保持原分辨率")
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

    extrinsics, intrinsics, records = [], [], []
    crop_info = None
    for n, i in enumerate(picked):
        fr = frames[i]
        src = raw / fr["file_path"].replace("./", "")
        if not src.is_file():
            sys.exit(f"❌ 图片缺失: {src}")

        with Image.open(src) as im:
            arr = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        h, w = arr.shape[:2]
        if args.max_side and max(w, h) > args.max_side:
            s = args.max_side / max(w, h)
            w2, h2 = int(round(w * s)), int(round(h * s))
            arr = cv2.resize(arr, (w2, h2), interpolation=cv2.INTER_AREA)
            w, h = w2, h2

        K = np.array([[fr["fl_x"], 0.0, fr["cx"]],
                      [0.0, fr["fl_y"], fr["cy"]],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        if (w, h) != (fr["w"], fr["h"]):
            # EXIF 转置或预缩放后，按比例把内参归到实际像素（含 --max-side 的缩放）
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
                crop_info = dict(src_wh=[int(fr["w"]), int(fr["h"])],
                                 out_wh=[int(arr.shape[1]), int(arr.shape[0])],
                                 roi=[int(v) for v in roi],
                                 dist=[float(v) for v in dist])

        M = np.array(fr["transform_matrix"], dtype=np.float64)  # c2w, OpenGL
        c2w = M @ FLIP

        name = f"{n:03d}_{src.stem}"
        Image.fromarray(arr).save(out / "images" / f"{name}.png", optimize=False)
        extrinsics.append({"camera_id": name, "matrix": c2w.tolist()})
        intrinsics.append({"camera_id": name, "matrix": K.tolist()})
        records.append(dict(idx=int(i), src=src.name, out=f"{name}.png",
                            wh=[int(arr.shape[1]), int(arr.shape[0])]))

    (out / "prior_camera.json").write_text(json.dumps({
        "num_cameras": len(extrinsics),
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }, indent=2), encoding="utf-8")

    # 自检：相机中心到场景原点距离 + 相邻视角夹角。退化（"三张同一视角"）会立刻暴露。
    C = np.array([e["matrix"] for e in extrinsics])[:, :3, 3]
    Rs = np.array([e["matrix"] for e in extrinsics])[:, :3, :3]
    rad = np.linalg.norm(C, axis=1)

    def ang(a, b):
        return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))

    pairs = [ang(Rs[i], Rs[j]) for i in range(len(Rs)) for j in range(i + 1, len(Rs))]
    meta = dict(source=str(raw), n_frames_total=len(frames), n_views=len(picked),
                picked_idx=picked, undistort=not args.no_undistort, max_side=args.max_side,
                crop=crop_info,
                cam_radius=dict(min=float(rad.min()), max=float(rad.max()), mean=float(rad.mean())),
                pairwise_deg=dict(min=float(np.min(pairs)), median=float(np.median(pairs)),
                                  max=float(np.max(pairs))),
                frames=records)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ 输出 {out}")
    print(f"   🖼️  {len(picked)}/{len(frames)} 帧  K[0] fx={intrinsics[0]['matrix'][0][0]:.1f} "
          f"cx={intrinsics[0]['matrix'][0][2]:.1f} cy={intrinsics[0]['matrix'][1][2]:.1f}")
    if crop_info:
        print(f"   🩹 去畸变: {crop_info['src_wh']} → {crop_info['out_wh']} (roi={crop_info['roi']})")
    print(f"   📷 相机半径 {rad.min():.3f}~{rad.max():.3f} m (mean {rad.mean():.3f})")
    print(f"   🔭 视角两夹角 min {np.min(pairs):.1f}° / median {np.median(pairs):.1f}° / max {np.max(pairs):.1f}°")
    if np.median(pairs) < 2.0:
        print("   ⚠️ 夹角中位数 <2°，位姿疑似退化（全部相机几乎同向）", file=sys.stderr)


if __name__ == "__main__":
    main()
