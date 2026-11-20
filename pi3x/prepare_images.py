#!/usr/bin/env python3
"""Remy 采集包（transforms.json + image/*.heic + pcd.ply）→ Pi3X 输入。

做的事（与 som_human/01_convert_capture.py 同口径）：
  1. HEIC 解码（pillow-heif），帧按 file_path 排序
  2. cv2.getOptimalNewCameraMatrix(alpha=0) 去畸变 + 裁 ROI，K' 主点平移
  3. 存 %05d.png（排序后顺序 == conditions 数组顺序，Pi3X loader 按文件名 sorted 读取，两者一致）
  4. GT 位姿：c2w(OpenCV) = M @ diag(1,-1,-1,1)（实测判定，勿凭直觉改）
  5. 写 conditions.npz（intrinsics (N,3,3) / poses (N,4,4)，供 Pi3X 条件注入与 04 评测共用）

Env vars:
  RAW_DIR   采集包路径（必填）
  OUT_DIR   场景输出目录（必填）
"""
import json
import os
import sys

import cv2
import numpy as np

RAW_DIR = os.environ.get("RAW_DIR", "")
OUT_DIR = os.environ.get("OUT_DIR", "")

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL -> OpenCV（右乘 = 翻 y/z 列）


def main():
    if not RAW_DIR or not os.path.isfile(os.path.join(RAW_DIR, "transforms.json")):
        sys.exit("❌ RAW_DIR 未设或缺 transforms.json: %s" % RAW_DIR)
    if not OUT_DIR:
        sys.exit("❌ OUT_DIR 未设")

    from PIL import Image
    from pillow_heif import register_heif_opener
    register_heif_opener()

    with open(os.path.join(RAW_DIR, "transforms.json")) as f:
        meta = json.load(f)
    frames = sorted(meta["frames"], key=lambda fr: fr["file_path"])
    n = len(frames)
    print("🚀 [prepare] %d frames（按 file_path 排序）" % n)

    # 内参全序列一致（som_human 实测逐帧相同），取 frame0
    f0 = frames[0]
    w, h = int(f0["w"]), int(f0["h"])
    K = np.array([[f0["fl_x"], 0, f0["cx"]],
                  [0, f0["fl_y"], f0["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array([f0["k1"], f0["k2"], f0["p1"], f0["p2"], f0["k3"]], dtype=np.float64)
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
    rx, ry, rw, rh = [int(v) for v in roi]
    if rw <= 0 or rh <= 0:
        sys.exit("❌ getOptimalNewCameraMatrix 返回空 ROI，畸变参数可能异常")
    newK[0, 2] -= rx
    newK[1, 2] -= ry
    print("  📐 undistort: %dx%d -> crop %dx%d, K' fx=%.1f fy=%.1f cx=%.1f cy=%.1f"
          % (w, h, rw, rh, newK[0, 0], newK[1, 1], newK[0, 2], newK[1, 2]))

    img_dir = os.path.join(OUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)
    names, poses = [], []
    for i, fr in enumerate(frames):
        dst = os.path.join(img_dir, "%05d.png" % i)
        if not os.path.exists(dst):
            im = Image.open(os.path.join(RAW_DIR, fr["file_path"]))
            if im.size != (w, h):
                sys.exit("❌ %s 解码尺寸 %s 与 json %dx%d 不符" % (fr["file_path"], im.size, w, h))
            arr = np.asarray(im.convert("RGB"))
            und = cv2.undistort(arr, K, dist, None, newK)
            Image.fromarray(und[ry:ry + rh, rx:rx + rw]).save(dst)
        names.append("%05d.png" % i)
        M = np.array(fr["transform_matrix"], dtype=np.float64)
        poses.append(M @ FLIP)  # c2w, OpenCV
        if (i + 1) % 40 == 0 or i == n - 1:
            print("  🖼️  decoded %d/%d" % (i + 1, n))

    poses = np.stack(poses)                       # (N,4,4) c2w OpenCV
    intrinsics = np.repeat(newK[None], n, axis=0)  # (N,3,3)
    np.savez(os.path.join(OUT_DIR, "conditions.npz"),
             intrinsics=intrinsics, poses=poses)
    with open(os.path.join(OUT_DIR, "gt_meta.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({
            "raw_dir": RAW_DIR, "n_frames": n, "frame_names": names,
            "orig_wh": [w, h], "crop_xywh": [rx, ry, rw, rh],
            "K_undistorted": newK.tolist(),
            "pose_convention": "c2w OpenCV = transform_matrix @ diag(1,-1,-1,1)",
        }, f, ensure_ascii=False, indent=2)
    print("✅ conditions.npz + gt_meta.json + %d 张去畸变 PNG -> %s" % (n, img_dir))
    print("🎉 Done.")


if __name__ == "__main__":
    main()
