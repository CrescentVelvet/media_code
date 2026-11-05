#!/usr/bin/env python3
"""画 DA3 位姿轨迹 vs COLMAP 参考（02 的验收可视化：漂移/跳变一目了然）。

输出:
  <OUTPUT_DIR>/viz/trajectory_3d.png   两条轨迹 + 误差棒
  <OUTPUT_DIR>/viz/pos_err_per_view.png 逐视角位置误差

Env vars:
  DA3_INFER_DIR   02 的输出目录（读 exports/mini_npz/results.npz）
  COLMAP_REF_DIR  COLMAP sparse/0 目录（读 images.txt）
  OUTPUT_DIR      可选，默认 <DA3_INFER_DIR>/viz
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401（注册 3d 投影）

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_da3_inference import camera_centers, parse_colmap_images, umeyama_sim3  # noqa: E402


def main():
    infer_dir = os.environ["DA3_INFER_DIR"]
    ref_dir = os.environ["COLMAP_REF_DIR"]
    out_dir = os.environ.get("OUTPUT_DIR", os.path.join(infer_dir, "viz"))
    os.makedirs(out_dir, exist_ok=True)

    z = np.load(os.path.join(infer_dir, "exports", "mini_npz", "results.npz"))
    w2c = z["extrinsics"]  # (N,4,4) DA3 w2c
    centers = camera_centers(np.asarray(w2c))

    ref = parse_colmap_images(os.path.join(ref_dir, "images.txt"))
    # mini_npz 的视角顺序 = 02 输入图像的排序（文件名序）
    import glob
    images = sorted(glob.glob(os.path.join(os.environ.get("INPUT_DIR", ""), "*.jpg"))) if os.environ.get("INPUT_DIR") else None
    if images is None:
        # 无 INPUT_DIR 时用 images.bin 的名字顺序（pycolmap 读取，与导出顺序一致）
        import pycolmap
        recon = pycolmap.Reconstruction(infer_dir)
        names = [recon.images[i + 1].name for i in range(len(centers))]
    else:
        names = [os.path.basename(p) for p in images]

    ref_centers = np.array([ref[n] for n in names])
    s, r, t = umeyama_sim3(centers, ref_centers)
    aligned = s * (centers @ r.T) + t
    err = np.linalg.norm(aligned - ref_centers, axis=1)

    print(f"📊 Sim(3) scale={s:.4f}  pos_err mean={err.mean()*1000:.1f}mm "
          f"p50={np.median(err)*1000:.1f}mm p95={np.percentile(err, 95)*1000:.1f}mm max={err.max()*1000:.1f}mm")

    # 3D 轨迹
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ref_centers[:, 0], ref_centers[:, 1], ref_centers[:, 2],
            "o-", color="#2196F3", ms=3, lw=1, label=f"COLMAP ref")
    ax.plot(aligned[:, 0], aligned[:, 1], aligned[:, 2],
            "x--", color="#F44336", ms=3, lw=1, label=f"DA3 aligned (mean err {err.mean()*1000:.1f}mm)")
    ax.scatter(*ref_centers[0], color="#2196F3", s=80, marker="s", label="start")
    ax.scatter(*ref_centers[-1], color="#4CAF50", s=80, marker="^", label="end")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("DA3 vs COLMAP camera trajectory")
    ax.legend()
    p = os.path.join(out_dir, "trajectory_3d.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"🖼️ {p}")

    # 逐视角误差
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(err * 1000, "-o", ms=3)
    ax.axhline(err.mean() * 1000, color="r", ls="--", label=f"mean {err.mean()*1000:.1f}mm")
    ax.set_xlabel("view index (sorted by filename)")
    ax.set_ylabel("position err (mm)")
    ax.set_title("Per-view position error after Sim(3) align")
    ax.legend()
    p = os.path.join(out_dir, "pos_err_per_view.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"🖼️ {p}")


if __name__ == "__main__":
    main()
