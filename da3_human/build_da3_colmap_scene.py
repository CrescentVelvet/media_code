#!/usr/bin/env python3
"""把 DA3 colmap 导出整理成原版 3DGS 可读的 COLMAP 场景目录。

DA3 的 export_to_colmap 把 cameras/images/points3D(.bin) 直接写在 export_dir 根；
原版 3DGS 的 read_write_model 需要 <scene>/images + <scene>/sparse/0/*.bin。
本脚本：
  1. <SCENE_DIR>/images  -> 软链到原图像目录（避免复制 125 张大图）
  2. <SCENE_DIR>/sparse/0 -> 复制 DA3 导出的 cameras.bin / images.bin
  3. points3D.bin 若超过 MAX_INIT_POINTS（DA3 导出 1429 万点，直接初始化 3DGS
     会把训练拖垮），做体素降采样后重写（空 track；3DGS train.py 只用 xyz+rgb，
     points2D 的 track 引用不参与训练，悬空安全）

Env vars:
  DA3_INFER_DIR   DA3 colmap 导出目录（02 的 OUTPUT_DIR）
  SCENE_DIR       目标场景目录
  IMAGES_DIR      原始图像目录（软链目标）
  MAX_INIT_POINTS 初始点上限，默认 1000000
"""
import os
import shutil
import struct
import sys

import numpy as np

COLMAP_BINS = ["cameras.bin", "images.bin"]  # points3D.bin 单独处理（可能降采样）


def read_points3d(infer_dir):
    """用 pycolmap 读 DA3 导出的 reconstruction（写出方同库解析，格式争议最小）"""
    import pycolmap

    recon = pycolmap.Reconstruction(infer_dir)
    keys = sorted(recon.points3D.keys())
    pts = np.array([recon.points3D[k].xyz for k in keys], dtype=np.float64)
    colors = np.array([recon.points3D[k].color for k in keys], dtype=np.uint8)
    return pts, colors


def write_points3d_bin(path, pts, colors):
    """写无 track 的 points3D.bin（id 从 1 连续编号）"""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for i in range(len(pts)):
            f.write(struct.pack("<q", i + 1))
            f.write(struct.pack("<3d", *pts[i]))
            f.write(struct.pack("<3B", *colors[i]))
            f.write(struct.pack("<d", 0.0))  # error
            f.write(struct.pack("<Q", 0))     # track length = 0


def voxel_downsample(pts, colors, target):
    """体素降采样到 target 附近。

    ⚠️ 深度反投影点云全在表面上（2D 流形），占用体素数 ∝ 1/voxel²（面积缩放），
    体积公式估算初值会过粗 10×+；且初始就可能低于 target，必须双向调整。
    每个体素取均值（float 累加避免 uint8 溢出）。
    """
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    extent = np.maximum(hi - lo, 1e-6)
    voxel = float((np.prod(extent) / target) ** (1.0 / 3.0))  # 初值（会偏粗）
    inv = counts = None
    for _ in range(15):
        keys = np.floor((pts - lo) / voxel).astype(np.int64)
        assert keys.max() < (1 << 20), "体素网格超出 20bit 编码范围"
        code = (keys[:, 0] << 42) | (keys[:, 1] << 21) | keys[:, 2]
        uniq, inv, counts = np.unique(code, return_inverse=True, return_counts=True)
        n = len(uniq)
        if n > target:
            voxel *= max((n / target) ** 0.5, 1.05)  # 面积缩放收紧
        elif n < target * 0.85:
            voxel *= 0.9  # 过粗则放松
        else:
            break
    s = np.zeros((n, 3), dtype=np.float64)
    c = np.zeros((n, 3), dtype=np.float64)
    np.add.at(s, inv, pts)
    np.add.at(c, inv, colors.astype(np.float64))
    return s / counts[:, None], (c / counts[:, None]).round().astype(np.uint8)


def main():
    infer_dir = os.environ["DA3_INFER_DIR"]
    scene_dir = os.environ["SCENE_DIR"]
    images_dir = os.environ["IMAGES_DIR"]

    for name in COLMAP_BINS + ["points3D.bin"]:
        src = os.path.join(infer_dir, name)
        if not os.path.isfile(src):
            sys.exit(f"❌ 缺少 DA3 导出文件: {src}")
    if not os.path.isdir(images_dir):
        sys.exit(f"❌ 原图像目录不存在: {images_dir}")

    os.makedirs(scene_dir, exist_ok=True)

    # 1) images 软链（已存在且正确则跳过；错误链接会重建）
    link = os.path.join(scene_dir, "images")
    if os.path.islink(link) and os.path.realpath(link) == os.path.realpath(images_dir):
        print(f"⏭️ images 软链已存在: {link}")
    else:
        if os.path.lexists(link):
            sys.exit(f"❌ {link} 已存在且不是指向目标的软链，请先人工确认后删除")
        os.symlink(images_dir, link)
        print(f"🔗 images -> {images_dir}")

    # 2) sparse/0 搬运 + points3D 降采样
    sparse0 = os.path.join(scene_dir, "sparse", "0")
    os.makedirs(sparse0, exist_ok=True)
    for name in COLMAP_BINS:
        src = os.path.join(infer_dir, name)
        dst = os.path.join(sparse0, name)
        if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
            print(f"⏭️ {name} 已就位（大小一致，跳过）")
            continue
        # 统一 copy 不 move：02 的导出目录是原始记录，04 只读副本
        shutil.copy2(src, dst)
        print(f"📦 {name}: {os.path.getsize(dst) / 1e6:.1f} MB")

    max_init = int(os.environ.get("MAX_INIT_POINTS", "500000"))
    dst_p3d = os.path.join(sparse0, "points3D.bin")
    pts, colors = read_points3d(infer_dir)
    print(f"📊 DA3 点云: {len(pts):,} 点")
    if len(pts) > max_init:
        pts, colors = voxel_downsample(pts, colors, max_init)
        print(f"✂️ 体素降采样 -> {len(pts):,} 点 (MAX_INIT_POINTS={max_init:,})")
    write_points3d_bin(dst_p3d, pts, colors)
    print(f"📦 points3D.bin: {os.path.getsize(dst_p3d) / 1e6:.1f} MB ({len(pts):,} 点, 空 track)")

    # 3) pycolmap 4.x 可能额外写 rigs.bin/frames.bin，原版 3DGS 解析器只认
    #    上面三个固定文件名，一并复制无碍
    for extra in ("rigs.bin", "frames.bin"):
        src = os.path.join(infer_dir, extra)
        if os.path.isfile(src):
            dst = os.path.join(sparse0, extra)
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)
                print(f"📦 {extra} (pycolmap 4.x 附加文件)")

    n_images = len([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
    print(f"✅ 场景就绪: {scene_dir} (images={n_images})")


if __name__ == "__main__":
    main()
