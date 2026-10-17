#!/usr/bin/env python3
"""Gen3R 输出 → COLMAP 场景（供 3DGS 训练直接消费）。

Gen3R 只产出「点云 + 相机 + 视频」，不产出 Gaussian；本脚本把它转成
gaussian-splatting 官方 train.py 认的格式：

    <OUT_DIR>/
      images/0000.png ...            # 从 rgb.mp4 抽的 49 帧（560x560）
      sparse/0/cameras.txt           # PINHOLE，逐帧内参
      sparse/0/images.txt            # w2c 四元数 + 平移（Gen3R 已是 OpenCV w2c，无需翻系）
      sparse/0/points3D.txt          # pcds.ply 的点（带 RGB，作为 3DGS init）

注意：
- Gen3R 的 extrinsics 是 4x4 world→camera（OpenCV），与 COLMAP 的 qvec/tvec 同义，
  直接转四元数即可，不要取逆。
- 世界尺度由 Gen3R/VGGT 决定、单图任务无真实尺度；跨场景对齐需另做相似变换。
- 点云与相机同源（同一 VGGT head 输出），自洽；但不是真实观测，属生成几何。

Env vars:
  GEN3R_OUT   Gen3R 单次运行的产物目录（含 rgb.mp4 / cameras.json / pcds.ply）
  OUT_DIR     输出场景目录（默认 <GEN3R_OUT>/colmap_scene）
  MAX_POINTS  点云上限（默认 200000，3DGS 初始化点太多会拖慢稠密化）
"""
import json
import os
import sys

import numpy as np


def rotmat_to_qvec(R):
    """旋转矩阵 → COLMAP 四元数 (qw, qx, qy, qz)。数值不稳时走最大分量分支。"""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def extract_frames(mp4, out_dir, limit=None):
    import imageio.v2 as imageio
    os.makedirs(out_dir, exist_ok=True)
    rd = imageio.get_reader(mp4)
    names = []
    for i, frame in enumerate(rd):
        if limit is not None and i >= limit:
            break
        name = "{:04d}.png".format(i)
        imageio.imwrite(os.path.join(out_dir, name), frame[..., :3])
        names.append(name)
    rd.close()
    return names


def write_cameras(path, Ks, w, h):
    with open(path, "w", newline="\n") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, K in enumerate(Ks):
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            f.write("{} PINHOLE {} {} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                i + 1, int(w), int(h), fx, fy, cx, cy))


def write_images(path, extrinsics, names):
    with open(path, "w", newline="\n") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, (E, name) in enumerate(zip(extrinsics, names)):
            E = np.asarray(E, dtype=np.float64)
            assert E.shape == (4, 4), "extrinsics 需 4x4，实得 {}".format(E.shape)
            q = rotmat_to_qvec(E[:3, :3])
            t = E[:3, 3]
            f.write("{} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {} {}\n\n".format(
                i + 1, q[0], q[1], q[2], q[3], t[0], t[1], t[2], i + 1, name))


def write_points(path, ply, max_points):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(ply)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if pts.shape[0] == 0:
        sys.exit("❌ 点云为空: {}".format(ply))
    if pts.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts.shape[0], max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    rgb = np.clip(cols * 255.0, 0, 255).astype(np.uint8) if cols.size else None
    with open(path, "w", newline="\n") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i, p in enumerate(pts):
            c = rgb[i] if rgb is not None else (128, 128, 128)
            f.write("{} {:.6f} {:.6f} {:.6f} {} {} {} 0.0\n".format(
                i + 1, p[0], p[1], p[2], int(c[0]), int(c[1]), int(c[2])))
    return pts.shape[0]


def main():
    src = os.environ.get("GEN3R_OUT", "")
    if not src or not os.path.isdir(src):
        sys.exit("❌ GEN3R_OUT 未设置或不是目录: {}".format(src))
    out = os.environ.get("OUT_DIR", os.path.join(src, "colmap_scene"))
    max_points = int(os.environ.get("MAX_POINTS", "200000"))

    cams = json.load(open(os.path.join(src, "cameras.json")))
    extrinsics = np.array(cams["extrinsics"], dtype=np.float64)  # [F,4,4] w2c
    Ks = np.array(cams["intrinsics"], dtype=np.float64)          # [F,3,3]

    img_dir = os.path.join(out, "images")
    names = extract_frames(os.path.join(src, "rgb.mp4"), img_dir)

    n = min(len(names), len(extrinsics))
    names = names[:n]
    extrinsics = extrinsics[:n]
    Ks = Ks[:n]

    # 帧尺寸从实际图片取（infer.py 固定 560x560，但别写死）
    from PIL import Image
    with Image.open(os.path.join(img_dir, names[0])) as im:
        w, h = im.size
    print("🖼️  frames={} size={}x{}".format(n, w, h))

    sparse = os.path.join(out, "sparse", "0")
    os.makedirs(sparse, exist_ok=True)

    # PINHOLE 需要 w/h：临时挂属性给 write_cameras（保持函数签名简单）
    write_cameras(os.path.join(sparse, "cameras.txt"), Ks, w, h)
    write_images(os.path.join(sparse, "images.txt"), extrinsics, names)
    npts = write_points(os.path.join(sparse, "points3D.txt"),
                        os.path.join(src, "pcds.ply"), max_points)

    print("✅ COLMAP scene ready: {}".format(out))
    print("   images={}  points={}  cameras={}".format(len(names), npts, len(Ks)))


if __name__ == "__main__":
    main()
