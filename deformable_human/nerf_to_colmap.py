#!/usr/bin/env python3
# nerf_to_colmap.py — 手机自带位姿（nerfstudio 风格 transforms.json + pcd.ply）
# → Deformable-GS 可训练的 COLMAP 文本场景（跳过 COLMAP SfM）。
#
# 输入约定（华为手机导出的真实格式，已 QC 验证）：
#   - transforms.json: frames[].transform_matrix 是 c2w、OpenGL 相机约定
#     （y 上 z 朝后）→ COLMAP/OpenCV w2c = inv(c2w @ diag(1,-1,-1))
#   - 图像带畸变，OPENCV 模型 (k1,k2,p1,p2,k3) 作用于原图（投影 QC 验证对齐）
#   - pcd.ply 与 transforms.json 同一世界系（米制、重力对齐）
#
# 输出（COLMAP text 格式，Deformable-GS loader 支持 .txt 回退）：
#   <out>/images/%05d.jpg        去畸变+裁剪后的图像（getOptimalNewCameraMatrix）
#   <out>/sparse/0/cameras.txt   单 PINHOLE 相机（新内参）
#   <out>/sparse/0/images.txt    每帧 w2c（四元数 qw qx qy qz + 平移）
#   <out>/sparse/0/points3D.txt  pcd 点云（RGB，error=1.0，空 track）
#   <out>/qc/proj_phone_*.jpg    投影叠加 QC（人工目检用）
#
# 图像命名 %05d 与 03_colmap_pose.sh 一致（字典序排序），
# 因此 masks / test 划分等下游约定可原样复用。
import argparse
import json
import os
import sys

import cv2
import numpy as np
from plyfile import PlyData

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL c2w → OpenCV c2w


def rotmat2qvec(R):
    """3x3 旋转矩阵 → COLMAP 四元数 (qw, qx, qy, qz)。Shepperd 方法，数值稳定。"""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s,
                         (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.empty(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transforms", required=True, help="transforms.json 路径")
    ap.add_argument("--frames_dir", required=True,
                    help="去 HEIC 后的 jpg 目录（文件名 stem 与 transforms 对应）")
    ap.add_argument("--out_scene", required=True, help="输出场景目录")
    ap.add_argument("--ply", default=None, help="pcd.ply（默认取 transforms 同目录）")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="getOptimalNewCameraMatrix 的 alpha（0=裁剪黑边）")
    args = ap.parse_args()

    tj = json.load(open(args.transforms))
    frames = tj["frames"]
    ply_path = args.ply or os.path.join(os.path.dirname(
        os.path.abspath(args.transforms)), tj.get("ply_file_path", "./pcd.ply"))

    # 内参一致性检查
    KEYS = ("w", "h", "cx", "cy", "fl_x", "fl_y", "k1", "k2", "k3", "p1", "p2")
    k0 = frames[0]
    for f in frames:
        for k in KEYS:
            assert abs(f[k] - k0[k]) < 1e-6, f"内参不一致: {f['file_path']} 字段 {k}"
    K = np.array([[k0["fl_x"], 0, k0["cx"]],
                  [0, k0["fl_y"], k0["cy"]], [0, 0, 1]], np.float64)
    dist = np.array([k0["k1"], k0["k2"], k0["p1"], k0["p2"], k0["k3"]], np.float64)
    W, H = int(k0["w"]), int(k0["h"])

    # stem → frame；输出命名按 stem 字典序（与 03_colmap_pose.sh 同规则）
    by_stem = {os.path.splitext(os.path.basename(f["file_path"]))[0]: f
               for f in frames}
    stems = sorted(by_stem.keys())
    print(f"[nerf2colmap] {len(stems)} 帧, 原图 {W}x{H}, ply={ply_path}")

    os.makedirs(os.path.join(args.out_scene, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out_scene, "sparse", "0"), exist_ok=True)
    os.makedirs(os.path.join(args.out_scene, "qc"), exist_ok=True)

    # 新内参 + ROI（一次计算，全部帧共用）
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), args.alpha, (W, H))
    x0, y0, rw, rh = roi
    print(f"[nerf2colmap] 去畸变后 {rw}x{rh} (roi={roi}), newK="
          f" fx={newK[0,0]:.2f} fy={newK[1,1]:.2f} cx={newK[0,2]:.2f} cy={newK[1,2]:.2f}")

    # cameras.txt（单 PINHOLE，注意 cx/cy 要减去 ROI 偏移）
    ncx, ncy = newK[0, 2] - x0, newK[1, 2] - y0
    with open(os.path.join(args.out_scene, "sparse", "0", "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {rw} {rh} {newK[0,0]:.6f} {newK[1,1]:.6f} {ncx:.6f} {ncy:.6f}\n")

    # images.txt + 去畸变图像
    lines = ["# Image list with two lines of data per image:\n",
             "#   IMAGE_ID, QW QX QY QZ, TX TY TZ, CAMERA_ID, NAME\n",
             "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"]
    for idx, stem in enumerate(stems):
        fr = by_stem[stem]
        c2w = np.array(fr["transform_matrix"], np.float64)
        w2c = np.linalg.inv(c2w @ FLIP)
        q = rotmat2qvec(w2c[:3, :3])
        t = w2c[:3, 3]
        name = f"{idx:05d}.jpg"
        lines.append(f"{idx+1} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                     f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} 1 {name}\n")
        lines.append("\n")  # 空 2D 点行
        src = os.path.join(args.frames_dir, stem + ".jpg")
        img = cv2.imread(src)
        assert img is not None, f"读不到 {src}"
        und = cv2.undistort(img, K, dist, None, newK)
        und = und[y0:y0 + rh, x0:x0 + rw]
        cv2.imwrite(os.path.join(args.out_scene, "images", name), und,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        if idx % 30 == 0:
            print(f"  [{idx+1}/{len(stems)}] {name} <- {stem}")
    with open(os.path.join(args.out_scene, "sparse", "0", "images.txt"), "w") as f:
        f.writelines(lines)

    # points3D.txt（pcd 世界点云，与 transforms 同世界系）
    v = PlyData.read(ply_path)["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1)
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    with open(os.path.join(args.out_scene, "sparse", "0", "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i in range(len(pts)):
            f.write(f"{i+1} {pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f} "
                    f"{rgb[i,0]} {rgb[i,1]} {rgb[i,2]} 1.0\n")
    print(f"[nerf2colmap] points3D: {len(pts)} 点")

    # QC：投影叠加（去畸变后应无畸变 → dist=0；抽首/中/末三帧）
    for idx in sorted({0, len(stems) // 2, len(stems) - 1}):
        fr = by_stem[stems[idx]]
        c2w = np.array(fr["transform_matrix"], np.float64)
        w2c = np.linalg.inv(c2w @ FLIP)
        rvec, _ = cv2.Rodrigues(w2c[:3, :3])
        Kc = newK.copy()
        Kc[0, 2] -= x0
        Kc[1, 2] -= y0
        uv, _ = cv2.projectPoints(pts.astype(np.float64), rvec,
                                  w2c[:3, 3], Kc, np.zeros(5))
        uv = uv.reshape(-1, 2)
        Xc = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
        inside = (Xc[:, 2] > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < rw) \
            & (uv[:, 1] >= 0) & (uv[:, 1] < rh)
        img = cv2.imread(os.path.join(args.out_scene, "images", f"{idx:05d}.jpg"))
        sel = np.nonzero(inside)[0]
        if len(sel) > 20000:
            sel = sel[np.random.choice(len(sel), 20000, replace=False)]
        for i in sel:
            u, vv = int(uv[i, 0]), int(uv[i, 1])
            img[vv, u] = (0, 0, 255)
        cv2.imwrite(os.path.join(args.out_scene, "qc", f"proj_phone_{idx:05d}.jpg"),
                    img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  [qc] frame {idx:05d}: 画面内点 {inside.mean()*100:.1f}%")

    print(f"[nerf2colmap] ✅ 完成 -> {args.out_scene}")


if __name__ == "__main__":
    main()
