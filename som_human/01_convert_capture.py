# -*- coding: utf-8 -*-
"""01_convert_capture.py — 手机采集格式（Remy/Harmony）→ D-3DGS COLMAP 场景转换器。

输入（峰哥 2026-10-13 提供的手部微动样例格式）：
  RAW_DIR/
    transforms.json   # 逐帧 w/h/cx/cy/fl_x/fl_y/k1..k3/p1/p2 + transform_matrix
    image/*.heic      # 逐帧 HEIC（文件名是时间戳）
    pcd.ply           # 场景点云（binary_little_endian，带 RGB）

位姿约定（2026-10-13 实测投影正深度率判定）：
  transform_matrix = **c2w，OpenGL 相机系**（y 上 z 朝后）。
  → OpenCV w2c = inv(M @ diag(1,-1,-1,1))，COLMAP 存的就是这个 w2c。

输出（D-3DGS readColmapSceneInfo 直接可读，txt 变体）：
  OUT_DIR/
    images/%05d.png           # 去畸变 + ROI 裁剪（时间升序，int 文件名 → fid）
    sparse/0/cameras.txt      # 单个 PINHOLE 相机（去畸变后内参）
    sparse/0/images.txt       # 逐帧 w2c 四元数
    sparse/0/points3D.txt     # pcd 过滤离群点后写入（空 track）

注意：D-3DGS 只支持 PINHOLE/SIMPLE_PINHOLE（光栅化不建模畸变），
本脚本用 cv2.getOptimalNewCameraMatrix(alpha=0) 去畸变并裁掉黑边。
"""
import argparse
import json
import os
import sys

import numpy as np


def rotmat2qvec(R):
    """旋转矩阵 → COLMAP 四元数 (qw,qx,qy,qz)。与 read_write_model.rotmat2qvec 同口径。"""
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = R.flat
    K = np.array([
        [rxx - ryy - rzz, 0, 0, 0],
        [0, ryy - rxx - rzz, 0, 0],
        [0, 0, rzz - rxx - ryy, 0],
        [0, 0, 0, rxx + ryy + rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec = -qvec
    return qvec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="采集原始目录（transforms.json/image/pcd.ply）")
    ap.add_argument("--out_dir", required=True, help="输出 COLMAP 场景目录")
    ap.add_argument("--keep_frac", type=float, default=0.90,
                    help="点云按「到相机中心距离」保留的比例（去掉离群飞点）")
    args = ap.parse_args()

    import cv2
    from PIL import Image
    from pillow_heif import register_heif_opener
    from plyfile import PlyData
    register_heif_opener()

    raw, out = args.raw_dir, args.out_dir
    with open(os.path.join(raw, "transforms.json")) as f:
        meta = json.load(f)
    frames = sorted(meta["frames"], key=lambda f: f["file_path"])
    n = len(frames)
    print("[01] %d frames, sorting by file_path (timestamp)" % n)

    # ── 内参（全序列一致，2026-10-13 实测逐帧相同）─────────────────────────
    f0 = frames[0]
    w, h = int(f0["w"]), int(f0["h"])
    K = np.array([[f0["fl_x"], 0, f0["cx"]],
                  [0, f0["fl_y"], f0["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array([f0["k1"], f0["k2"], f0["p1"], f0["p2"], f0["k3"]], dtype=np.float64)
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
    rx, ry, rw, rh = [int(v) for v in roi]
    if rw <= 0 or rh <= 0:
        print("ERROR: getOptimalNewCameraMatrix 返回空 ROI，畸变参数可能异常", file=sys.stderr)
        sys.exit(1)
    # 裁剪后主点平移
    newK[0, 2] -= rx
    newK[1, 2] -= ry
    print("[01] undistort: %dx%d -> crop %dx%d, K' fx=%.1f fy=%.1f cx=%.1f cy=%.1f"
          % (w, h, rw, rh, newK[0, 0], newK[1, 1], newK[0, 2], newK[1, 2]))

    # ── 逐帧：解码 HEIC → 去畸变 → 裁剪 → PNG ─────────────────────────────
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)
    for i, fr in enumerate(frames):
        dst = os.path.join(img_dir, "%05d.png" % i)
        if os.path.exists(dst):
            continue
        im = Image.open(os.path.join(raw, fr["file_path"]))
        if im.size != (w, h):
            print("ERROR: %s 解码尺寸 %s 与 json %dx%d 不符" % (fr["file_path"], im.size, w, h),
                  file=sys.stderr)
            sys.exit(1)
        arr = np.asarray(im.convert("RGB"))
        und = cv2.undistort(arr, K, dist, None, newK)
        crop = und[ry:ry + rh, rx:rx + rw]
        Image.fromarray(crop).save(dst)
        if (i + 1) % 40 == 0 or i == n - 1:
            print("[01] decoded %d/%d" % (i + 1, n))

    # ── 位姿：c2w(OpenGL) → w2c(OpenCV)，写 COLMAP txt ────────────────────
    FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
    sparse_dir = os.path.join(out, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    cam_positions = []
    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as fc, \
            open(os.path.join(sparse_dir, "images.txt"), "w") as fi:
        fc.write("# Camera list with one line of data per camera:\n")
        fc.write("1 PINHOLE %d %d %.6f %.6f %.6f %.6f\n"
                 % (rw, rh, newK[0, 0], newK[1, 1], newK[0, 2], newK[1, 2]))
        fi.write("# Image list with two lines of data per image:\n")
        for i, fr in enumerate(frames):
            M = np.array(fr["transform_matrix"], dtype=np.float64)  # c2w, OpenGL
            M_cv = M @ FLIP                                          # c2w, OpenCV
            M_w2c = np.linalg.inv(M_cv)
            R, t = M_w2c[:3, :3], M_w2c[:3, 3]
            q = rotmat2qvec(R)
            cam_positions.append(M_cv[:3, 3])
            fi.write("%d %.9f %.9f %.9f %.9f %.9f %.9f %.9f 1 %05d.png\n"
                     % (i + 1, q[0], q[1], q[2], q[3], t[0], t[1], t[2], i))
            fi.write("\n")  # 空 points2D 行
    cam_positions = np.array(cam_positions)

    # ── 点云：读 pcd.ply → 离群点过滤 → points3D.txt ──────────────────────
    v = PlyData.read(os.path.join(raw, "pcd.ply"))["vertex"]
    P = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    C = np.stack([v["red"], v["green"], v["blue"]], axis=1)
    cen = cam_positions.mean(0)
    d = np.linalg.norm(P - cen, axis=1)
    thr = np.quantile(d, args.keep_frac)
    keep = d <= thr
    Pk, Ck = P[keep], C[keep]
    print("[01] points: %d -> %d (keep_frac=%.2f, dist<=%.2f of cam centroid)"
          % (len(P), len(Pk), args.keep_frac, thr))
    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as fp:
        fp.write("# 3D point list with one line of data per point:\n")
        for j in range(len(Pk)):
            fp.write("%d %.6f %.6f %.6f %d %d %d 1.0\n"
                     % (j + 1, Pk[j, 0], Pk[j, 1], Pk[j, 2], Ck[j, 0], Ck[j, 1], Ck[j, 2]))

    # ── 自检：用最终 w2c + 新内参把保留点投回图像 ─────────────────────────
    print("[01] verify projection with stored params:")
    ok = True
    for i in range(0, n, max(1, n // 8)):
        M = np.array(frames[i]["transform_matrix"], dtype=np.float64) @ FLIP
        M_w2c = np.linalg.inv(M)
        Ph = np.concatenate([Pk[::50], np.ones((len(Pk[::50]), 1))], axis=1)
        Pc = (M_w2c @ Ph.T).T
        z = Pc[:, 2]
        front = z > 0.01
        u = newK[0, 0] * Pc[:, 0] / np.maximum(z, 1e-9) + newK[0, 2]
        vv = newK[1, 1] * Pc[:, 1] / np.maximum(z, 1e-9) + newK[1, 2]
        inside = front & (u >= 0) & (u < rw) & (vv >= 0) & (vv < rh)
        print("  frame %3d: depth>0 %.3f  in-image %.3f" % (i, front.mean(), inside.mean()))
        if front.mean() < 0.5:
            ok = False
    if not ok:
        print("ERROR: 投影自检失败（位姿约定可能变了），不要进训练", file=sys.stderr)
        sys.exit(1)
    print("[01] ✅ done -> %s" % out)


if __name__ == "__main__":
    main()
