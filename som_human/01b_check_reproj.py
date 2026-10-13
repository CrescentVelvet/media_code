#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""01b_check_reproj.py — 采集包转换后的位姿/内参质量诊断（01 的配套检查）

背景：double_wave 三轮训练（v1 216k 点 / v3 147 万点 / 纯静态消融）test PSNR
全部停在 13~14，且静态消融里背景建筑仍呈方向性涂抹条纹 → 怀疑对象从算法
转向数据链路（App 位姿精度 / 去畸变内参更新）。本脚本给出两个证据：
  1. 数值：把 pcd 点云投影到若干帧，与落点处图像颜色求平均色差（静态结构上
     位姿/内参正确 ⇒ 色差小；错位 ⇒ 色差大、且错位方向可视化）。
  2. 图像：投影点叠加 PNG（点带 pcd 原色），肉眼可直接看「点是否坐在正确
     的结构上」。

注意：pcd 是采集时刻的快照，会动的人/物在其它帧必然错位——判读时以
建筑/地面/静止植被等静态结构为准（叠加图里人身上的错位不算数）。

用法（在 deformable_human env 内）：
  python 01b_check_reproj.py <scene_dir> <raw_pcd.ply> [n_frames] [out_dir]
  scene_dir  : 01_convert_capture 产出的 colmap_scene（含 sparse/0/*.txt 与 images/）
  raw_pcd    : 采集包里的原始 pcd.ply（世界系与 transforms.json 一致）
输出：
  <out_dir>/reproj_<imgname>  叠加图（默认可先拷到 Windows 侧肉眼检查）
  stdout 打印每帧 inlier 比例与平均 RGB 色差（0~255 口径，<15 大致可信、
  >30 基本可判错位；受光照/曝光影响，以同帧静态结构横向比较为准）。
"""
import os
import sys

import numpy as np
from PIL import Image

# colmap_loader 在官方仓里（DG_DIR 由 _env.sh 注入；脚本直跑时兜底）
sys.path.insert(0, os.environ.get("DG_DIR", os.path.expanduser(
    "~/repos/Deformable-3D-Gaussians")))
from scene.colmap_loader import (read_extrinsics_text, read_intrinsics_text,
                                 qvec2rotmat)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    scene_dir, pcd_path = sys.argv[1], sys.argv[2]
    n_check = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    out_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(
        scene_dir, "reproj_check")
    os.makedirs(out_dir, exist_ok=True)

    from plyfile import PlyData
    v = PlyData.read(pcd_path)["vertex"]
    P = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    C = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float64)

    ext = read_extrinsics_text(os.path.join(scene_dir, "sparse/0/images.txt"))
    intr = read_intrinsics_text(os.path.join(scene_dir, "sparse/0/cameras.txt"))
    ids = sorted(ext.keys())
    pick = [ids[int(i * (len(ids) - 1) / max(1, n_check - 1))] for i in range(n_check)]

    print("[01b] %d points, checking %d/%d frames" % (len(P), n_check, len(ids)))
    for img_id in pick:
        e = ext[img_id]
        R = qvec2rotmat(e.qvec)
        t = e.tvec.astype(np.float64)
        cam = intr[e.camera_id]
        fx, fy, cx, cy = cam.params
        Pc = (R @ P.T).T + t
        z = Pc[:, 2]
        ok = z > 0.1
        u = fx * Pc[:, 0] / np.maximum(z, 1e-9) + cx
        w = fy * Pc[:, 1] / np.maximum(z, 1e-9) + cy
        ok &= (u >= 0) & (u < cam.width) & (w >= 0) & (w < cam.height)

        im = Image.open(os.path.join(scene_dir, "images", e.name)).convert("RGB")
        a = np.asarray(im).astype(np.float64)
        ui, wi = u[ok].astype(int), w[ok].astype(int)
        diff = np.abs(a[wi, ui] - C[ok]).mean()
        print("[01b] %-12s inlier=%6d/%d  mean|dRGB|=%.1f" % (
            e.name, int(ok.sum()), len(P), diff))

        overlay = a.copy()
        overlay[wi, ui] = C[ok]
        Image.fromarray(overlay.astype(np.uint8)).save(
            os.path.join(out_dir, "reproj_" + os.path.basename(e.name)))
    print("[01b] overlays -> %s" % out_dir)


if __name__ == "__main__":
    main()
