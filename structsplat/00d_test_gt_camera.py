#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00d_test_gt_camera.py — GT 相机转置变换的纯几何单测（不需要权重 / GPU）

验证 load_gt_cameras 在「竖图 → 横图转置」时给出的 (w2c, K) 是否真的
能把同一个 3D 点投到转置后图像的对应像素上。

判据（非常重要）：转置是
    t[c, i, j] = img[c, j, i]
即新图 (y'=i, x'=j) 取自原图 (y=j, x=i)，所以
    u' = v,  v' = u
做法：在原始图像上取一个像素 (u, v)，反投影成 3D 点（固定深度 Z），
再分别用「原始 K + 原始 c2w」和「load_gt_cameras 返回的 K + w2c」投影，
检查后者是否落在 (v, u) 而不是 (u, v)。两种口径（有/无轴交换）都算一遍，
证明只有加了 M 轴交换的那一版才对。

用法：python 00d_test_gt_camera.py
"""

import importlib.util
import json
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_infer_module():
    spec = importlib.util.spec_from_file_location("inf", os.path.join(HERE, "infer.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_fake_transforms(tmpdir, W0, H0, fx, fy, cx, cy, R, t):
    """写一个单帧 transforms.json（R 为 3x3 c2w 旋转，t 为 c2w 平移）。"""
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = t
    meta = {
        "w": W0, "h": H0,
        "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
        "camera_model": "OPENCV",
        "frames": [{
            "file_path": "img0.heic",
            "w": W0, "h": H0,
            "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
            "transform_matrix": c2w.tolist(),
        }],
    }
    p = os.path.join(tmpdir, "transforms.json")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f)
    return p


def project(K, w2c, X):
    """X: (3,) 世界点 → 像素 (u, v)。"""
    Xh = np.concatenate([X, [1.0]])
    Xc = w2c @ Xh
    x, y, z = Xc[0], Xc[1], Xc[2]
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return u, v, z


def main():
    m = load_infer_module()
    rng = np.random.default_rng(0)

    W0, H0 = 1440, 1920          # 真实数据集的竖图尺寸
    fx, fy, cx, cy = 1359.49, 1361.67, 716.07, 952.97
    new_h, new_w = 294, 518      # 网络输入（横图）

    # 一个「斜着看」的 c2w：绕 X 30°、绕 Y 20°，相机在 z=+2 处朝 -z
    ax, ay = np.deg2rad(30), np.deg2rad(20)
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    R = Ry @ Rx
    t = np.array([0.1, -0.2, 2.0])

    with tempfile.TemporaryDirectory() as td:
        tp = build_fake_transforms(td, W0, H0, fx, fy, cx, cy, R, t)
        fake_img = os.path.join(td, "img0.heic")
        open(fake_img, "wb").close()

        w2cs, Ks, matched = m.load_gt_cameras(tp, [fake_img], new_h, new_w)
        assert len(matched) == 1, f"未匹配到图: {matched}"

    K_new = Ks[0].numpy().astype(np.float64)
    w2c_new = w2cs[0].numpy().astype(np.float64)

    K_orig = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    c2w_orig = np.eye(4)
    c2w_orig[:3, :3] = R
    c2w_orig[:3, 3] = t
    w2c_orig = np.linalg.inv(c2w_orig)

    print("=" * 66)
    print(" 转置后内参检查（期望 fx'≈fy*s_u, fy'≈fx*s_v）")
    s_u, s_v = new_w / H0, new_h / W0
    print(f"   s_u={s_u:.5f}  s_v={s_v:.5f}")
    print(f"   实测 K_new[0,0]={K_new[0,0]:.3f}  期望 fy*s_u={fy*s_u:.3f}")
    print(f"   实测 K_new[1,1]={K_new[1,1]:.3f}  期望 fx*s_v={fx*s_v:.3f}")
    print(f"   实测 K_new[0,2]={K_new[0,2]:.3f}  期望 cy*s_u={cy*s_u:.3f}")
    print(f"   实测 K_new[1,2]={K_new[1,2]:.3f}  期望 cx*s_v={cx*s_v:.3f}")

    ok_k = (abs(K_new[0, 0] - fy * s_u) < 1e-3 and abs(K_new[1, 1] - fx * s_v) < 1e-3
            and abs(K_new[0, 2] - cy * s_u) < 1e-3 and abs(K_new[1, 2] - cx * s_v) < 1e-3)
    print(f"   内参: {'✅ PASS' if ok_k else '❌ FAIL'}")

    # ---- 关键判据：同一个 3D 点，两种投影应呈 (u,v) <-> (v,u) 关系 ----
    print("=" * 66)
    print(" 投影判据：原图 (u,v) → 转置图应为 (v,u) 经缩放后")
    # 转置图未 resize 尺寸 = (H=W0, W=H0)。原图 (u,v) 在转置图里是 (v,u)。
    # 转置后宽 = H0(=1920)，高 = W0(=1440)；再 resize 到 (new_w=518, new_h=294)
    # 所以 u' = v * new_w/H0, v' = u * new_h/W0
    errs_axis_swap = []
    errs_no_swap = []
    errs_left_mul = []
    Mnp = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c2w_left = c2w_orig.copy()
    c2w_left[:3, :3] = Mnp @ c2w_left[:3, :3]
    c2w_left[:3, 3] = Mnp @ c2w_left[:3, 3]
    w2c_left = np.linalg.inv(c2w_left)

    for _ in range(8):
        u = rng.uniform(0.15, 0.85) * W0
        v = rng.uniform(0.15, 0.85) * H0
        Z = rng.uniform(1.0, 4.0)
        # 原图反投影（相机坐标系）
        Xc = np.array([(u - cx) / fx * Z, (v - cy) / fy * Z, Z])
        Xw = c2w_orig[:3, :3] @ Xc + c2w_orig[:3, 3]

        # 期望：转置图上的像素
        u_exp = v * new_w / H0
        v_exp = u * new_h / W0

        # 实测 A：修复后（右乘 M，作用在相机系）
        u_a, v_a, z_a = project(K_new, w2c_new, Xw)
        err_a = max(abs(u_a - u_exp), abs(v_a - v_exp))

        # 实测 B：不交换轴的版本（复现旧 bug）
        K_old = np.array([[fy * s_u, 0, cy * s_u],
                          [0, fx * s_v, cx * s_v],
                          [0, 0, 1]], dtype=np.float64)
        u_b, v_b, z_b = project(K_old, w2c_orig, Xw)
        err_b = max(abs(u_b - u_exp), abs(v_b - v_exp))

        # 实测 C：错把 M 左乘到世界系（我第一版的错误写法）
        u_c, v_c, z_c = project(K_old, w2c_left, Xw)
        err_c = max(abs(u_c - u_exp), abs(v_c - v_exp))

        errs_axis_swap.append(err_a)
        errs_no_swap.append(err_b)
        errs_left_mul.append(err_c)
        print(f"   Z={Z:.2f} 期望({u_exp:7.2f},{v_exp:7.2f}) "
              f"右乘({u_a:7.2f},{v_a:7.2f}) e={err_a:6.2f}px | "
              f"无交换 e={err_b:7.2f}px | 左乘 e={err_c:7.2f}px")

    mx_a, mx_b, mx_c = max(errs_axis_swap), max(errs_no_swap), max(errs_left_mul)
    print("=" * 66)
    print(f" 右乘 M（正确）  最大误差 {mx_a:8.3f} px  → {'✅ PASS' if mx_a < 0.5 else '❌ FAIL'}")
    print(f" 无轴交换        最大误差 {mx_b:8.3f} px  → {'（应 > 右乘）' if mx_b > mx_a else '⚠️'}")
    print(f" 左乘 M（错误）  最大误差 {mx_c:8.3f} px  → {'（应 >> 右乘，左乘更差）' if mx_c > mx_a * 10 else '⚠️'}")
    print("=" * 66)
    ok = ok_k and mx_a < 0.5 and mx_b > mx_a and mx_c > mx_a
    print(("🎉 全部通过" if ok else "❌ 有检查未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
