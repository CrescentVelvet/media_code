#!/usr/bin/env python3
# qc_gravity_estimator.py — 校验 01c 的重力估计是否用对了矩阵约定。
#
# 背景：pose_adjuster.estimate_down_vec 用「相机 right 向量」的 SVD 最小奇异向量
# 估重力，取的是 r[:, 0]（矩阵第 0 列）。而 01c 传进去的是 **w2c** 旋转矩阵，
# 对 w2c 而言「相机 right 轴在世界系」是第 0 **行**，第 0 列是世界 X 轴在相机系
# 的坐标（各相机不同系，跨相机比较无物理意义）→ 重力估计失效。
#
# 本脚本在原始坐标系（sparse/0_raw）上对比两种取法的估计值，并以两个独立参考
# 交叉验证：
#   ref1 = 各相机「图像下方」均值（w2c 第 1 行，手持竖拍时 ≈ 重力向下）
#   ref2 = 相机轨迹平面法向（环绕拍摄时 ≈ 重力）
#
# 用法：SCENE_DIR=<colmap_scene> python deformable_human/qc_gravity_estimator.py
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vggt_human"))
from pose_adjuster import estimate_down_vec  # noqa: E402
import torch  # noqa: E402


def qvec_to_rotmat(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
        [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
    ], dtype=np.float64)


def ang(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def load_raw(scene_dir):
    """sparse/0_raw → 相机 w2c R 与相机中心（用 colmap model_converter 转 TXT）。"""
    src = os.path.join(scene_dir, "sparse", "0_raw")
    if not os.path.isdir(src):
        return None, None
    tmp = tempfile.mkdtemp(prefix="qc_grav_")
    subprocess.run(["colmap", "model_converter", "--input_path", src,
                    "--output_path", tmp, "--output_type", "TXT"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    Rs, Cs = [], []
    with open(os.path.join(tmp, "images.txt")) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip() and not ln.startswith("#")]
    for ln in lines[::2]:                      # 奇数行是 points2D
        p = ln.split()
        R = qvec_to_rotmat(*[float(x) for x in p[1:5]])
        t = np.array([float(x) for x in p[5:8]])
        Rs.append(R)
        Cs.append(-R.T @ t)
    return np.array(Rs), np.array(Cs)


def main():
    scene_dir = os.path.expanduser(os.environ.get("SCENE_DIR", ""))
    if not scene_dir or not os.path.isdir(scene_dir):
        sys.exit("❌ 需要 SCENE_DIR=<colmap_scene>")
    label = os.path.basename(os.path.dirname(scene_dir))
    Rs, Cs = load_raw(scene_dir)
    if Rs is None:
        print(f"══ {label} ══ 无 sparse/0_raw，跳过")
        return

    # 代码现状：对 R[:, 0]（列）做 SVD
    down_code = estimate_down_vec([torch.tensor(R, dtype=torch.float64) for R in Rs]).numpy()
    # 修正版：对 R[0, :]（行 = 相机 right 轴在世界系）做 SVD
    down_fix = estimate_down_vec(
        [torch.tensor(R.T, dtype=torch.float64) for R in Rs]).numpy()

    # ── 硬参考：mask 主轴反解的重力（在训练坐标系），再映回原始坐标系 ──
    #    adjusted → original 的方向映射：d_orig = pred_rot @ d_adj
    import json
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qc_gravity_axis import gravity_from_mask_orientation

    scene_root = os.path.dirname(os.path.abspath(scene_dir.rstrip("/")))
    model_dir = os.path.join(scene_root, "model")
    ref = None
    used = 0
    if os.path.isdir(model_dir):
        cams = json.load(open(os.path.join(model_dir, "cameras.json")))
        Rc = np.array([c["rotation"] for c in cams])
        Cc = np.array([c["position"] for c in cams])
        g_adj, used = gravity_from_mask_orientation(model_dir, cams, Rc, Cc, "c2w")
        if g_adj is not None:
            with open(os.path.join(scene_dir, "pose_adjuster.json")) as f:
                pred_rot = np.array(json.load(f)["pred_rot"])
            ref = pred_rot @ g_adj            # 映回原始坐标系
            ref = ref / np.linalg.norm(ref)

    print(f"══ {label} ══  ({len(Rs)} 相机)")
    if ref is None:
        print("   （mask 参考不可用，只列两种估计之间的夹角）")
        print(f"   列 vs 行 夹角: {min(ang(down_code, down_fix), 180-ang(down_code, down_fix)):.2f}°")
        return
    a = ang(down_code, ref)
    a = min(a, 180 - a)
    b = ang(down_fix, ref)
    b = min(b, 180 - b)
    print(f"   参考（mask 主轴反解，{used} 帧）")
    print(f"   代码现状（列）vs 参考: {a:6.2f}°   ← 当前 01c 用的")
    print(f"   修正版（行）  vs 参考: {b:6.2f}°   ← 若明显更小，则证实约定 bug")



if __name__ == "__main__":
    main()
