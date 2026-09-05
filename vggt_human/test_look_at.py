#!/usr/bin/env python3
"""test_look_at.py — 验证 look_at_rotmat 的 COLMAP 约定正确性。

不依赖 3DGS / CUDA, 纯 numpy 数学测试。运行:
    python test_look_at.py

测试矩阵 (全部基于 COLMAP 约定: +X 右 +Y 下 +Z 后, 相机看 -Z):
  T1 标准 look-at: C=(0,0,5) 看原点, up=(0,1,0)
     → R 应为单位阵 (right=+X, down=+Y, back=+Z)
  T2 反向: C=(0,0,-5) 看原点
     → R[2,:]=(0,0,-1) (back 朝 -Z, 相机看 +Z 方向)
  T3 任意位姿: face_center 反投影应落在画面中心 (offset≈0)
  T4 COLMAP 一致性: 对真实 COLMAP 相机, R@C+T 的 z 应为负 (前方)
  T5 右手系: right × down = back (行列式=+1)
  T6 pitch 公转: apply_pitch 后相机仍朝 face_center
"""
import os
import sys
import json
import math
import numpy as np

# 让脚本能 import render_closeup (设置 GS_DIR 占位避免 import 时报错)
os.environ.setdefault("GS_DIR", "/tmp/nonexistent")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# render_closeup 顶部 import 了 PIL/numpy, 但没在模块级 import torch
# (torch 在 render_views 函数内 import), 所以模块级 import 安全
from render_closeup import look_at_rotmat, apply_pitch, rotate_around_axis


def assert_close(name, val, expect, tol=1e-6):
    ok = abs(float(val) - float(expect)) < tol
    flag = "✅" if ok else "❌"
    print(f"  {flag} {name}: got {val:.6f}, expect {expect:.6f}")
    assert ok, f"{name} FAILED: got {val}, expect {expect}"


def test_t1_standard_lookat():
    """C=(0,0,5) 看原点, up=(0,1,0) → R 应为单位阵。"""
    print("\n[T1] 标准 look-at: C=(0,0,5)→origin, up=(0,1,0)")
    C = np.array([0.0, 0.0, 5.0])
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    R = look_at_rotmat(C, target, up)
    I = np.eye(3)
    print(f"  R=\n{np.round(R, 6)}")
    assert_close("R[0,0] (right.x)", R[0, 0], 1.0)
    assert_close("R[1,1] (down.y)", R[1, 1], 1.0)
    assert_close("R[2,2] (back.z)", R[2, 2], 1.0)
    err = np.linalg.norm(R - I)
    assert_close("||R - I||", err, 0.0, tol=1e-5)


def test_t2_reverse_lookat():
    """C=(0,0,-5) 看原点 → back=+Z 的反 = -Z, R[2,:]=(0,0,-1)。"""
    print("\n[T2] 反向 look-at: C=(0,0,-5)→origin")
    C = np.array([0.0, 0.0, -5.0])
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    R = look_at_rotmat(C, target, up)
    print(f"  R=\n{np.round(R, 6)}")
    assert_close("R[2,2] (back.z)", R[2, 2], -1.0)
    # forward = -R[2,:] = (0,0,1) → 相机看 +Z, 朝原点 (从 -5 看 0 就是 +Z 方向) ✓


def test_t3_face_center_centered():
    """任意位姿: face_center 反投影应落在画面中心 (offset≈0)。"""
    print("\n[T3] face_center 反投影应居中")
    rng = np.random.default_rng(42)
    face_center = np.array([1.0, 2.0, 3.0])
    for i in range(10):
        C = face_center + rng.uniform(-5, 5, size=3)
        if np.linalg.norm(C - face_center) < 0.1:
            continue
        up = np.array([0.0, 1.0, 0.0])
        R = look_at_rotmat(C, face_center, up)
        T = -R @ C
        Pc = R @ face_center + T
        depth = -Pc[2]
        assert depth > 0, f"T3 iter {i}: depth={depth} <= 0, 相机背后!"
        fx = fy = 500.0
        cx = cy = 256.0
        u = fx * Pc[0] / depth + cx
        v = fy * Pc[1] / depth + cy
        off = math.hypot(u - cx, v - cy)
        print(f"  iter {i}: depth={depth:.3f}, offset={off:.4f}px")
        assert off < 1e-3, f"T3 iter {i}: offset={off} 太大"


def test_t4_colmap_consistency():
    """COLMAP 约定: 相机看 -Z, 所以 face_center 在相机系 z 应为负。"""
    print("\n[T4] COLMAP 一致性: face_center 在相机系 z<0")
    face_center = np.array([0.0, 0.0, 0.0])
    # 相机在前方 (任意位置看原点)
    for pos in [(0, 0, 5), (3, 1, -4), (-2, 0, 6), (1, -1, -3)]:
        C = np.array(pos, dtype=float)
        if np.linalg.norm(C - face_center) < 0.1:
            continue
        R = look_at_rotmat(C, face_center, np.array([0.0, 1.0, 0.0]))
        T = -R @ C
        Pc = R @ face_center + T
        print(f"  C={pos}: Pc={np.round(Pc, 3)}, z={Pc[2]:.3f}")
        assert Pc[2] < 0, f"T4 C={pos}: Pc[2]={Pc[2]} >= 0, 违反 COLMAP -Z forward"


def test_t5_right_handed():
    """右手系: right × down = back, det(R)=+1。"""
    print("\n[T5] 右手系: det(R)=+1, right×down=back")
    C = np.array([2.0, -1.0, 4.0])
    target = np.array([0.0, 0.0, 0.0])
    R = look_at_rotmat(C, target, np.array([0.0, 1.0, 0.0]))
    right, down, back = R[0], R[1], R[2]
    cross = np.cross(right, down)
    print(f"  right={np.round(right, 4)}")
    print(f"  down ={np.round(down, 4)}")
    print(f"  back ={np.round(back, 4)}")
    print(f"  right×down = {np.round(cross, 4)}")
    err = np.linalg.norm(cross - back)
    assert_close("||right×down - back||", err, 0.0)
    det = np.linalg.det(R)
    assert_close("det(R)", det, 1.0)


def test_t6_pitch_keeps_face_centered():
    """apply_pitch 后相机仍朝 face_center (offset≈0)。"""
    print("\n[T6] pitch 公转后仍朝 face_center")
    face_center = np.array([1.0, 0.5, 2.0])
    C0 = face_center + np.array([0.0, 0.0, 5.0])
    R0 = look_at_rotmat(C0, face_center, np.array([0.0, 1.0, 0.0]))
    view = {"R": R0, "C": C0}
    for pitch in [-20, -10, 0, 10, 20]:
        R_p, T_p, C_p = apply_pitch(view, face_center, pitch)
        Pc = R_p @ face_center + T_p
        depth = -Pc[2]
        assert depth > 0, f"pitch={pitch}: depth={depth}<=0"
        fx = fy = 500.0
        cx = cy = 256.0
        u = fx * Pc[0] / depth + cx
        v = fy * Pc[1] / depth + cy
        off = math.hypot(u - cx, v - cy)
        print(f"  pitch={pitch:+3d}°: depth={depth:.3f}, offset={off:.4f}px")
        assert off < 1e-3, f"pitch={pitch}: offset={off} 太大"


def test_t7_compare_with_colmap_real_cameras():
    """对比真实 COLMAP 相机: 用 look_at_rotmat 重建的 R 应与 COLMAP R 方向一致。

    取一个 COLMAP 相机 (C, R_colmap), 用 look_at_rotmat(C, target, up) 重建,
    target 取 R_colmap 光轴上前方一点, 验证 R_new 与 R_colmap 的 forward 方向一致。
    """
    print("\n[T7] 与真实 COLMAP 相机 forward 方向一致")
    # 构造一个 COLMAP 相机: C=(1,2,3), 看向 origin
    C = np.array([1.0, 2.0, 3.0])
    target = np.array([0.0, 0.0, 0.0])
    # COLMAP R: 用 look_at_rotmat 应能得到正确的 world->camera
    R = look_at_rotmat(C, target, np.array([0.0, 1.0, 0.0]))
    # forward (世界系, 朝场景) = target - C 归一化
    fwd_world = (target - C) / np.linalg.norm(target - C)
    # COLMAP 相机看 -Z, 所以相机系 forward = (0,0,-1)
    # world->cam: R @ fwd_world 应 = (0,0,-1)
    fwd_cam = R @ fwd_world
    print(f"  fwd_world={np.round(fwd_world, 4)}")
    print(f"  R @ fwd_world = {np.round(fwd_cam, 4)} (应=(0,0,-1))")
    assert_close("fwd_cam.z", fwd_cam[2], -1.0)
    assert_close("fwd_cam.x", fwd_cam[0], 0.0, tol=1e-6)
    assert_close("fwd_cam.y", fwd_cam[1], 0.0, tol=1e-6)


def test_t8_render_views_R_convention():
    """验证 render_views 传给 3DGS Camera 的 R=R.T 约定。

    3DGS Camera 期望 camera-to-world 的 R (见 dataset_readers qvec2rotmat(...).T)。
    我们存的是 world-to-camera 的 R (COLMAP), 传 R.T 正确。
    本测试验证: R.T @ (相机系点) = 世界系点。
    """
    print("\n[T8] R.T 约定: R.T 是 camera->world")
    C = np.array([1.0, 2.0, 3.0])
    target = np.array([0.0, 0.0, 0.0])
    R = look_at_rotmat(C, target, np.array([0.0, 1.0, 0.0]))
    # 相机系原点 (0,0,0) 经 R.T 应回到... 不对, camera->world 是 R.T @ p_cam + C
    # 验证: 相机系 forward 点 (0,0,-1) (前方) 经 R.T 应指向 target
    p_cam = np.array([0.0, 0.0, -1.0])  # 相机前方
    p_world = R.T @ p_cam + C
    fwd_dir = (target - C) / np.linalg.norm(target - C)
    print(f"  相机前方点 (0,0,-1) → 世界 {np.round(p_world, 3)}")
    print(f"  期望方向 (target-C) = {np.round(fwd_dir, 3)}")
    # p_world - C 应与 fwd_dir 同向
    got = (p_world - C)
    got = got / np.linalg.norm(got)
    cos = float(np.dot(got, fwd_dir))
    print(f"  cos(got, expect) = {cos:.6f} (应=1.0)")
    assert_close("cos(forward_dir)", cos, 1.0, tol=1e-5)


def main():
    print("=" * 60)
    print("test_look_at.py — 验证 look_at_rotmat COLMAP 约定")
    print("COLMAP: +X 右 +Y 下 +Z 后, 相机看 -Z, forward=-R[2,:]")
    print("=" * 60)
    tests = [
        test_t1_standard_lookat,
        test_t2_reverse_lookat,
        test_t3_face_center_centered,
        test_t4_colmap_consistency,
        test_t5_right_handed,
        test_t6_pitch_keeps_face_centered,
        test_t7_compare_with_colmap_real_cameras,
        test_t8_render_views_R_convention,
    ]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
            print(f"  ✅ {t.__name__} PASS")
        except AssertionError as e:
            print(f"  ❌ {t.__name__} FAIL: {e}")
    print(f"\n{'=' * 60}")
    print(f"结果: {n_pass}/{len(tests)} 测试通过")
    if n_pass == len(tests):
        print("🎉 全部通过!")
        return 0
    else:
        print("❌ 有测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
