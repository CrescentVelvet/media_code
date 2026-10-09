#!/usr/bin/env python3
"""pose_adjust_colmap.py — 对 COLMAP 场景做一次性位姿调整（居中 + 重力对齐 + 尺度归一化）。

变换核心直接复用 vggt_human/pose_adjuster.py 的 PoseAdjuster（跨目录 import，
不复制代码；同一套实现已在 vggt_human 的 npz_to_colmap 链路里验证过）：
  Camera:   new_w2c_t = scale * (w2c_t + w2c_r @ center)
            new_w2c_r = w2c_r @ pred_rot
  Points:   new_pts   = scale * (pts - center) @ pred_rot   （行向量约定）

输入/输出均为 COLMAP **TXT** 模型目录（cameras.txt / images.txt / points3D.txt），
TXT↔BIN 转换由外层 01c_pose_adjust.sh 调 colmap model_converter 完成。

用法：
  python pose_adjust_colmap.py \
      --txt_dir /tmp/txt_raw --out_dir /tmp/txt_adj \
      --json  $SCENE_DIR/pose_adjuster.json \
      [--no-trans] [--no-rotate] [--no-scale] [--gravity-prior]

环境变量：
  VGGT_HUMAN_DIR   pose_adjuster.py 所在目录（默认 ../vggt_human，相对本文件）
"""
import os
import sys
import json
import math
import shutil
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_VH = os.environ.get("VGGT_HUMAN_DIR", os.path.normpath(os.path.join(_HERE, "..", "vggt_human")))
if _VH not in sys.path:
    sys.path.insert(0, _VH)

import torch  # noqa: E402
from pose_adjuster import PoseAdjuster  # noqa: E402


# ---------------------------------------------------------------------------
# 四元数 ↔ 旋转矩阵（numpy，Shepperd 全分支；COLMAP 约定 [qw,qx,qy,qz]，w2c）
# ---------------------------------------------------------------------------
def qvec_to_rotmat(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
        [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
    ], dtype=np.float64)


def rotmat_to_qvec(R):
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw, qx, qy, qz = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw, qx, qy, qz = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw, qx, qy, qz = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q


# ---------------------------------------------------------------------------
# COLMAP TXT 解析（保留无法理解的行原样回写）
# ---------------------------------------------------------------------------
def parse_images_txt(path):
    """返回 (header_lines, entries)。entries: list of dict，含原始 points2D 行。"""
    header, entries = [], []
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or not line.strip():
            header.append(line)
            i += 1
            continue
        parts = line.split()
        entry = {
            "image_id": int(parts[0]),
            "qvec": np.array([float(x) for x in parts[1:5]], dtype=np.float64),
            "tvec": np.array([float(x) for x in parts[5:8]], dtype=np.float64),
            "camera_id": int(parts[8]),
            "name": " ".join(parts[9:]),
            "pts2d_line": lines[i + 1] if i + 1 < len(lines) else "",
        }
        entries.append(entry)
        i += 2
    return header, entries


def parse_points3d_txt(path):
    header, entries = [], []
    with open(path) as f:
        lines = f.read().splitlines()
    for line in lines:
        if line.startswith("#") or not line.strip():
            header.append(line)
            continue
        parts = line.split()
        entries.append({
            "pid": int(parts[0]),
            "xyz": np.array([float(x) for x in parts[1:4]], dtype=np.float64),
            "rgb": parts[4:7],
            "error": parts[7],
            "track_tail": " ".join(parts[8:]),
        })
    return header, entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt_dir", required=True, help="输入 COLMAP TXT 模型目录")
    ap.add_argument("--out_dir", required=True, help="输出 COLMAP TXT 模型目录")
    ap.add_argument("--json", default="", help="pose_adjuster.json 输出路径（反变换用）")
    ap.add_argument("--no-trans", action="store_true")
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--gravity-prior", action="store_true",
                    help="不估计重力，直接用世界系 -Y 当 down（默认从相机 right 向量 SVD 估计）")
    args = ap.parse_args()

    images_path = os.path.join(args.txt_dir, "images.txt")
    points_path = os.path.join(args.txt_dir, "points3D.txt")
    cameras_path = os.path.join(args.txt_dir, "cameras.txt")
    for p in (images_path, points_path, cameras_path):
        if not os.path.isfile(p):
            sys.exit(f"❌ 缺少 {p}（先 colmap model_converter 转 TXT）")

    img_header, images = parse_images_txt(images_path)
    pts_header, points = parse_points3d_txt(points_path)
    if not images:
        sys.exit("❌ images.txt 里没有已注册图像")
    print(f"📥 输入: {len(images)} 图像, {len(points)} 稀疏点")

    # ── 组装 PoseAdjuster 输入（torch float64, CPU 足够）─────────────────────
    dtype = torch.float64
    w2c_r_list, w2c_t_list, c2w_t_list, sight_list = [], [], [], []
    for e in images:
        R = qvec_to_rotmat(*e["qvec"])          # w2c
        t = e["tvec"]                            # w2c
        c = -R.T @ t                             # 相机中心（c2w 平移）
        sight = R.T @ np.array([0.0, 0.0, 1.0])  # 相机光轴（世界系）
        w2c_r_list.append(torch.tensor(R, dtype=dtype))
        w2c_t_list.append(torch.tensor(t, dtype=dtype))
        c2w_t_list.append(torch.tensor(c, dtype=dtype))
        sight_list.append(torch.tensor(sight, dtype=dtype))

    adjuster = PoseAdjuster(
        w2c_r_list, w2c_t_list, c2w_t_list, sight_list,
        enable_trans=not args.no_trans,
        enable_rotate=not args.no_rotate,
        enable_scale=not args.no_scale,
        gravity_prior=args.gravity_prior,
        device="cpu")

    center = adjuster.center.to(dtype).numpy()
    pred_rot = adjuster.pred_rot.to(dtype).numpy()
    scale = float(adjuster.scale.item())

    # ── 变换相机（w2c 约定，与 pose_adjuster 文档一致）──────────────────────
    for e in images:
        R = qvec_to_rotmat(*e["qvec"])
        t = e["tvec"]
        new_r = R @ pred_rot
        new_t = scale * (t + R @ center)
        e["qvec"] = rotmat_to_qvec(new_r)
        e["tvec"] = new_t

    # ── 变换稀疏点 ──────────────────────────────────────────────────────────
    for p in points:
        p["xyz"] = scale * (p["xyz"] - center) @ pred_rot

    # ── 写出 ────────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    shutil.copy2(cameras_path, os.path.join(args.out_dir, "cameras.txt"))

    with open(os.path.join(args.out_dir, "images.txt"), "w") as f:
        for h in img_header:
            f.write(h + "\n")
        for e in images:
            q, t = e["qvec"], e["tvec"]
            f.write(f"{e['image_id']} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} "
                    f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {e['camera_id']} {e['name']}\n")
            f.write(e["pts2d_line"] + "\n")

    with open(os.path.join(args.out_dir, "points3D.txt"), "w") as f:
        for h in pts_header:
            f.write(h + "\n")
        for p in points:
            x = p["xyz"]
            tail = f" {p['track_tail']}" if p["track_tail"] else ""
            f.write(f"{p['pid']} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f} "
                    f"{p['rgb'][0]} {p['rgb'][1]} {p['rgb'][2]} {p['error']}{tail}\n")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        adjuster.save(args.json)

    print(f"🎉 调整完成 -> {args.out_dir}")
    print(f"   center={center.round(4).tolist()} scale={scale:.4f}")


if __name__ == "__main__":
    main()
