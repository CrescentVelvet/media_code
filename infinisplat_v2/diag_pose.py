#!/usr/bin/env python3
"""诊断一批结果的相机几何：两两光轴夹角、基线（相对场景半径），配每视角 PSNR 与渲染覆盖率。

用途：判断渲染质量差是"模型重建不行"还是"DA3 位姿估错"。三视角大角度间隔时后者很常见，
表现是某些视角渲染几乎空白（覆盖率骤降）而非局部糊。

用法:
  python diag_pose.py --results ~/output/infinisplat_v2_results --prefix threeview
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_ply_points(path, max_pts=200000):
    from plyfile import PlyData
    el = PlyData.read(str(path))["vertex"]
    xyz = np.stack([np.asarray(el[a], dtype=np.float64) for a in ("x", "y", "z")], axis=1)
    if len(xyz) > max_pts:
        xyz = xyz[np.random.default_rng(0).choice(len(xyz), max_pts, replace=False)]
    return xyz


def coverage(png):
    """白底渲染里非白像素占比（背景纯白 = 1.0）。"""
    a = np.asarray(Image.open(png).convert("RGB"), dtype=np.float32) / 255.0
    return float((a.min(axis=2) < 0.985).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path.home() / "output/infinisplat_v2_results"))
    ap.add_argument("--prefix", default="threeview")
    args = ap.parse_args()

    res = Path(args.results)
    print(f"{'case':8s} {'views':>5s} {'angle°(两两)':>16s} {'baseline/r':>11s} "
          f"{'per-view PSNR':>22s} {'coverage':>22s}")
    for d in sorted(p for p in res.glob(f"{args.prefix}_*") if p.is_dir()):
        cams = d / "cameras.npz"
        ply = d / "scene.ply"
        if not (cams.exists() and ply.exists()):
            continue
        case = d.name[len(args.prefix) + 1:]
        c2w = np.load(cams)["c2w"]
        n = len(c2w)
        centers = c2w[:, :3, 3]
        fwd = c2w[:, :3, 2]                      # OpenCV c2w: z 轴朝前
        fwd = fwd / np.linalg.norm(fwd, axis=1, keepdims=True)
        angs = [np.degrees(np.arccos(np.clip(fwd[i] @ fwd[j], -1, 1)))
                for i in range(n) for j in range(i + 1, n)]
        xyz = load_ply_points(ply)
        r = float(np.linalg.norm(xyz, axis=1).mean())
        base = [float(np.linalg.norm(centers[i] - centers[j])) / r
                for i in range(n) for j in range(i + 1, n)]
        m = d / "eval" / f"metrics_rerender.json"
        ps = json.loads(m.read_text())["per_view"]["psnr"] if m.exists() else []
        cov = [coverage(p) for p in sorted((d / "eval").glob("render_*.png"))]
        fmt = lambda v, f: "[" + " ".join(format(x, f) for x in v) + "]"
        print(f"{case:8s} {n:5d} {fmt(angs,'6.1f'):>16s} {fmt(base,'5.2f'):>11s} "
              f"{fmt(ps,'6.2f'):>22s} {fmt(cov,'6.3f'):>22s}")


if __name__ == "__main__":
    main()
