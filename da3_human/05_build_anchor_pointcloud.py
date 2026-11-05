# 05_build_anchor_pointcloud.py — DA3 深度反投影 anchor 点云（阶段 C 输入）
# 计划: EXPeriment_PLAN.md 阶段 C [锚定] 步
#   DA3 深度(504p) 反投影 + 多视角一致性筛选(reproj 深度差 > τ 剔除) + 跨视角体素去重
# 产物: anchors.npz(xyz/rgb/conf/n_consistent/src_view) + anchors.ply(可视化) + anchor_summary.json
# 用法: python 05_build_anchor_pointcloud.py [--infer-dir ...] [--out-dir ...]
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# DA3 导出 key（mini_npz results.npz，2026-11-05 实测形状）：
#   depth [N,504,378] / conf [N,504,378] / extrinsics [N,3,4](w2c) / intrinsics [N,3,3](504p 内参)


def load_da3_exports(infer_dir: Path):
    npz_path = infer_dir / "exports" / "mini_npz" / "results.npz"
    if not npz_path.exists():
        npz_path = infer_dir / "exports" / "npz" / "results.npz"
    if not npz_path.exists():
        sys.exit(f"ERROR: DA3 npz 导出不存在: {infer_dir}/exports/(mini_npz|npz)/results.npz，先跑 02")
    z = np.load(npz_path)
    depth, conf = z["depth"], z["conf"]
    # w2c [N,3,4] → [N,4,4]（沿行轴拼齐次末行）
    ext = z["extrinsics"]
    bottom = np.tile(np.array([[[0, 0, 0, 1]]], np.float32), (len(ext), 1, 1))
    w2c = np.concatenate([ext, bottom], axis=1)
    K = z["intrinsics"]
    c2w = np.linalg.inv(w2c)
    return depth, conf, w2c, c2w, K


def block_downsample(img: np.ndarray, f: int) -> np.ndarray:
    """H,W → H//f,W//f 块均值（NaN 感知）。要求 H,W 整除 f。"""
    H, W = img.shape
    assert H % f == 0 and W % f == 0, f"depth 分辨率 {H}x{W} 不整除块大小 {f}"
    blk = img.reshape(H // f, f, W // f, f)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.nanmean(blk, axis=3), axis=1)


def camera_centers(c2w: np.ndarray) -> np.ndarray:
    return c2w[:, :3, 3]


def pick_neighbors(centers: np.ndarray, view_i: int, k: int) -> np.ndarray:
    d = np.linalg.norm(centers - centers[view_i], axis=1)
    d[view_i] = np.inf  # 排除自身
    return np.argsort(d)[:k]


def backproject(depth_i: np.ndarray, K_i: np.ndarray, c2w_i: np.ndarray) -> np.ndarray:
    """depth_i [H,W] → 世界坐标点云 [H*W,3]（无效深度为 NaN）"""
    H, W = depth_i.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5, np.arange(H, dtype=np.float32) + 0.5)
    fx, fy = K_i[0, 0], K_i[1, 1]
    cx, cy = K_i[0, 2], K_i[1, 2]
    z = depth_i.ravel()
    x = (u.ravel() - cx) / fx * z
    y = (v.ravel() - cy) / fy * z
    cam = np.stack([x, y, z, np.ones_like(z)], axis=0)  # [4, HW]
    world = (c2w_i @ cam)[:3].T  # [HW,3]
    world[~np.isfinite(world).all(axis=1)] = np.nan
    return world


def count_consistent(world_pts: np.ndarray, depth_nb: np.ndarray, K_nb: np.ndarray,
                     w2c_nb: np.ndarray, tau_rel: float, tau_abs: float) -> np.ndarray:
    """world_pts [M,3]（NaN 保留）投到邻域视角，统计深度一致数。返回 [M] int"""
    M = world_pts.shape[0]
    ok = np.zeros(M, dtype=np.int32)
    valid = np.isfinite(world_pts).all(axis=1)
    pts = world_pts[valid]
    hom = np.concatenate([pts, np.ones((len(pts), 1), np.float32)], axis=1)  # [m,4]
    cam = (w2c_nb @ hom.T).T  # [m,4]
    z = cam[:, 2]
    good_z = z > 1e-6
    u = K_nb[0, 0] * cam[:, 0] / np.where(good_z, z, 1.0) + K_nb[0, 2]
    v = K_nb[1, 1] * cam[:, 1] / np.where(good_z, z, 1.0) + K_nb[1, 2]
    H, W = depth_nb.shape
    inside = good_z & (u >= 0) & (u < W - 0.5) & (v >= 0) & (v < H - 0.5)
    ui = np.clip(np.round(u[inside]).astype(np.int32), 0, W - 1)
    vi = np.clip(np.round(v[inside]).astype(np.int32), 0, H - 1)
    d_nb = depth_nb[vi, ui]
    z_in = z[inside]
    consistent = inside.copy()
    consistent[inside] = np.isfinite(d_nb) & (np.abs(d_nb - z_in) <= np.maximum(tau_rel * z_in, tau_abs))
    sub = np.zeros(M, dtype=bool)
    sub[np.where(valid)[0]] = True
    sub[valid] &= consistent
    ok[sub] = 1
    return ok


def voxel_dedup(xyz: np.ndarray, conf: np.ndarray, rgb: np.ndarray, voxel: float):
    """跨视角体素去重：每体素保留 conf 最高的点。xyz 量级任意，voxel 同单位。"""
    keys = np.floor(xyz / voxel).astype(np.int64)
    # 体素 key 哈希（无碰撞顾虑的 1D 编码，坐标范围 ±2^20 体素足够）
    code = (keys[:, 0] << 42) | (keys[:, 1] << 21) | keys[:, 2]
    order = np.argsort(code, kind="stable")
    code_s = code[order]
    boundaries = np.flatnonzero(np.diff(code_s)) + 1
    groups = np.split(order, boundaries)
    # 每体素 conf 最大者
    keep = np.empty(len(groups), dtype=np.int64)
    for gi, g in enumerate(groups):
        keep[gi] = g[np.argmax(conf[g])]
    return keep


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    data = np.column_stack([xyz, rgb8])
    header = ("ply\nformat ascii 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.5f %.5f %.5f %d %d %d")


def main():
    ap = argparse.ArgumentParser(description="DA3 anchor 点云构建（多视角一致性筛选）")
    # 路径由 05_build_anchor_pointcloud.sh 经环境变量注入（仓内约定，同 run_da3_inference.py）
    ap.add_argument("--infer-dir", default=os.environ.get("DA3_INFER_DIR", ""))
    ap.add_argument("--out-dir", default=os.environ.get("ANCHOR_DIR", ""))
    ap.add_argument("--images-dir", default=os.environ.get("IMAGES_DIR", ""))
    ap.add_argument("--ds-factor", type=int, default=int(os.environ.get("ANCHOR_DS_FACTOR", "1")),
                    help="深度块降采样倍数（1=用 DA3 原生 504p）")
    ap.add_argument("--conf-pct", type=float, default=float(os.environ.get("ANCHOR_CONF_PCT", "40")),
                    help="conf 百分位阈值（同 02 colmap 导出 P40）")
    ap.add_argument("--tau-rel", type=float, default=float(os.environ.get("ANCHOR_TAU_REL", "0.03")),
                    help="邻域深度一致相对阈值")
    ap.add_argument("--tau-abs", type=float, default=float(os.environ.get("ANCHOR_TAU_ABS", "0.05")),
                    help="邻域深度一致绝对下限（米）")
    ap.add_argument("--neighbor-views", type=int, default=int(os.environ.get("ANCHOR_NEIGHBOR_VIEWS", "8")),
                    help="每个视角参与一致性检查的最近邻视角数")
    ap.add_argument("--min-consistent", type=int, default=int(os.environ.get("ANCHOR_MIN_CONSISTENT", "3")),
                    help="至少与多少个邻域视角一致才保留")
    ap.add_argument("--voxel", type=float, default=float(os.environ.get("ANCHOR_VOXEL", "0") or 0),
                    help="跨视角去重体素边长（米）；0 = 0.004×深度中值")
    args = ap.parse_args()

    infer_dir = args.infer_dir
    out_dir = args.out_dir
    images_dir = args.images_dir
    if not infer_dir or not out_dir:
        sys.exit("ERROR: --infer-dir/--out-dir（或 DA3_INFER_DIR/ANCHOR_DIR）未设置，请经 05 脚本运行")

    depth, conf, w2c, c2w, K = load_da3_exports(Path(infer_dir))
    N, H0, W0 = depth.shape
    f = args.ds_factor
    print(f"📥 DA3 导出: {N} 视角, 深度 {H0}x{W0}, ds_factor={f}")

    conf_th = float(np.nanpercentile(conf, args.conf_pct))
    print(f"🔒 conf 阈值 P{args.conf_pct:g} = {conf_th:.4f}")

    centers = camera_centers(c2w)
    median_depth = float(np.nanmedian(depth))

    all_xyz, all_rgb, all_conf, all_ncons, all_src = [], [], [], [], []
    total_px = 0
    for i in range(N):
        d_i = depth[i] if f == 1 else block_downsample(depth[i], f)
        c_i = conf[i] if f == 1 else block_downsample(conf[i], f)
        K_i = K[i].copy()
        if f != 1:
            K_i[0, 0] /= f; K_i[1, 1] /= f; K_i[0, 2] /= f; K_i[1, 2] /= f
        H, W = d_i.shape
        valid = np.isfinite(d_i) & (c_i >= conf_th) & (d_i > 1e-3)
        if not valid.any():
            print(f"  ⏭️  view {i}: 无有效像素")
            continue
        world = backproject(np.where(valid, d_i, np.nan), K_i, c2w[i])
        nbrs = pick_neighbors(centers, i, args.neighbor_views)
        n_ok = np.zeros(world.shape[0], dtype=np.int32)
        for j in nbrs:
            n_ok += count_consistent(world, depth[j], K[j], w2c[j], args.tau_rel, args.tau_abs)
        keep = np.isfinite(world).all(axis=1) & (n_ok >= args.min_consistent)
        if not keep.any():
            print(f"  ⏭️  view {i}: 一致性筛选后无保留点")
            continue
        xyz_i = world[keep]
        conf_i = c_i.ravel()[keep]
        # 颜色：原图降采样到 anchor 分辨率采样
        if images_dir:
            try:
                import cv2
                imgs = sorted(Path(images_dir).glob("*.jpg")) + sorted(Path(images_dir).glob("*.png"))
                img = cv2.imread(str(imgs[i]), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img_ds = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                rgb_i = img_ds.reshape(-1, 3)[keep]
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠️ view {i} 颜色读取失败（{e}），退 conf 灰度")
                rgb_i = np.stack([conf_i] * 3, axis=1)
        else:
            rgb_i = np.stack([conf_i] * 3, axis=1)
        all_xyz.append(xyz_i); all_rgb.append(rgb_i); all_conf.append(conf_i)
        all_ncons.append(n_ok[keep]); all_src.append(np.full(len(xyz_i), i, np.int32))
        total_px += int(valid.sum())
        if i % 25 == 0 or i == N - 1:
            print(f"  view {i:3d}/{N}: 有效 {int(valid.sum()):7d} → 保留 {int(keep.sum()):7d}")

    xyz = np.concatenate(all_xyz); rgb = np.concatenate(all_rgb)
    confs = np.concatenate(all_conf); ncons = np.concatenate(all_ncons); src = np.concatenate(all_src)
    n_before = len(xyz)
    voxel = args.voxel if args.voxel else 0.004 * median_depth
    keep_idx = voxel_dedup(xyz, confs, rgb, voxel)
    xyz, rgb, confs, ncons, src = xyz[keep_idx], rgb[keep_idx], confs[keep_idx], ncons[keep_idx], src[keep_idx]

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "anchors.npz", xyz=xyz.astype(np.float32), rgb=rgb.astype(np.float32),
                        conf=confs.astype(np.float32), n_consistent=ncons.astype(np.int32),
                        src_view=src.astype(np.int32))
    write_ply(out / "anchors.ply", xyz, rgb)
    summary = {
        "n_views": N, "depth_res": [H0, W0], "ds_factor": f,
        "conf_threshold": conf_th, "tau_rel": args.tau_rel, "tau_abs": args.tau_abs,
        "neighbor_views": args.neighbor_views, "min_consistent": args.min_consistent,
        "voxel": voxel, "median_depth": median_depth,
        "raw_backproject_px": total_px, "after_consistency": int(n_before), "after_dedup": int(len(xyz)),
        "mean_n_consistent": float(ncons.mean()),
    }
    with open(out / "anchor_summary.json", "w", encoding="utf-8", newline="\n") as fp:
        json.dump(summary, fp, indent=4, ensure_ascii=False)
    print(f"🎉 anchors: 反投影 {total_px:,} px → 一致性后 {n_before:,} → 去重后 {len(xyz):,}（voxel={voxel:.4f}m）")
    print(f"   产物: {out}/anchors.npz | anchors.ply | anchor_summary.json")


if __name__ == "__main__":
    main()
