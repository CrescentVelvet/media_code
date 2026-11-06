#!/usr/bin/env python3
"""对棚拍多视角图像跑 Depth-Anything-3，导出深度/位姿/内参/置信度并做验收自检。

验收项（对应 EXPERIMENT_PLAN.md 阶段 B）：
  1. 深度：无 NaN、量级合理（NESTED 输出为米）；逐视角 min/max/mean
  2. 位姿：相机中心轨迹相邻位移无异常跳变；首末闭合漂移
  3. 对齐校验（可选）：与 COLMAP 参考 sparse/0 按文件名匹配，Sim(3) 对齐后报位置误差

Env vars:
  INPUT_DIR        输入图像目录（jpg/png，文件名排序）
  OUTPUT_DIR       输出目录
  DA3_MODEL_REPO   HF repo id（本地无权重时用）
  DA3_MODEL_DIR    本地权重目录（有 model.safetensors 时优先）
  PROCESS_RES      处理分辨率，默认 504
  MAX_VIEWS        0=全部；显存不足时均匀抽帧
  EXPORT_FORMAT    mini_npz-colmap-depth_vis 等（DA3 支持 - 组合）
  COLMAP_REF_DIR   可选，COLMAP sparse/0 目录（含 images.txt）
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image


def log(msg):
    print(msg, flush=True)


def load_images(input_dir, max_views):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(input_dir, e)))
    files = sorted(files)
    if not files:
        sys.exit(f"❌ INPUT_DIR 下没有图像: {input_dir}")
    if max_views and len(files) > max_views:
        idx = np.linspace(0, len(files) - 1, max_views).round().astype(int)
        files = [files[i] for i in idx]
        log(f"⏭️ 均匀抽帧: {len(files)} / {len(idx)} 视角 (MAX_VIEWS={max_views})")
    return files


def camera_centers(w2c):
    """w2c (N,4,4) -> 相机中心 (N,3)：c = -R^T t"""
    r = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    return np.einsum("nij,nj->ni", -r.transpose(0, 2, 1), t)


def umeyama_sim3(src, dst):
    """dst ≈ s * R @ src + t（src/dst: (N,3)）。返回 s, R, t。"""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = (dst_c.T @ src_c) / len(src)
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    diag = np.diag([1.0, 1.0, d])
    r = u @ diag @ vt
    var_s = (src_c**2).sum() / len(src)
    # 标准 Umeyama：s = tr(D @ Sigma) / var_s，Sigma 为 cov 的奇异值对角阵
    sigma = np.diag(np.linalg.svd(cov, compute_uv=False))
    s = float(np.trace(diag * sigma) / var_s)
    t = mu_d - s * (r @ mu_s)
    return s, r, t


def parse_colmap_images(images_txt):
    """解析 COLMAP images.txt -> {文件名: 相机中心}

    ⚠️ points2D 行为空串（无 2D 点的注册图像）不能丢，否则 header/points 配对错位、
    大约一半图像匹配不上（2026-11-05 实测 63/125）。
    """
    centers = {}
    with open(images_txt, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        q = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        t = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
        name = parts[9]
        # quaternion (w,x,y,z) -> R
        w, x, y, z = q
        r = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        centers[name] = -(r.T @ t)
    return centers


def main():
    input_dir = os.environ["INPUT_DIR"]
    output_dir = os.environ["OUTPUT_DIR"]
    process_res = int(os.environ.get("PROCESS_RES", "504"))
    max_views = int(os.environ.get("MAX_VIEWS", "0"))
    export_format = os.environ.get("EXPORT_FORMAT", "mini_npz-colmap-depth_vis")
    # INFER_GS=1 启用官方 GSDPT 高斯头：一次前向直接出 gaussians.ply（无需 COLMAP + 3DGS 训练）
    # 需配合 EXPORT_FORMAT=gs_ply（可叠加，如 "mini_npz-gs_ply"）
    infer_gs = os.environ.get("INFER_GS", "0") == "1"
    colmap_ref = os.environ.get("COLMAP_REF_DIR", "")
    model_dir = os.environ.get("DA3_MODEL_DIR", "")
    model_repo = os.environ.get("DA3_MODEL_REPO", "depth-anything/DA3NESTED-GIANT-LARGE-1.1")

    images = load_images(input_dir, max_views)
    with Image.open(images[0]) as im:
        orig_w, orig_h = im.size
    log(f"🖼️ 输入: {len(images)} 视角, 原始分辨率 {orig_w}x{orig_h}")

    # 本地权重优先（snapshot_download 的 local_dir 直接可用）
    if model_dir and os.path.isfile(os.path.join(model_dir, "model.safetensors")):
        model_path = model_dir
        log(f"🏋️ 加载本地权重: {model_dir}")
    else:
        model_path = model_repo
        log(f"🏋️ 从 HF 加载: {model_repo}")

    from depth_anything_3.api import DepthAnything3  # noqa: 延迟到 torch 环境就绪后

    if infer_gs:
        # api.inference 见到 "gs" 会强制追加 gs_video 渲染（需要 gsplat）。
        # env 不装 gsplat 时该步抛 NameError，但 gs_ply 已先写出——这里直接短路，
        # 让 GS 导出保持干净 exit 0；要看视频再装 gsplat 并设 DA3_GS_VIDEO=1。
        import importlib.util
        if importlib.util.find_spec("gsplat") is None and os.environ.get("DA3_GS_VIDEO", "0") != "1":
            from depth_anything_3.utils import export as _da3_export
            _da3_export.export_to_gs_video = lambda *a, **k: None  # noqa: ARG005
            log("ℹ️ 未装 gsplat：跳过 gs_video 渲染，只出 gs_ply（DA3_GS_VIDEO=1 可强制）")

    t0 = time.time()
    model = DepthAnything3.from_pretrained(model_path).to("cuda")
    log(f"⏱️ 模型加载 {time.time() - t0:.1f}s")

    t0 = time.time()
    prediction = model.inference(
        images,
        process_res=process_res,
        export_dir=output_dir,
        export_format=export_format,
        infer_gs=infer_gs,
    )
    elapsed = time.time() - t0
    log(f"⏱️ 推理+导出 {elapsed:.1f}s")

    # ---- 验收自检 ----
    depth = prediction.depth  # (N,H,W) float32，NESTED 为米
    conf = prediction.conf
    w2c = np.asarray(prediction.extrinsics)  # (N,4,4) opencv/colmap w2c
    ixt = np.asarray(prediction.intrinsics)  # (N,3,3)
    n = len(images)
    assert depth.shape[0] == n, f"深度视角数不符: {depth.shape[0]} vs {n}"

    nan_cnt = int(np.isnan(depth).sum())
    d_med = float(np.median(depth))
    log(f"📊 深度: median={d_med:.3f}m min={float(depth.min()):.3f} max={float(depth.max()):.3f} NaN={nan_cnt}")
    if nan_cnt > 0:
        log("⚠️ 深度含 NaN，需人工检查")
    log(f"📊 置信度: mean={float(conf.mean()):.3f}")

    n_gauss = 0
    if infer_gs:
        g = getattr(prediction, "gaussians", None)
        if g is None:
            log("⚠️ infer_gs=True 但 prediction.gaussians 为空（检查权重是否含 gs_head）")
        else:
            n_gauss = int(np.prod(np.asarray(g.means.shape)[:-1]))
            log(f"📊 高斯: {n_gauss} 个（官方 GSDPT 头一次前向输出）")

    centers = camera_centers(w2c)
    step = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    radius = float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).mean())
    loop_gap = float(np.linalg.norm(centers[0] - centers[-1]))
    log(f"📊 位姿: 轨迹半径≈{radius:.3f}m, 相邻位移 median={float(np.median(step)):.3f}m "
        f"max={float(step.max()):.3f}m, 首末闭合差={loop_gap:.3f}m")

    summary = {
        "scene": os.environ.get("SCENE_NAME", "unknown"),
        "model": model_path,
        "n_views": n,
        "orig_resolution": [orig_w, orig_h],
        "process_res": process_res,
        "is_metric": int(prediction.is_metric),
        "depth_median_m": d_med,
        "depth_min_m": float(depth.min()),
        "depth_max_m": float(depth.max()),
        "depth_nan_count": nan_cnt,
        "conf_mean": float(conf.mean()),
        "traj_radius_m": radius,
        "traj_step_median": float(np.median(step)),
        "traj_step_max": float(step.max()),
        "traj_loop_gap_m": loop_gap,
        "infer_seconds": round(elapsed, 1),
        "export_format": export_format,
        "infer_gs": int(infer_gs),
        "n_gaussians": n_gauss,
        "per_view": {
            "depth_min": [float(d.min()) for d in depth],
            "depth_max": [float(d.max()) for d in depth],
            "conf_mean": [float(c.mean()) for c in conf],
        },
    }

    # 可选：与 COLMAP 参考对齐校验
    if colmap_ref:
        ref_txt = os.path.join(colmap_ref, "images.txt")
        if not os.path.isfile(ref_txt):
            log(f"⚠️ COLMAP_REF_DIR 无 images.txt: {colmap_ref}，跳过对齐校验")
        else:
            ref = parse_colmap_images(ref_txt)
            matched_ref, matched_est = [], []
            unmatched = 0
            for i, p in enumerate(images):
                name = os.path.basename(p)
                if name in ref:
                    matched_ref.append(ref[name])
                    matched_est.append(centers[i])
                else:
                    unmatched += 1
            matched_ref = np.array(matched_ref)
            matched_est = np.array(matched_est)
            s, r, t = umeyama_sim3(matched_est, matched_ref)
            aligned = s * (matched_est @ r.T) + t
            err = np.linalg.norm(aligned - matched_ref, axis=1)
            log(f"🎯 COLMAP 对齐: 匹配 {len(matched_ref)}/{n}（未匹配 {unmatched}）, "
                f"Sim(3) scale={s:.4f}, 位置误差 mean={float(err.mean()):.4f}m median={float(np.median(err)):.4f}m")
            summary["colmap_align"] = {
                "matched": len(matched_ref),
                "unmatched": unmatched,
                "scale": float(s),
                "pos_err_mean_m": float(err.mean()),
                "pos_err_median_m": float(np.median(err)),
            }

    summary_path = os.path.join(output_dir, "da3_summary.json")
    with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    log(f"✅ summary: {summary_path}")
    log("🎉 DA3 推理完成")


if __name__ == "__main__":
    main()
