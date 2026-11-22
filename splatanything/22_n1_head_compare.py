#!/usr/bin/env python3
"""N1：SA 原生 P0 / Zip FF 同三图、同 GT 相机的头脸几何与外观对照。

只读两模型原生 PLY；各自仅用 source 相机估计一个 X_gt=s R X_model+T。
原生高斯保持不变，将共同 GT 相机逆回模型系渲染，ED 乘 s 换回米制。
Zip 另跑 source 全部朝向的 Wahba 敏感性分支，但不按 target 选择对齐。
不训练、不跑 TokenBA、不拼 PLY、不用 target 拟合或选参。
ED 为 alpha 归一化轴向 Z；中心到深度点云距离不是独立网格真值。
"""
import argparse
import csv
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from plyfile import PlyData

ROOT = Path(__file__).resolve().parent


def load_mod(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M03 = load_mod("n1_render03", "03_render_eval.py")
M13 = load_mod("n1_select13", "13_face_select.py")
M15 = load_mod("n1_target15", "15_face_target_eval.py")
M21 = load_mod("n1_geometry21", "21_geometry_audit_v2.py")


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    def convert(v):
        if isinstance(v, dict):
            return {str(k): convert(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [convert(x) for x in v]
        if isinstance(v, np.ndarray):
            return convert(v.tolist())
        if isinstance(v, np.generic):
            return v.item()
        return v
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(convert(value), f, ensure_ascii=False, indent=2)


def load_ply(path, stored_quat_order):
    """读 PLY，GPU 保留 WXYZ；只有 scipy 的旋转组合边界临时转 XYZW。
    存储标签不能由相机工具推断，须由各自原生 PLY 回放门单独验证。
    """
    if stored_quat_order not in ("passthrough", "wxyz"):
        raise ValueError(stored_quat_order)
    p = PlyData.read(str(path))["vertex"]
    n_rest = sum(k.startswith("f_rest_") for k in p.data.dtype.names)
    n_sh = 1 + n_rest // 3
    degree = int(math.sqrt(n_sh)) - 1
    if n_rest % 3 or (degree + 1) ** 2 != n_sh:
        raise RuntimeError("Invalid SH channel count")
    sh = np.empty((len(p), n_sh, 3), np.float32)
    sh[:, 0] = np.stack([p[f"f_dc_{i}"] for i in range(3)], 1)
    if n_rest:
        sh[:, 1:] = np.stack([p[f"f_rest_{i}"] for i in range(n_rest)], 1).reshape(
            len(p), 3, n_sh - 1).transpose(0, 2, 1)
    quats = np.stack([p[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    if stored_quat_order == "wxyz":
        quats = quats.copy()  # GPU 原生 WXYZ；仅 sim3 组合时临时转成 scipy XYZW。
    return {"means": np.stack([p[k] for k in ("x", "y", "z")], 1).astype(np.float32),
            "scales": np.exp(np.stack([p[f"scale_{i}"] for i in range(3)], 1)).astype(np.float32),
            "quats": quats,
            "opacities": (1 / (1 + np.exp(-p["opacity"].astype(np.float32)))).astype(np.float32),
            "sh": sh, "degree": degree}


def fit_sim3(gt_centers, pred_centers):
    """返回 X_gt=s R X_pred+T；R 为列向量旋转，行实现 pred@R.T。"""
    G, P = np.asarray(gt_centers, np.float64), np.asarray(pred_centers, np.float64)
    gm, pm = G.mean(0), P.mean(0)
    gc, pc = G - gm, P - pm
    U, sig, Vt = np.linalg.svd(pc.T @ gc / len(P))
    fix = np.diag([1., 1., np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ fix @ U.T
    s = float(len(P) * np.sum(sig * np.diag(fix)) / np.sum(pc ** 2))
    T = gm - s * R @ pm
    if s <= 0 or abs(np.linalg.det(R) - 1) > 1e-5:
        raise RuntimeError("Degenerate source-only Sim3")
    return s, R, T, {"center": camera_center_fit(s, R, T, G, P)}


def camera_center_fit(s, R, T, gt_centers, pred_centers):
    residual = float(s) * (np.asarray(pred_centers) @ np.asarray(R).T) + T - gt_centers
    return {"rmse_m": float(np.sqrt(np.mean(residual ** 2))),
            "max_abs_m": float(np.max(np.abs(residual))),
            "per_view_error_m": np.linalg.norm(residual, axis=1).tolist()}


def fit_wahba_center(gt_c2w, pred_c2w):
    """仅用 source：全部朝向求 Wahba 旋转，中心求尺度/平移，方向为 model→GT。

    Wahba 求 R_w 使 R_pred≈R_w@R_gt，故返回 model→GT 旋转 R_w.T。
    尺度沿用中心 Umeyama 的 model→GT 解，不拿 target 修正焦距或深度。
    """
    gt_c2w, pred_c2w = np.asarray(gt_c2w, np.float64), np.asarray(pred_c2w, np.float64)
    if gt_c2w.shape != pred_c2w.shape or gt_c2w.shape[1:] != (4, 4):
        raise ValueError("Wahba inputs must be matching Nx4x4 c2w")
    M = np.einsum("nij,nkj->ik", pred_c2w[:, :3, :3], gt_c2w[:, :3, :3])
    U, _, Vt = np.linalg.svd(M)
    # 行向量定义下 Rw@R_pred≈R_gt；M=sum(R_pred@R_gt.T)，故 Rw=U@D@Vt。
    # 这是 zipsplat/evaluate_jiajia.py 旧转置实现的修正版；历史 JSON 不在此改写。
    # 最终返回 model→GT 的 Rw.T，精确随机相机测试直接验证这个方向。
    Rw = U @ np.diag([1., 1., np.linalg.det(U @ Vt)]) @ Vt
    R = Rw.T
    s = fit_sim3(gt_c2w[:, :3, 3], pred_c2w[:, :3, 3])[0]
    T = gt_c2w[:, :3, 3].mean(0) - s * R @ pred_c2w[:, :3, 3].mean(0)
    if s <= 0 or abs(np.linalg.det(R) - 1) > 1e-8:
        raise RuntimeError("Invalid Wahba source-only alignment")
    return s, R, T, {"center": camera_center_fit(
        s, R, T, gt_c2w[:, :3, 3], pred_c2w[:, :3, 3]),
        "rule": "source-only Wahba rotation + center-Umeyama scale/translation"}


def source_camera_residuals(gt_c2w, pred_c2w, sim3, sources):
    s, R, T, info = sim3
    out = {"center": camera_center_fit(s, R, T, gt_c2w[:, :3, 3], pred_c2w[:, :3, 3]),
           "orientation_residual_deg": {}}
    for stem, gt, pred in zip(sources, gt_c2w, pred_c2w):
        out["orientation_residual_deg"][stem] = angle(
            (R @ pred[:3, :3]).T @ gt[:3, :3])
    center_sv = np.linalg.svd(gt_c2w[:, :3, 3] - gt_c2w[:, :3, 3].mean(0),
                              compute_uv=False) / np.sqrt(len(gt_c2w))
    out["center_pca_std_m"] = center_sv.tolist()
    out["planar_degenerate"] = bool(center_sv[-1] < .1 * center_sv[0])
    info.update(out)
    return out


def wahba_test():
    rng = np.random.default_rng(224)
    from scipy.spatial.transform import Rotation
    R_gt_to_model = Rotation.from_euler("xyz", [21., -33., 47.], degrees=True).as_matrix()
    s_gt_to_model = 1.73
    T_gt_to_model = np.array([.2, -.3, .7])
    G, P = [], []
    for _ in range(5):
        R_gt = Rotation.random(random_state=rng).as_matrix()
        C_gt = rng.normal(size=3)
        R_model = R_gt_to_model @ R_gt
        C_model = s_gt_to_model * R_gt_to_model @ C_gt + T_gt_to_model
        G.append(np.block([[R_gt, C_gt[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
        P.append(np.block([[R_model, C_model[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
    G, P = np.stack(G), np.stack(P)
    fit = fit_wahba_center(G, P)
    recovered = np.stack([transform_camera(g, fit) for g in P])
    if np.max(np.abs(recovered - G)) > 1e-9:
        raise RuntimeError("Wahba model-to-GT camera roundtrip failed")
    source_camera_residuals(G, P, fit, [str(i) for i in range(len(G))])
    inverse = inverse_sim3(fit)
    if np.max(np.abs(np.stack([transform_camera(g, inverse) for g in G]) - P)) > 1e-9:
        raise RuntimeError("Wahba GT-to-model inverse camera roundtrip failed")
    print("Wahba source-only camera self-test passed", flush=True)


def inverse_sim3(sim3):
    """X_b=s R X_a+T 的逆：X_a=(1/s) R.T X_b-R.T T/s。"""
    s, R, T, fit = sim3
    R_inv = np.asarray(R, np.float64).T
    s_inv = 1.0 / float(s)
    T_inv = -s_inv * (R_inv @ np.asarray(T, np.float64))
    return s_inv, R_inv, T_inv, {"inverse_of": fit}


def sim3_test():
    rng = np.random.default_rng(22)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    P = rng.normal(size=(3, 3))
    G = 1.73 * (P @ Q.T) + np.array([.2, -.3, .7])
    s, R, T, info = fit_sim3(G, P)
    assert abs(s - 1.73) < 1e-9 and np.max(np.abs(R - Q)) < 1e-9
    assert info["center"]["max_abs_m"] < 1e-9
    si, Ri, Ti, _ = inverse_sim3((s, R, T, info))
    recovered = si * (G @ Ri.T) + Ti
    assert np.max(np.abs(recovered - P)) < 1e-9


def rotate_sh_degree1(sh, R):
    """q'_l=R q_l；[N,K,3] 中旋转 K 维，不能把 RGB 通道当向量维。"""
    if sh.shape[1] < 4:
        return sh.copy()
    # gsplat degree-1 基为 (-y,z,-x)/sqrt(4pi/3)，按列排成 A。
    # 等式 A.T q' = R (A.T q)，故 q'=(A.T)^-1 R A.T q=A R A.T q。
    A = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]], np.float64)
    D = A @ np.asarray(R, np.float64) @ A.T
    out = sh.copy()
    out[:, 1:4] = np.einsum("ij,njc->nic", D, sh[:, 1:4].astype(np.float64)
                            ).astype(np.float32)
    return out


def transform_gaussians(g, sim3):
    from scipy.spatial.transform import Rotation
    s, R, T, _ = sim3
    out = dict(g)
    if abs(s - 1) < 1e-8 and np.max(np.abs(R - np.eye(3))) < 1e-8 and np.max(np.abs(T)) < 1e-8:
        return out
    out["means"] = (s * (g["means"].astype(np.float64) @ R.T) + T).astype(np.float32)
    out["scales"] = (g["scales"] * s).astype(np.float32)
    out["quats"] = m03_quaternion_rotate(g["quats"], R)
    out["sh"] = rotate_sh_degree1(g["sh"], R)
    if g["degree"] > 1:
        raise NotImplementedError("N1 requires an exact rotation for SH degree > 1")
    return out


def m03_quaternion_rotate(q, R):
    from scipy.spatial.transform import Rotation
    """WXYZ 在 PLY/GPU 中保持不变；仅 scipy 边界显式转 XYZW。"""
    q = np.asarray(q)
    q_xyzw = q[..., [1, 2, 3, 0]]
    return (Rotation.from_matrix(R) * Rotation.from_quat(q_xyzw)).as_quat()[..., [3, 0, 1, 2]].astype(np.float32)


def rotation_self_test():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(221)
    for _ in range(20):
        R = Rotation.random(random_state=rng).as_matrix()
        q = Rotation.random(random_state=rng).as_quat()[[3, 0, 1, 2]]
        cov = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        actual = Rotation.from_quat(m03_quaternion_rotate(q[None], R)[0][[1, 2, 3, 0]]).as_matrix()
        if np.max(np.abs(actual - R @ cov)) > 2e-7:
            raise RuntimeError("Quaternion composition differs from R @ Q")
    print("Quaternion composition self-test passed: Q_new=R@Q", flush=True)


def sh_rotation_self_test(device):
    """用实际 CUDA kernel 验证方向保持，防止把 RGB 维当成 SH 基向量维。"""
    from gsplat import spherical_harmonics
    rng = np.random.default_rng(223)
    sh = rng.normal(size=(8, 4, 3)).astype(np.float32)
    sh[:, 0] = 0.
    dirs = rng.normal(size=(8, 3))
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("xyz", [23., -37., 61.], degrees=True).as_matrix()
    A = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
    expected = np.einsum("ij,njc->nic", A @ R @ A.T, sh[:, 1:4])
    rotated = rotate_sh_degree1(sh, R)
    if np.max(np.abs(rotated[:, 1:4] - expected)) > 1e-7:
        raise RuntimeError("SH tensor-axis implementation mismatch")
    old = spherical_harmonics(1, torch.from_numpy(dirs.astype(np.float32)).to(device),
                              torch.from_numpy(sh).to(device)).cpu().numpy()
    new = spherical_harmonics(1, torch.from_numpy((dirs @ R.T).astype(np.float32)).to(device),
                              torch.from_numpy(rotated).to(device)).cpu().numpy()
    delta = float(np.abs(old - new).max())
    if delta > 2e-6:
        raise RuntimeError("Degree-1 SH rotation is not equivariant: " + str(delta))
    print("SH degree-1 CUDA equivariance self-test passed, max=" + str(delta), flush=True)


def to_gpu(g, device):
    """gsplat 接受 WXYZ；内部与 scipy 的 XYZW 只在边界处转换一次。"""
    out = {k: torch.from_numpy(np.ascontiguousarray(g[k])).to(device).float()
           for k in ("means", "scales", "quats", "opacities", "sh")}
    return out


def render(g, degree, c2w, K, size, device, eps2d, height=None):
    from gsplat import rasterization
    height = size if height is None else height
    native_background = inspect.signature(rasterization).parameters["backgrounds"].default
    if native_background is not None:
        raise SystemExit("Common evaluation requires an explicit zero background")
    view = torch.from_numpy(np.linalg.inv(c2w)[None].astype(np.float32)).to(device)
    intr = torch.from_numpy(np.asarray(K, np.float32)[None]).to(device)
    rgb, alpha, _ = rasterization(
        means=g["means"], quats=g["quats"],
        scales=g["scales"], opacities=g["opacities"], colors=g["sh"],
        viewmats=view, Ks=intr, width=size, height=height, sh_degree=degree,
        eps2d=eps2d, near_plane=.01, far_plane=400., packed=False,
        backgrounds=torch.zeros((1, 3), device=device), render_mode="RGB+ED")
    return tuple(x[0].cpu().numpy() for x in (rgb[..., :3].clamp(0, 1), alpha[..., 0], rgb[..., 3]))


def transform_camera(c2w, sim3):
    c2w = np.asarray(c2w, np.float64)
    s, R, T, _ = sim3
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R @ c2w[:3, :3]
    out[:3, 3] = s * (R @ c2w[:3, 3]) + T
    return out


def angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def roi_metrics(gt, rd, alpha, mask, lp, device):
    n = int(mask.sum())
    if n < 32:
        return {"n": n, "status": "N/A"}
    mse = float(np.mean((gt[mask].astype(np.float64) - rd[mask]) ** 2))
    ys, xs = np.where(mask)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    def crop(a):
        im = Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255)).crop(box).resize((128, 128), Image.Resampling.LANCZOS)
        return torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None].to(device) * 2 - 1
    with torch.no_grad():
        lp_value = float(lp(crop(gt), crop(rd)).item())
    return {"n": n, "psnr": float(-10 * np.log10(max(mse, 1e-12))),
            "mae": float(np.mean(np.abs(gt[mask] - rd[mask]))),
            "alpha_mean": float(alpha[mask].mean()), "coverage_05": float((alpha[mask] > .5).mean()),
            "lpips_crop128": lp_value}


def depth_metrics(ed, gt, mask):
    err = (ed - gt)[mask & (ed > 1e-5)]
    if len(err) < 32:
        return {"n": int(len(err)), "status": "N/A"}
    return {"n": int(len(err)), "mae_mm": float(np.abs(err).mean() * 1000),
            "bias_mm": float(err.mean() * 1000), "p90_abs_mm": float(np.percentile(np.abs(err), 90) * 1000)}


def depth_scale_test():
    ed = np.full(40, .5)
    gt = np.full(40, .45)
    result = fit_source_depth_scale(ed, gt, np.ones(40, bool), .8)
    assert abs(result["additional_ratio_of_medians"] - 1.125) < 1e-12
    assert result["after"]["mae_mm"] < 1e-9
    assert abs(result["scale_model_to_gt"] - .9) < 1e-12


def fit_source_depth_scale(ed, gt, mask, base_scale):
    """仅用 source 有效像素求一个 ratio-of-medians；绝不在 target 上调用。"""
    selected = mask & (ed > 1e-5) & (gt > 1e-5)
    if selected.sum() < 32:
        raise RuntimeError("Insufficient source pixels for depth-scale diagnostic")
    ratio = float(np.median(gt[selected] / (ed[selected] * base_scale)))
    corrected = ed * base_scale * ratio
    return {"scale_model_to_gt": base_scale * ratio,
            "additional_ratio_of_medians": ratio, "n_source_pixels": int(selected.sum()),
            "before": depth_metrics(ed * base_scale, gt, mask),
            "after": depth_metrics(corrected, gt, mask),
            "rule": "median(GT_depth/(native_ED*base_scale)) pooled over source tight-face pixels; no target access"}


def silhouette(valid, alpha):
    a, b = valid.astype(np.uint8), (alpha > .5).astype(np.uint8)
    da, db = cv2.distanceTransform(1 - a, cv2.DIST_L2, 3), cv2.distanceTransform(1 - b, cv2.DIST_L2, 3)
    ea, eb = cv2.Canny(a * 255, 50, 150) > 0, cv2.Canny(b * 255, 50, 150) > 0
    d = np.concatenate([db[ea], da[eb]])
    return {"iou": float(np.sum((a & b) > 0) / max(np.sum((a | b) > 0), 1)),
            "boundary_mean_px": float(d.mean()), "boundary_p90_px": float(np.percentile(d, 90)),
            "definition": "whole alpha silhouette, coverage/scale confounded; not facial mesh"}


def heat(rgb, error, mask, vmax=.05):
    a = np.clip(np.abs(error) / vmax, 0, 1)
    a[~mask] = 0
    return np.uint8(np.clip(rgb * .65 + np.stack([a, 0 * a, 1 - a], -1) * .35, 0, 1) * 255)


def head_mask(points, person, size):
    oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
            379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
            234, 127, 162, 21, 54, 103, 67, 109]
    p = points[oval]
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    fw, fh = x1 - x0, y1 - y0
    box = [int(np.clip(np.floor(x0 - .25 * fw), 0, size - 1)),
           int(np.clip(np.floor(y0 - .60 * fh), 0, size - 1)),
           int(np.clip(np.ceil(x1 + .25 * fw), 1, size)),
           int(np.clip(np.ceil(y1 + .20 * fh), 1, size))]
    mask = np.zeros((size, size), bool)
    mask[box[1]:box[3], box[0]:box[2]] = person[box[1]:box[3], box[0]:box[2]]
    return box, mask


def save_comparison(out, stem, gt, renders, depth, valid, fbox, hbox):
    x0, y0, x1, y1 = fbox
    hx0, hy0, hx1, hy1 = hbox
    panels = []
    for title, im in [("GT", gt)] + [(k, v[0]) for k, v in renders.items()]:
        p = Image.fromarray(np.uint8(im * 255)).resize((300, 300))
        ImageDraw.Draw(p).text((8, 8), title, fill=(255, 30, 30), stroke_width=1, stroke_fill=(255, 255, 255))
        panels.append(np.asarray(p))
    for name, value in renders.items():
        p = Image.fromarray(heat(gt, value[2] - depth, valid)).resize((300, 300))
        ImageDraw.Draw(p).text((8, 8), name + " ED error (m)", fill=(255, 255, 255))
        panels.append(np.asarray(p))
    Image.fromarray(np.concatenate(panels, 1)).save(out / f"{stem}_rgb_ed.png")
    rows = []
    images = [("GT", gt)] + [(k, v[0]) for k, v in renders.items()]
    images += [(k + " lowpass sigma2", cv2.GaussianBlur(v[0], (0, 0), 2.)) for k, v in renders.items()]
    for title, im in images:
        face = Image.fromarray(np.uint8(im[y0:y1, x0:x1] * 255)).resize((360, 360), Image.Resampling.LANCZOS)
        head = Image.fromarray(np.uint8(im[hy0:hy1, hx0:hx1] * 255)).resize((300, 360), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (660, 360))
        tile.paste(head, (0, 0))
        tile.paste(face, (300, 0))
        ImageDraw.Draw(tile).text((8, 8), title, fill=(255, 30, 30), stroke_width=1, stroke_fill=(255, 255, 255))
        rows.append(np.asarray(tile))
    Image.fromarray(np.concatenate(rows, 0)).save(out / f"{stem}_head_face_lowpass.png")


def main():
    ap = argparse.ArgumentParser()
    for name in ("sa-result", "zip-result", "gt-pkg", "source-scene", "target-scene",
                 "landmarks", "roi-qa", "n0-audit", "official-repo", "zip-roundtrip-audit", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--size", type=int, default=756)
    ap.add_argument("--eps2d", type=float, default=.01)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--render-validation-dir",
                    default="/mnt/d/output/results/splatanything_results/n1_head_compare_20261122/render_chain_validation_v1")
    ap.add_argument("--skip-render-validation", action="store_true",
                    help="仅限复算已验证结果；新输入/新评测不得跳过原生渲染与 ED 硬门")
    args = ap.parse_args()
    sim3_test()
    wahba_test()
    depth_scale_test()
    rotation_self_test()
    sh_rotation_self_test(args.device)
    # 跨 Windows 调 WSL 时 shell 可能预展开 $CONDA_PREFIX 为空，解释器被错选。
    if Path(sys.executable).resolve() != (Path.home() / "miniconda3/envs/splatanything/bin/python").resolve():
        raise SystemExit("N1 must run in splatanything env, got " + sys.executable)
    sys.path.insert(0, str(Path(args.official_repo).resolve()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "audit_command.json", {"argv": sys.argv})
    sa_res, zip_res, pkg = Path(args.sa_result), Path(args.zip_result), Path(args.gt_pkg)
    n0 = json.loads(Path(args.n0_audit).read_text(encoding="utf-8"))
    zip_roundtrip = json.loads(Path(args.zip_roundtrip_audit).read_text(encoding="utf-8"))
    if zip_roundtrip.get("status") != "passed":
        raise SystemExit("Zip native PLY roundtrip gate has not passed")
    expected_sources = sorted(Path(args.source_scene).joinpath("images").glob("*.png"),
                              key=lambda p: p.name)
    for item in zip_roundtrip["source_images"]:
        if item["sha256"] != sha256(item["path"]):
            raise SystemExit("Zip roundtrip source image hash mismatch: " + item["path"])
    if zip_roundtrip["source_images"] != [
            {"path": str(p), "sha256": sha256(p)} for p in expected_sources]:
        raise SystemExit("Zip roundtrip is not on the exact ordered N1 sources")
    if zip_roundtrip["ply_sha256"] != sha256(zip_roundtrip["ply_path"]):
        raise SystemExit("Zip roundtrip PLY hash mismatch")
    if Path(zip_roundtrip["ply_path"]).resolve() != (Path(args.zip_result) / "gaussians_feedforward.ply").resolve():
        raise SystemExit("Zip roundtrip is not auditing the requested PLY")
    for key, limit in {"means": 1e-7, "scales": 1e-6, "quats": 1e-7,
                       "opacities": 1e-6, "sh": 1e-7}.items():
        if zip_roundtrip["tensor_deltas"][key] > limit:
            raise SystemExit("Zip PLY tensor delta exceeds gate: " + key)
    if zip_roundtrip["render_comparisons"]["ply_lowlevel_eps0.3"]["max"] > 3 / 255:
        raise SystemExit("Zip native low-level render gate has not passed")
    if not n0["summary"]["measurement_gates"]["manifest_file_mapping_verified"]:
        raise SystemExit("N0 measurement gate has not passed")
    manifest = json.loads((pkg / "cameras.json").read_text(encoding="utf-8"))
    cams_by_id = {int(c["id"]): c for c in manifest["cameras"]}
    K_gt = np.asarray(manifest["intrinsics"]["K"], np.float64)
    gt_size = (int(manifest["intrinsics"]["width"]), int(manifest["intrinsics"]["height"]))
    prep = M21.official_preprocess_geometry(K_gt, gt_size, (args.size, args.size))
    src_prior = json.loads((Path(args.source_scene) / "prior_camera.json").read_text(encoding="utf-8"))
    tgt_prior = json.loads((Path(args.target_scene) / "prior_camera.json").read_text(encoding="utf-8"))
    priors = {c["camera_id"]: np.asarray(c["matrix"], np.float64)
              for c in src_prior["extrinsics"] + tgt_prior["extrinsics"]}
    sources = [c["camera_id"] for c in src_prior["extrinsics"]]
    targets = [c["camera_id"] for c in tgt_prior["extrinsics"]]
    if set(sources) & set(targets):
        raise SystemExit("Source/target leakage")
    sa_pkg_to_model_reference = M15.estimate_sim3(sa_res, Path(args.source_scene))
    zip_pred = np.load(zip_res / "predicted_cameras.npz", allow_pickle=False)
    zip_run = json.loads((zip_res / "metrics.json").read_text(encoding="utf-8"))
    if zip_run["token_ba"] or list(zip_pred["names"]) != [s + ".png" for s in sources]:
        raise SystemExit("Zip must be ba=False on the exact ordered sources")
    zip_cams = {str(n)[:-4]: c for n, c in zip(zip_pred["names"], zip_pred["c2w"])}
    camera_delta = np.abs(zip_pred["c2w"] - np.asarray(zip_roundtrip["saved_c2w"], np.float64)).max()
    if camera_delta > 1e-6:
        raise SystemExit(f"Zip saved pose differs from native audit: {camera_delta}")
    for got, expected in zip(zip_pred["cameras"], zip_roundtrip["saved_cameras"]):
        if np.abs(got - np.asarray(expected)).max() > 1e-4:
            raise SystemExit("Zip saved intrinsics differ from native audit")
    zip_sim3 = fit_sim3(np.stack([priors[s][:3, 3] for s in sources]),
                        np.stack([zip_cams[s][:3, 3] for s in sources]))
    source_camera_residuals(np.stack([priors[s] for s in sources]),
                            np.stack([zip_cams[s] for s in sources]), zip_sim3, sources)
    zip_wahba = fit_wahba_center(np.stack([priors[s] for s in sources]),
                                 np.stack([zip_cams[s] for s in sources]))
    source_camera_residuals(np.stack([priors[s] for s in sources]),
                            np.stack([zip_cams[s] for s in sources]), zip_wahba, sources)
    sa_c2w, sa_K, sa_ids = M03.load_cameras(sa_res / "cameras.json")
    if sa_ids != sources:
        raise SystemExit("SA source order mismatch")
    # 共同评测优先把 GT 相机逆回模型系，严格复用 N0 的 model_camera。
    # 正向 Sim3 仅作为等价实现交叉检查，不让 float32 旋转噪声阻塞原生模型评测。
    sa_pkg_to_model = fit_sim3(
        sa_c2w[:, :3, 3], np.stack([priors[s][:3, 3] for s in sources]))
    sa_model_to_pkg = inverse_sim3(sa_pkg_to_model)
    for label, fit in (("model_to_gt", sa_model_to_pkg[3]),
                       ("gt_to_model", sa_pkg_to_model[3])):
        fit["orientation_residual_deg"] = {
            s: angle((sa_pkg_to_model[1] @ priors[s][:3, :3]).T @ cam[:3, :3])
            for s, cam in zip(sources, sa_c2w)}
    sa_sim3_reference = M15.estimate_sim3(sa_res, Path(args.source_scene))
    reference_delta = float(np.abs(sa_pkg_to_model[0] - sa_sim3_reference[0]))
    reference_delta = max(reference_delta, float(np.max(np.abs(
        sa_pkg_to_model[1] - sa_sim3_reference[1]))), float(np.max(np.abs(
        sa_pkg_to_model[2] - sa_sim3_reference[2]))))
    if reference_delta > 2e-6:
        raise SystemExit("Orthogonal SA Sim3 differs from N0 reference: "
                         + str(reference_delta))
    methods = {
        "SA_P0": {"ply": sa_res / "gaussians.ply", "sim3": sa_model_to_pkg,
                  "source_cams": dict(zip(sa_ids, sa_c2w)), "source_K": dict(zip(sa_ids, sa_K)),
                  "condition": "RGB + known pose", "stored_quat_order": "wxyz",
                  "native_size": args.size, "native_eps2d": .01},
        "Zip_FF": {"ply": zip_res / "gaussians_feedforward.ply", "sim3": zip_sim3,
                   "source_cams": zip_cams, "source_K": {}, "condition": "RGB only, ba=False",
                   "stored_quat_order": "wxyz", "native_size": 504, "native_eps2d": .3,
                   "alignment_rule": "preregistered source-only center Umeyama"},
        "Zip_FF_Wahba_sensitivity": {"ply": zip_res / "gaussians_feedforward.ply", "sim3": zip_wahba,
                   "source_cams": zip_cams, "source_K": {}, "condition": "RGB only, ba=False",
                   "stored_quat_order": "wxyz", "native_size": 504, "native_eps2d": .3,
                   "alignment_rule": "source-only Wahba orientation + center scale/translation; sensitivity only, no target-based selection",
                   "formal_ranking": False}}
    # 官方 renders/ 用 model_cameras@504²；保存的 cameras 是原图 1024² 内参。
    # 正方形输入无 crop，二者仅差统一像素缩放。
    for stem, cam in zip(zip_pred["names"], zip_pred["cameras"]):
        w, h, fx, fy, cx, cy = map(float, cam)
        if (round(w), round(h)) != gt_size:
            raise SystemExit("Zip predicted camera size differs from GT")
        native_size = methods["Zip_FF"]["native_size"]
        K = np.array([
            [fx * native_size / w, 0, cx * native_size / w],
            [0, fy * native_size / h, cy * native_size / h], [0, 0, 1]])
        for name in ("Zip_FF", "Zip_FF_Wahba_sensitivity"):
            methods[name]["source_K"][stem[:-4]] = K.copy()
    zip_camera_diagnostics = {
        "intrinsics": {stem: {
            "fx_ratio_pred_to_gt": float(zip_pred["cameras"][i, 2] / K_gt[0, 0]),
            "fy_ratio_pred_to_gt": float(zip_pred["cameras"][i, 3] / K_gt[1, 1]),
            "note": "predicted focal error is not repaired by source-only Sim3; common GT K is evaluation-only"
        } for i, stem in enumerate(sources)},
        "alignment_selection": "center Umeyama is primary; Wahba sensitivity is predeclared before its target renders; no target-based selection"
    }
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(args.device).eval()
    landmarks = json.loads(Path(args.landmarks).read_text(encoding="utf-8"))["views"]
    roi_qa = json.loads(Path(args.roi_qa).read_text(encoding="utf-8"))
    exclusions = {s: q.get("excluded_rois", []) for s, q in roi_qa["views"].items()}
    report = {"schema": "n1_head_compare_v1",
              "status": "single static synthetic identity; N2 remains frozen",
              "protocol": {"sources": sources, "targets": targets, "size": [args.size, args.size],
                           "background": 0, "eps2d": args.eps2d, "lowpass_sigma": 2,
                           "lpips": "AlexNet, GT mask bbox crop resized to 128 square",
                           "common_gt_camera_and_K": True, "target_fit_or_tuning": False,
                           "token_ba": False, "ply_concatenation": False, "training": False},
              "sim3": {k: {"s_R_T": list(m["sim3"]), "direction": "X_gt=s R X_model+T",
                           "common_evaluation_camera": "inverse rigid GT-to-model camera; native PLY; ED multiplied by s to metric GT",
                           "N0_reference_sim3_delta": reference_delta if k == "SA_P0" else None,
                           "alignment_rule": m.get("alignment_rule", "N0 source-only rigid camera Sim3"),
                           "formal_ranking": m.get("formal_ranking", True)}
                       for k, m in methods.items()},
              "limitations": ["one identity/two held-out views does not select an N2 training base",
                              "SA known pose vs Zip RGB-only; common evaluation controls cameras, not learned priors",
                              "Zip scale comes from source-camera alignment, not independently certified metric scale",
                              "ED is alpha-normalized axial Z; depth-cloud nearest distance is not independent FBX mesh truth",
                              "head mask includes hair; tight face/visibility proxy is not skin segmentation",
                              "alpha silhouette mixes scale and coverage; lowpass is only a visual aid"],
              "input_sha256": {}, "views": {},
              "zip_source_camera_diagnostics": zip_camera_diagnostics}
    for name, m in methods.items():
        m["g"] = load_ply(m["ply"], m["stored_quat_order"])
        m["gpu_native"] = to_gpu(m["g"], args.device)
        m["g_gt"] = transform_gaussians(m["g"], m["sim3"])
        m["gpu"] = to_gpu(m["g_gt"], args.device)
        report["input_sha256"][str(m["ply"])] = sha256(m["ply"])
        report.setdefault("model_conditions", {})[name] = {
            "input": m["condition"], "gaussians": len(m["g"]["means"]), "sh_degree": m["g"]["degree"],
            "stored_quat_order": "wxyz", "sim3_composition_order": "xyzw", "gsplat_quat_order": "wxyz",
            "native_render_size": m["native_size"], "native_eps2d": m["native_eps2d"]}
    for path in (sa_res / "cameras.json", zip_res / "predicted_cameras.npz", zip_res / "metrics.json",
                 Path(args.source_scene) / "prior_camera.json", Path(args.target_scene) / "prior_camera.json",
                 Path(args.landmarks), Path(args.roi_qa), Path(args.n0_audit)):
        report["input_sha256"][str(path)] = sha256(path)
    # 三个概念分开验收：原生 PLY 回放、原生与 Sim(3) 的数学等变性、共同 GT 渲染。
    # ED 等变门只看双方 alpha>.01；无 alpha 像素的归一化深度没有稳定几何意义。
    # Zip 原生保留 504²/eps=.3；共同评测统一 756²/eps=.01，不混称原生参数。
    if not args.skip_render_validation:
        validations = {}
        vdir = Path(args.render_validation_dir)
        if vdir.exists():
            raise SystemExit("Validation output already exists: " + str(vdir))
        vdir.mkdir(parents=True)
        for name, m in methods.items():
            validations[name] = {}
            native_size = m["native_size"]
            for stem in sources:
                reference = (Path(args.n0_audit).parent / f"{stem}_render_export.png"
                             if name == "SA_P0" else zip_res / f"renders/{sources.index(stem):03d}.png")
                ref = np.asarray(Image.open(reference).convert("RGB"), np.float32) / 255
                native, native_alpha, native_ed = render(
                    m["gpu_native"], m["g"]["degree"], m["source_cams"][stem],
                    m["source_K"][stem], native_size, args.device, m["native_eps2d"])
                if ref.shape != native.shape:
                    raise SystemExit(f"Native reference size mismatch: {name}: {ref.shape} vs {native.shape}")
                delta = float(np.abs(native - ref).max())
                mae = float(np.abs(native - ref).mean())
                transformed, alpha_gt, ed_gt = render(
                    m["gpu"], m["g"]["degree"],
                    transform_camera(m["source_cams"][stem], m["sim3"]), m["source_K"][stem],
                    native_size, args.device, m["native_eps2d"])
                eq_delta = float(np.abs(transformed - native).max())
                alpha_delta = float(np.abs(native_alpha - alpha_gt).max())
                covered = (native_alpha > .01) & (alpha_gt > .01)
                if not covered.any():
                    raise SystemExit("Sim3 equivariance has no covered pixels: " + name)
                if name.startswith("Zip_FF"):
                    camera_inv = M21.model_camera(
                        m["source_cams"][stem], 1.0 / m["sim3"][0], m["sim3"][1].T,
                        -(m["sim3"][1].T @ m["sim3"][2]) / m["sim3"][0])
                    camera_error = float(np.abs(camera_inv - transform_camera(
                        m["source_cams"][stem], inverse_sim3(m["sim3"]))).max())
                    if camera_error > 1e-9:
                        raise SystemExit("Zip inverse-camera formula mismatch: " + name)
                else:
                    camera_error = None
                ed_error = np.abs(native_ed * m["sim3"][0] - ed_gt)
                record = {
                    "reference": str(reference), "native_rgb_max_abs": delta,
                    "native_rgb_mae": mae, "equivariance_rgb_max_abs": eq_delta,
                    "equivariance_rgb_mae": float(np.abs(transformed - native).mean()),
                    "equivariance_alpha_max_abs": alpha_delta,
                    "equivariance_alpha_mean_abs": float(np.abs(native_alpha - alpha_gt).mean()),
                    "equivariance_ED_max_abs_covered": float(ed_error[covered].max()),
                    "equivariance_ED_mean_abs_covered": float(ed_error[covered].mean()),
                    "equivariance_ED_p99_9_abs_covered": float(np.percentile(ed_error[covered], 99.9)),
                    "equivariance_ED_max_abs_all_pixels": float(ed_error.max()),
                    "covered_pixels": int(covered.sum()),
                    "camera_inverse_formula_max_abs": camera_error,
                    "native_rgb_gate": 3 / 255,
                    "equivariance_rgb_max_gate": 3 / 255,
                    "equivariance_rgb_mean_gate": 2e-5,
                    "equivariance_alpha_mean_gate": 2e-5,
                    "equivariance_ED_mean_gate_m": 2e-5,
                    "equivariance_ED_p99_9_gate_m": 1e-3,
                    "size": native_size, "eps2d": m["native_eps2d"],
                    "coordinate_transform": "X_gt=s R X_model+T; degree-1 SH also rotated",
                    "ED_gate_scope": "both alpha>.01; all-pixel extrema retained as diagnostic only"}
                validations[name][stem] = record
                if name == "SA_P0":
                    record["forward_sim3_usage"] = (
                        "diagnostic only; not accepted as a passed gate. Formal evaluation uses "
                        "native PLY + N0 model_camera + inverse GT camera, with per-view N0 ED gate")
                Image.fromarray(np.uint8(native * 255)).save(vdir / f"{stem}_{name}_native.png")
                Image.fromarray(np.uint8(transformed * 255)).save(vdir / f"{stem}_{name}_sim3.png")
                print("RENDER VALIDATION", name, stem, record, flush=True)
                if delta > 3 / 255:
                    raise SystemExit("Native PLY replay failed: " + name + " " + stem)
                if name.startswith("Zip_FF") and (
                        eq_delta > 3 / 255
                        or record["equivariance_rgb_mae"] > 2e-5
                        or record["equivariance_alpha_mean_abs"] > 2e-5
                        or record["equivariance_ED_mean_abs_covered"] > 2e-5
                        or record["equivariance_ED_p99_9_abs_covered"] > 1e-3):
                    raise SystemExit("Sim3 render equivariance failed: " + name + " " + stem)
        sa_model = M21.model_camera(priors[sources[0]], *sa_pkg_to_model[:3])
        export_delta = float(np.max(np.abs(
            methods["SA_P0"]["source_cams"][sources[0]] - sa_model)))
        if export_delta > 2e-6:
            raise SystemExit("SA export camera differs from N0 pkg-to-model Sim3: "
                             + str(export_delta))
        roundtrip_delta = float(np.max(np.abs(
            transform_camera(methods["SA_P0"]["source_cams"][sources[0]],
                             methods["SA_P0"]["sim3"]) - priors[sources[0]])))
        if roundtrip_delta > 1e-6:
            raise SystemExit("SA model-to-GT camera roundtrip failed: "
                             + str(roundtrip_delta))
        base = M03.load_ply(sa_res / "gaussians.ply")
        ed_ref, ref_alpha = M13.render_expected_depth(
            base, sa_model, prep["K_reference"], args.size, args.size,
            device=args.device, eps2d=args.eps2d)
        ed_ref, ref_alpha = ed_ref[0], ref_alpha[0]
        raw_native = M03.render(
            base, sa_model, prep["K_reference"], args.size, args.size,
            background=0, device=args.device)[0][0]
        common_native, native_alpha, native_ed = render(
            methods["SA_P0"]["gpu_native"], methods["SA_P0"]["g"]["degree"],
            methods["SA_P0"]["source_cams"][sources[0]], prep["K_reference"],
            args.size, args.device, args.eps2d)
        covered_native = (ref_alpha[..., 0] > .01) & (native_alpha > .01)
        native_error = np.abs(native_ed - ed_ref)
        cross_record = {
            "camera_export_vs_N0_max_abs": export_delta,
            "camera_model_to_GT_roundtrip_max_abs": roundtrip_delta,
            "native_loader_vs_N0_ED_max_abs_covered": float(native_error[covered_native].max()),
            "native_loader_vs_N0_ED_mean_abs_covered": float(native_error[covered_native].mean()),
            "native_loader_vs_N0_ED_p99_9_abs_covered": float(np.percentile(native_error[covered_native], 99.9)),
            "native_loader_vs_N0_ED_max_abs_all_pixels": float(native_error.max()),
            "native_loader_vs_N0_RGB_max_abs": float(np.abs(common_native - raw_native).max()),
            "gate_scope": "both N0 and native render alpha>.01; RGB gate also exact native reference",
            "native_reference": "M03.load_ply + M03.render; same camera as M13 N0 reference"}
        validations["SA_ED_vs_N0"] = cross_record
        print("RENDER VALIDATION SA_ED_vs_N0", cross_record, flush=True)
        if (cross_record["native_loader_vs_N0_ED_mean_abs_covered"] > 2e-5
                or cross_record["native_loader_vs_N0_ED_p99_9_abs_covered"] > 1e-3
                or cross_record["native_loader_vs_N0_RGB_max_abs"] > 3 / 255):
            raise SystemExit("Expected depth differs from N0")
        write_json(vdir / "render_validation.json", {
            "status": "passed for the implemented native-camera evaluation chain",
            "checks": validations,
            "scope": "native replay for both; Zip forward-Sim3 gate; SA forward-Sim3 diagnostics and N0 ED cross-check; five-view SA N0 ED gate; not model acceptance",
            "common_evaluation": "native PLY, inverse rigid GT-to-model camera, ED multiplied by source-only scale to metric GT"})
    report["render_validation_dir"] = str(Path(args.render_validation_dir))
    report["render_validation_skipped"] = bool(args.skip_render_validation)
    # 共同相机先逆回各自原生模型系：SA 使用与 N0 完全相同的 model_camera。
    # 仅在世界刚体旋转/相机中心尺度后渲染，深度乘 s 回米制；SH 不需要旋转。
    # 正式渲染不经过 Sim3 对 means/scales/quats/SH 的 float32 往返。
    common_g = {name: m["gpu_native"] for name, m in methods.items()}
    # source 深度尺度是独立于相机中心的诊断，不是正式对齐择优。
    # 先看三源原始 ED 求一个倍率；source/target 分流在后续循环中严格执行。
    source_scale_input = {name: [] for name in methods if name.startswith("Zip_FF")}
    for stem in sources:
        numeric = int(stem.rsplit("_", 1)[1])
        mc = cams_by_id[numeric]
        _, alpha_native = M21.load_rgb(pkg / mc["file_rgb"], gt_size)
        depth_native, valid_native = M21.load_depth(pkg / mc["file_depth"])
        valid_native &= alpha_native > .999
        depth = M21.remap_nearest(np.where(valid_native, depth_native, 0).astype(np.float32), prep)
        valid = M21.remap_nearest(valid_native, prep)
        points = np.asarray(landmarks[stem]["points"], np.float64)
        face = M21.remap_nearest(M21.region_masks(points, depth_native.shape)["face_tight_landmark"], prep) & valid
        for name, m in methods.items():
            if name not in source_scale_input:
                continue
            s_gt, R_gt, T_gt, _ = m["sim3"]
            camera = M21.model_camera(priors[stem], 1.0 / s_gt, R_gt.T,
                                      -(R_gt.T @ T_gt) / s_gt)
            _, _, ed_model = render(common_g[name], m["g"]["degree"], camera,
                                    prep["K_reference"], args.size, args.device, args.eps2d)
            source_scale_input[name].append((ed_model, depth, face, s_gt))
    report["source_depth_scale_diagnostic"] = {}
    for name, items in source_scale_input.items():
        ed = np.concatenate([x[0][x[2]] for x in items])
        depth = np.concatenate([x[1][x[2]] for x in items])
        if len({x[3] for x in items}) != 1:
            raise RuntimeError("Inconsistent source-only scale")
        report["source_depth_scale_diagnostic"][name] = fit_source_depth_scale(
            ed, depth, np.ones_like(ed, dtype=bool), items[0][3])
        report["source_depth_scale_diagnostic"][name]["per_source"] = {
            stem: fit_source_depth_scale(x[0], x[1], x[2], x[3])
            for stem, x in zip(sources, items)}
    rows = []
    from scipy.spatial import cKDTree
    for stem in sources + targets:
        numeric = int(stem.rsplit("_", 1)[1])
        mc = cams_by_id[numeric]
        gt_native, alpha_native = M21.load_rgb(pkg / mc["file_rgb"], gt_size)
        gt = M21.official_preprocess_rgb(pkg / mc["file_rgb"], prep)
        depth_native, valid_native = M21.load_depth(pkg / mc["file_depth"])
        valid_native &= alpha_native > .999
        depth = M21.remap_nearest(np.where(valid_native, depth_native, 0).astype(np.float32), prep)
        valid = M21.remap_nearest(valid_native, prep)
        points = np.asarray(landmarks[stem]["points"], np.float64)
        rois_native = M21.region_masks(points, depth_native.shape)
        rois = {k: M21.remap_nearest(v, prep) for k, v in rois_native.items()}
        mask_dir = sa_res / "face_opt" / ("masks" if stem in sources else "masks_target")
        person = np.asarray(Image.open(mask_dir / f"{stem}_person.png")) > 127
        if person.shape != valid.shape:
            person = cv2.resize(person.astype(np.uint8), (args.size, args.size), interpolation=cv2.INTER_NEAREST).astype(bool)
        person &= valid
        hbox, head = head_mask(points * args.size / gt_size[0], person, args.size)
        visible = M21.remap_nearest(rois_native["face_tight_landmark"] & M21.normal_visibility_mask(
            depth_native, valid_native, K_gt, priors[stem], rois_native, exclusions.get(stem, [])), prep) & valid
        regions = {"full_valid": valid, "person": person, "head": head,
                   "face_tight": rois["face_tight_landmark"] & valid, "face_visibility_filtered": visible}
        for name in M21.REGIONS:
            if name not in exclusions.get(stem, []):
                regions[name] = rois[name] & valid
        yy, xx = np.where(rois_native["face_tight_landmark"])
        fbox = [int(xx.min() * args.size / gt_size[0]), int(yy.min() * args.size / gt_size[1]),
                int((xx.max() + 1) * args.size / gt_size[0]), int((yy.max() + 1) * args.size / gt_size[1])]
        ys, xs = np.where(valid_native)
        take = np.linspace(0, len(xs) - 1, min(120000, len(xs))).astype(np.int64)
        ys, xs = ys[take], xs[take]
        pc = (np.stack([xs, ys, np.ones_like(xs)], 1) * depth_native[ys, xs, None]) @ np.linalg.inv(K_gt).T
        surface = cKDTree(pc @ priors[stem][:3, :3].T + priors[stem][:3, 3])
        view = {"id": numeric, "is_source": stem in sources, "excluded_rois": exclusions.get(stem, []),
                "files": {k: mc[k] for k in ("file_rgb", "file_depth", "file_normal")},
                "roi_pixels": {k: int(v.sum()) for k, v in regions.items()},
                "head_box": hbox, "face_box": fbox, "methods": {}}
        rendered = {}
        for name, m in methods.items():
            s_gt, R_gt, T_gt, _ = m["sim3"]
            model_cam = M21.model_camera(priors[stem], 1.0 / s_gt, R_gt.T,
                                         -(R_gt.T @ T_gt) / s_gt)
            rd, al, ed_model = render(
                common_g[name], m["g"]["degree"], model_cam,
                prep["K_reference"], args.size, args.device, args.eps2d)
            ed = ed_model * s_gt
            rd_forward, al_forward, ed_forward = render(
                m["gpu"], m["g"]["degree"], priors[stem],
                prep["K_reference"], args.size, args.device, args.eps2d)
            ed_pair = np.abs(ed_model - ed_forward / s_gt)
            covered_pair = (al > .01) & (al_forward > .01)
            implementation_check = {
                "forward_sim3_vs_inverse_camera_rgb_max_abs": float(np.abs(rd - rd_forward).max()),
                "forward_sim3_vs_inverse_camera_rgb_mean_abs": float(np.abs(rd - rd_forward).mean()),
                "forward_sim3_vs_inverse_camera_ED_mean_abs_covered": float(ed_pair[covered_pair].mean()),
                "forward_sim3_vs_inverse_camera_ED_p99_9_abs_covered": float(np.percentile(ed_pair[covered_pair], 99.9)),
                "forward_sim3_vs_inverse_camera_ED_max_abs_all_pixels": float(ed_pair.max()),
                "ED_unit": "model units",
                "note": "two implementations of one rigid/similarity light path; RGB extrema retained, no model ranking here"}
            if name == "SA_P0":
                # 正式 SA 深度以 N0 的 PLY/相机实现为准，不让第二读取器的
                # exp/sigmoid 重舍入混入被比较的模型。
                base = M03.load_ply(m["ply"])
                base_ed, base_al = M13.render_expected_depth(
                    base, model_cam, prep["K_reference"], args.size, args.size,
                    device=args.device, eps2d=args.eps2d)
                base_rgb, _ = M03.render(
                    base, model_cam, prep["K_reference"], args.size, args.size,
                    background=0, device=args.device)
                base_rgb, base_ed = base_rgb[0], base_ed[0]
                n0_delta = np.abs(ed_model - base_ed)
                covered_n0 = (al > .01) & (base_al[0, ..., 0] > .01)
                implementation_check["N0_ED_max_abs_covered"] = float(n0_delta[covered_n0].max())
                implementation_check["N0_ED_mean_abs_covered"] = float(n0_delta[covered_n0].mean())
                implementation_check["N0_ED_p99_9_abs_covered"] = float(np.percentile(n0_delta[covered_n0], 99.9))
                implementation_check["N0_RGB_max_abs"] = float(np.abs(rd - base_rgb).max())
                implementation_check["N0_RGB_mean_abs"] = float(np.abs(rd - base_rgb).mean())
                implementation_check["N0_reference_render"] = "same original PLY + M21.model_camera + M13 RGB+ED"
                if (implementation_check["N0_ED_mean_abs_covered"] > 2e-5
                        or implementation_check["N0_ED_p99_9_abs_covered"] > 1e-3
                        or implementation_check["N0_RGB_max_abs"] > 3 / 255):
                    raise SystemExit("Common evaluation SA ED differs from N0: " + stem)
                # N0 原生读取器是唯一正式 SA 输出；通用读取器仅作敏感性诊断。
                rd, al, ed = base_rgb, base_al[0, ..., 0], base_ed * s_gt
            rendered[name] = (rd, al, ed)
            result = {"appearance": {k: roi_metrics(gt, rd, al, mask, lp, args.device) for k, mask in regions.items()},
                      "depth": {k: depth_metrics(ed, depth, mask) for k, mask in regions.items()},
                      "alpha_gt_mean": float(al[valid].mean()), "alpha_gt_coverage_05": float((al[valid] > .5).mean()),
                      "silhouette": silhouette(valid, al), "centers_to_depth_backprojection": {},
                      "implementation_check": implementation_check}
            if name.startswith("Zip_FF"):
                diagnostic = report["source_depth_scale_diagnostic"][name]
                extra_scale = diagnostic["additional_ratio_of_medians"]
                result["source_depth_scale_sensitivity"] = {
                    "frozen_from_sources": True, "additional_ratio": extra_scale,
                    "face_tight": depth_metrics(ed * extra_scale, depth, regions["face_tight"]),
                    "nose_tip": depth_metrics(ed * extra_scale, depth, regions["nose_tip"]) if "nose_tip" in regions else None,
                    "note": "same frozen ratio for source and target; diagnostic only, no target-fitted oracle"}
                native_rd, native_al, native_ed = render(
                    common_g[name], m["g"]["degree"], model_cam,
                    prep["K_reference"], args.size, args.device, m["native_eps2d"])
                result["native_eps_sensitivity"] = {
                    "eps2d": m["native_eps2d"], "common_eps2d": args.eps2d,
                    "note": "native eps is diagnostic at common resolution, not native 504 replay",
                    "face_tight_depth": depth_metrics(native_ed * s_gt, depth, regions["face_tight"]),
                    "face_tight_rgb": roi_metrics(gt, native_rd, native_al, regions["face_tight"], lp, args.device)}
            centers = common_g[name]["means"].detach().cpu().numpy().astype(np.float64)
            centers = (s_gt * (centers @ R_gt.T) + T_gt).astype(np.float64)
            pc_all = (centers - priors[stem][:3, 3]) @ priors[stem][:3, :3]
            u = K_gt[0, 0] * pc_all[:, 0] / np.maximum(pc_all[:, 2], 1e-12) + K_gt[0, 2]
            v = K_gt[1, 1] * pc_all[:, 1] / np.maximum(pc_all[:, 2], 1e-12) + K_gt[1, 2]
            ui, vi = np.floor(u + .5).astype(int), np.floor(v + .5).astype(int)
            inside = (pc_all[:, 2] > 0) & (ui >= 0) & (ui < gt_size[0]) & (vi >= 0) & (vi < gt_size[1])
            for region in ("face_tight_landmark", "nose_tip"):
                if region in exclusions.get(stem, []):
                    continue
                selected = inside.copy()
                selected[inside] &= rois_native[region][vi[inside], ui[inside]]
                dist = surface.query(centers[selected])[0] if selected.any() else np.array([])
                result["centers_to_depth_backprojection"][region] = {
                    "n": int(selected.sum()), "median_mm": float(np.median(dist) * 1000) if len(dist) else None,
                    "p90_mm": float(np.percentile(dist, 90) * 1000) if len(dist) else None,
                    "definition": "all native centers in ROI to depth cloud; no visibility or material assignment"}
            if stem in sources:
                own_rd, own_al, _ = render(
                    m["gpu"], m["g"]["degree"],
                    transform_camera(m["source_cams"][stem], m["sim3"]),
                    np.diag([args.size / m["native_size"], args.size / m["native_size"], 1])
                    @ m["source_K"][stem], args.size, args.device, args.eps2d)
                result["source_camera_diagnostic"] = {
                    "own_pose_K_full_psnr": M03.psnr(own_rd, gt), "common_gt_pose_K_full_psnr": M03.psnr(rd, gt),
                    "own_pose_K_gt_valid": roi_metrics(gt, own_rd, own_al, valid, lp, args.device),
                    "note": "source pose/K residual; not held-out geometry proof"}
                Image.fromarray(np.uint8(own_rd * 255)).save(out / f"{stem}_{name}_own_pose.png")
            Image.fromarray(np.uint8(rd * 255)).save(out / f"{stem}_{name}.png")
            Image.fromarray(np.uint8(al * 255)).save(out / f"{stem}_{name}_alpha.png")
            Image.fromarray(heat(gt, ed - depth, valid & rois["face_tight_landmark"])).save(out / f"{stem}_{name}_ed_face.png")
            view["methods"][name] = result
            for region, values in result["appearance"].items():
                rows.append({"view": stem, "is_source": stem in sources, "method": name, "region": region,
                             **{k: v for k, v in values.items() if k in ("n", "psnr", "mae", "alpha_mean", "coverage_05", "lpips_crop128")}})
            print(stem, name, "face", result["appearance"]["face_tight"], "depth", result["depth"]["face_tight"], flush=True)
        save_comparison(out, stem, gt, rendered, depth, valid, fbox, hbox)
        report["views"][stem] = view
        for path in (pkg / mc["file_rgb"], pkg / mc["file_depth"], mask_dir / f"{stem}_person.png"):
            report["input_sha256"][str(path)] = sha256(path)
    report["summary"] = {
        "status": "N1 single-package comparison complete; no universal winner; N2 frozen",
        "metric_scope": "ED only on valid GT and ED>1e-5 within the stated ROI; full_valid excludes invalid background; ROI alpha coverage is separate from whole-GT coverage"
    }
    for name in methods:
        report["summary"][name] = {}
        for split, stems in (("source", sources), ("target", targets)):
            report["summary"][name][split] = {}
            for region in ("full_valid", "person", "head", "face_tight", "nose_tip"):
                values = [report["views"][s]["methods"][name] for s in stems]
                valid_psnr = [v["appearance"][region].get("psnr") for v in values if v["appearance"][region].get("psnr") is not None]
                valid_depth = [v["depth"][region].get("mae_mm") for v in values if v["depth"][region].get("mae_mm") is not None]
                valid_lpips = [v["appearance"][region].get("lpips_crop128") for v in values if v["appearance"][region].get("lpips_crop128") is not None]
                report["summary"][name][split][region] = {
                    "n_views": len(valid_psnr), "psnr_mean": float(np.mean(valid_psnr)) if valid_psnr else None,
                    "lpips_crop128_mean": float(np.mean(valid_lpips)) if valid_lpips else None,
                    "ed_mae_mm_mean": float(np.mean(valid_depth)) if valid_depth else None,
                    "roi_alpha_mean": float(np.mean([v["appearance"][region]["alpha_mean"] for v in values])),
                    "roi_alpha_coverage_05_mean": float(np.mean([v["appearance"][region]["coverage_05"] for v in values])),
                    "whole_gt_valid_alpha_coverage_05_mean": float(np.mean([v["alpha_gt_coverage_05"] for v in values]))}
        worst = min(targets, key=lambda s: report["views"][s]["methods"][name]["appearance"]["face_tight"]["psnr"])
        report["summary"][name]["worst_target_face_view"] = worst
    write_json(out / "n1_head_compare.json", report)
    with open(out / "per_view_metrics.csv", "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
