#!/usr/bin/env python3
"""ZipSplat：输入视角自洽 + GT 相机对齐的独立留出视角评测。

评测集不参与预测/TokenBA；仅用于将 GT 相机通过输入相机 Sim(3) 对齐到预测坐标系。
3 视角近平面轨迹只报告带退化标志的对齐指标，不把它冒充通用 NVS 基准。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from skimage.metrics import structural_similarity

# 只复用已核对方向的 Umeyama；不把 plyfile / matplotlib 拉进官方推理进程。
def fit_sim3(src, dst):
    """返回 dst ≈ s·R·src+t；src/dst 为 N×3 相机中心，R 为左手行向量实现。"""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    U, sig, Vt = np.linalg.svd(S.T @ D / len(src))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    s = float((sig * np.array([1.0, 1.0, d])).sum() / (S**2).sum() * len(src))
    return s, R, md - s * (R @ ms)


def apply_sim3(c2w, s, R, t):
    out = np.array(c2w, dtype=np.float64, copy=True)
    out[:, :3, :3] = np.einsum("ij,njk->nik", R, out[:, :3, :3])
    out[:, :3, 3] = s * (out[:, :3, 3] @ R.T) + t
    return out.astype(np.float32)


def wahba(Rp, Rg):
    M = np.einsum("nij,nkj->ik", Rp, Rg)
    U, _, Vt = np.linalg.svd(M)
    return Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def load_ply(path):
    p = PlyData.read(str(path))["vertex"]
    n = len(p["x"])
    K = 1 + sum(k.startswith("f_rest_") for k in p.data.dtype.names) // 3
    sh = np.empty((n, K, 3), np.float32)
    sh[:, 0] = np.stack([p[f"f_dc_{i}"] for i in range(3)], 1)
    # 官方 3DGS 文件约定：channel-first (3,K-1) 后 flatten；官方 Gaussians 内存是 (K,3)。
    sh[:, 1:] = np.stack([p[f"f_rest_{i}"] for i in range(3 * (K - 1))], 1).reshape(n, 3, K - 1).transpose(0, 2, 1)
    return {
        "means": np.stack([p[k] for k in ("x", "y", "z")], 1),
        "scales": np.exp(np.stack([p[f"scale_{i}"] for i in range(3)], 1)),
        "quats": np.stack([p[f"rot_{i}"] for i in range(4)], 1),
        "opacities": 1 / (1 + np.exp(-p["opacity"].astype(np.float32))),
        "sh": sh,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--source-cameras", required=True)
    ap.add_argument("--target-cameras", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ply", default=None, help="默认 result/gaussians.ply，缺失则用 feedforward")
    ap.add_argument("--lpips", action="store_true")
    args = ap.parse_args()
    result = Path(args.result)
    out = Path(args.out) if args.out else result / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    pred = np.load(result / "predicted_cameras.npz", allow_pickle=False)
    source = np.load(args.source_cameras, allow_pickle=False)
    target = np.load(args.target_cameras, allow_pickle=False)
    assert len(pred["c2w"]) == len(source["c2w"])
    assert list(pred["names"]) == list(source["names"])
    assert set(source["names"]).isdisjoint(set(target["names"])), "评测帧泄漏进输入"
    P = pred["c2w"][:, :3, 3].astype(np.float64)
    G = source["c2w"][:, :3, 3].astype(np.float64)
    scale, _, _ = fit_sim3(G, P)
    diag_s, _, _ = fit_sim3(P, G)
    # 环绕采集的中心轨迹可能近平面；中心 Umeyama 的旋转会与 Wahba 相差几十度。
    # 用全部相机朝向估计全局旋转，尺度/平移仍只来自输入相机中心。
    Rw = wahba(pred["c2w"][:, :3, :3].astype(np.float64), source["c2w"][:, :3, :3].astype(np.float64))
    R, t = Rw.T, P.mean(0) - scale * (G.mean(0) @ Rw)
    diag_R, diag_t = Rw, G.mean(0) - diag_s * (P.mean(0) @ Rw.T)
    aligned_centers = diag_s * (P @ diag_R.T) + diag_t
    terr = np.linalg.norm(aligned_centers - G, axis=1)
    sv = np.linalg.svd(G - G.mean(0), compute_uv=False) / np.sqrt(len(G))
    rerr = [rot_deg(source["c2w"][i, :3, :3] @ (Rw @ pred["c2w"][i, :3, :3]).T)
            for i in range(len(P))]
    fx_rel = np.abs(pred["cameras"][:, 2] - source["intrinsics"][:, 0, 0]) / source["intrinsics"][:, 0, 0]
    fy_rel = np.abs(pred["cameras"][:, 3] - source["intrinsics"][:, 1, 1]) / source["intrinsics"][:, 1, 1]
    c2w_target = apply_sim3(target["c2w"], scale, R, t)
    np.savez(out / "target_cameras_aligned.npz", c2w=c2w_target,
             intrinsics=target["intrinsics"], names=target["names"])
    summary = {
        "protocol": "source cameras only: global Wahba rotation + center-derived scale/translation align GT target poses; targets never enter prediction or BA",
        "num_source_views": len(P), "num_target_views": len(target["names"]),
        "sim3_gt_to_pred_scale": scale, "sim3_pred_to_gt_scale": diag_s, "ate_rmse_m": float(np.sqrt(np.mean(terr**2))),
        "center_pca_std_m": sv.tolist(), "planar_degenerate": bool(sv[-1] < 0.1 * sv[0]),
        "wahba_rot_deg_mean": float(np.mean(rerr)), "wahba_rot_deg_median": float(np.median(rerr)),
        "fx_rel_error_mean": float(fx_rel.mean()), "fy_rel_error_mean": float(fy_rel.mean()),
        "source_reconstruction_fullres": {}, "target_nvs": {},
    }
    print(f"🔗 GT→pred scale={scale:.4f}; ATE={summary['ate_rmse_m']:.4f}m; "
          f"Wahba={np.mean(rerr):.3f}deg; planar={summary['planar_degenerate']}", flush=True)

    ply = Path(args.ply) if args.ply else result / "gaussians.ply"
    if not ply.is_file():
        ply = result / "gaussians_feedforward.ply"
    g = load_ply(ply)
    N, K = g["means"].shape[0], g["sh"].shape[1]
    degree = int(np.sqrt(K)) - 1
    assert degree >= 0 and (degree + 1)**2 == K
    for k in g:
        g[k] = torch.from_numpy(np.ascontiguousarray(g[k])).cuda().float()
    from gsplat import rasterization

    def render(c2w, intrinsics, w, h):
        colors = g["sh"][None].expand(1, -1, -1, -1).contiguous()
        rgb, alpha, _ = rasterization(
            means=g["means"], quats=g["quats"], scales=g["scales"], opacities=g["opacities"],
            colors=colors, viewmats=torch.from_numpy(np.linalg.inv(c2w[None]).astype(np.float32)).cuda(),
            Ks=torch.from_numpy(intrinsics[None]).cuda(), width=w, height=h,
            sh_degree=degree, render_mode="RGB", packed=False,
        backgrounds=torch.zeros((1, 3), device="cuda"))
        return rgb[0].clamp(0, 1).cpu().numpy(), alpha[0, :, :, 0].cpu().numpy()

    lp = None
    if args.lpips:
        import lpips
        lp = lpips.LPIPS(net="vgg").cuda().eval()

    def score_set(kind, images_dir, c2ws, Ks, names):
        rows = []
        (out / kind).mkdir(exist_ok=True)
        for i, name in enumerate(names):
            gt = np.asarray(Image.open(Path(images_dir) / str(name)).convert("RGB"), np.float32) / 255
            h, w = gt.shape[:2]
            rd, al = render(c2ws[i], Ks[i], w, h)
            psnr = float(-10 * np.log10(np.maximum(np.mean((gt - rd)**2), 1e-12)))
            ssim = float(structural_similarity(gt, rd, channel_axis=2, data_range=1.0))
            row = {"name": str(name), "psnr": psnr, "ssim": ssim,
                   "alpha_mean": float(al.mean()), "coverage_05": float((al > 0.5).mean())}
            if lp:
                with torch.no_grad():
                    a = torch.from_numpy(gt).permute(2, 0, 1)[None].cuda() * 2 - 1
                    b = torch.from_numpy(rd).permute(2, 0, 1)[None].cuda() * 2 - 1
                    row["lpips"] = float(lp(a, b).item())
            rows.append(row)
            Image.fromarray((rd * 255).round().astype(np.uint8)).save(out / kind / f"{i:03d}.png")
            print(f"  {kind} [{i}] PSNR={psnr:.3f} SSIM={ssim:.4f} cov={row['coverage_05']:.3f}", flush=True)
        return {"views": rows,
                "psnr_mean": float(np.mean([r["psnr"] for r in rows])),
                "ssim_mean": float(np.mean([r["ssim"] for r in rows])),
                "coverage_05_mean": float(np.mean([r["coverage_05"] for r in rows])),
                "lpips_mean": float(np.mean([r["lpips"] for r in rows])) if args.lpips else None}

    summary["gaussians"] = N
    summary["sh_degree"] = degree
    summary["source_reconstruction_fullres"] = score_set(
        "source", Path(args.source_cameras).parent / "images", pred["c2w"],
        np.stack([np.array([[c[2], 0, c[4]], [0, c[3], c[5]], [0, 0, 1]], np.float32)
                  for c in pred["cameras"]]), pred["names"])
    summary["target_nvs"] = score_set(
        "target", Path(args.target_cameras).parent / "images", c2w_target,
        target["intrinsics"], target["names"])
    with open(out / "metrics.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("🎉 " + json.dumps({"source_fullres": summary["source_reconstruction_fullres"],
                              "target": summary["target_nvs"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
