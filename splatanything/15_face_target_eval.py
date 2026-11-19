#!/usr/bin/env python3
"""15_face_target_eval.py — held-out target 视角评估 P0~P3 回退 PLY（target 只评分，不进优化）。

对应 da3_human/EXPERIMENT_PLAN.md 的"每包 3 source + 至少 2 独立 target"验收：
把 14 各臂导出的回退 PLY 在 **source 之外的 target 帧相机**下渲染，与 target GT 对拍。
target 的相机来自同一采集包 transforms.json（07 同一链路去畸变），但**不能直接渲染**：
SA 推理时对世界做了全局 Sim(3) 归一化（实测 s=0.5019，R=res[0]@inv(prior[0])，
残差 Δt≤2.5e-5——见 EXPERIMENTS.md），PLY 在模型系。脚本用 3 对 source 相机
（prior_camera.json ↔ cameras.json）估计该 Sim(3) 并转换 target 相机到模型系，
报告拟合残差（计划纪律：只用 source 对估计，target 只评分）。
K 不动：Sim(3) 世界缩放不改变像素内参，target 用去畸变后的原生 K + 原生分辨率。

指标（逐 target 视角，ROI 逐视角定框——固定框纪律同 14）：
  full PSNR/SSIM、face ROI PSNR/SSIM/LPIPS(可选)、脸锐度比、person(非脸) PSNR
判定：Δ(arm − P0) 在 ≥2 个独立 target 上方向一致才算泛化证据；单 target 不一致即存疑。

用法：
  python 15_face_target_eval.py --result <02输出目录> --target <target场景目录> \
      --masks <target mask目录> [--iters 300] [--arms p0,p1,p2,p3]
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

_RE14 = None


def re14():
    """复用 14_face_local_opt.py 的 Renderer/load_ply_raw/ssim_t/roi_box 等（文件名带数字前缀）。"""
    global _RE14
    if _RE14 is None:
        spec = importlib.util.spec_from_file_location(
            "face_local_opt", str(Path(__file__).parent / "14_face_local_opt.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _RE14 = m
    return _RE14


def load_target_cams(scene):
    """target 场景 prior_camera.json → (c2w list, K list, ids)。与 07 约定一致：OpenCV c2w。"""
    d = json.loads((scene / "prior_camera.json").read_text())
    extr = {e["camera_id"]: np.asarray(e["matrix"], np.float64) for e in d["extrinsics"]}
    intr = {e["camera_id"]: np.asarray(e["matrix"], np.float64) for e in d["intrinsics"]}
    ids = [e["camera_id"] for e in d["extrinsics"] if e["camera_id"] in intr
           and (scene / "images" / f"{e['camera_id']}.png").exists()]
    if not ids:
        raise SystemExit(f"❌ {scene} 相机/图片对不上")
    return [extr[i] for i in ids], [intr[i] for i in ids], ids


def estimate_sim3(res, source_scene):
    """用 source 相机对（场景 prior ↔ 结果 cameras.json）估计模型系 Sim(3)。

    返回 (s, R, T, 残差list)。c2w_model[i] = [sR|T] @ c2w_prior[i]，
    R = R_res[0] @ inv(R_prior[0])（res[0]≈I 时即 inv(R_prior[0])），
    s = res/prior 相机中心间距比中位数，T 由第 0 对解出。
    """
    d_prior = json.loads((source_scene / "prior_camera.json").read_text())
    d_res = json.loads((res / "cameras.json").read_text())
    P = {e["camera_id"]: np.asarray(e["matrix"], np.float64) for e in d_prior["extrinsics"]}
    Q = {e["camera_id"]: np.asarray(e["matrix"], np.float64) for e in d_res["extrinsics"]}
    ids = [k for k in P if k in Q]
    if len(ids) < 2:
        raise SystemExit("❌ source 相机对不足，无法估计 Sim(3)")
    k0 = ids[0]
    R = Q[k0][:3, :3] @ np.linalg.inv(P[k0][:3, :3])
    ortho = float(np.abs(R @ R.T - np.eye(3)).max())
    # 尺度：相机中心间距比中位数
    ratios = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dp = np.linalg.norm(P[ids[i]][:3, 3] - P[ids[j]][:3, 3])
            dq = np.linalg.norm(Q[ids[i]][:3, 3] - Q[ids[j]][:3, 3])
            if dp > 1e-9:
                ratios.append(dq / dp)
    s = float(np.median(ratios))
    T = Q[k0][:3, 3] - s * R @ P[k0][:3, 3]
    resid = []
    for k in ids:
        dt = np.abs(s * R @ P[k][:3, 3] + T - Q[k][:3, 3]).max()
        dR = np.abs(R @ P[k][:3, :3] - Q[k][:3, :3]).max()
        resid.append(dict(id=k, max_dt=float(dt), max_dR=float(dR)))
    if ortho > 1e-4 or max(r["max_dt"] for r in resid) > 1e-3:
        raise SystemExit(f"❌ Sim(3) 拟合不过关: ortho={ortho:.2e} resid={resid}")
    return s, R, T, dict(ortho=ortho, resid=resid, s_per_pair=ratios)


def apply_sim3(c2w, s, R, T):
    out = np.eye(4)
    out[:3, :3] = R @ c2w[:3, :3]
    out[:3, 3] = s * R @ c2w[:3, 3] + T
    return out


def ssim_np(m14, a, b, dev):
    """a,b: [H,W,3] float 0..1 → SSIM。"""
    t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].to(dev)  # noqa: E731
    return float(m14.ssim_t(t(a), t(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="02 输出目录（face_opt 所在）")
    ap.add_argument("--source-scene", default=None,
                    help="source 场景目录（含 prior_camera.json，Sim(3) 估计用）；"
                         "默认 ~/output/splatanything_results/_scenes/<result名>")
    ap.add_argument("--target", required=True, help="target 场景目录（07 产物，含 images/+prior_camera.json）")
    ap.add_argument("--masks", required=True, help="target mask 目录（12 --images 产物）")
    ap.add_argument("--iters", type=int, default=300, help="回退 PLY 的步数后缀")
    ap.add_argument("--arms", default="p0,p1,p2,p3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lpips", action="store_true", help="启用 face ROI LPIPS（需 lpips 包）")
    args = ap.parse_args()

    dev = "cuda"
    m14 = re14()
    res = Path(args.result)
    scene = Path(args.target)
    masks_dir = Path(args.masks)
    out_dir = Path(args.out) if args.out else res / "face_opt" / "target_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",")]

    # ---------- target 相机与 GT/mask ----------
    source_scene = Path(args.source_scene) if args.source_scene else \
        Path("~/output/splatanything_results/_scenes") .expanduser() / res.name
    s, R_sim3, T_sim3, sim3_info = estimate_sim3(res, source_scene)
    print(f"🔧 Sim(3): s={s:.6f} 残差 Δt={max(r['max_dt'] for r in sim3_info['resid']):.2e} "
          f"ortho={sim3_info['ortho']:.2e}")
    c2w_prior, K, ids = load_target_cams(scene)
    c2w = [apply_sim3(c, s, R_sim3, T_sim3) for c in c2w_prior]
    gts, face_ms, person_ms = [], [], []
    for cid in ids:
        img_path = scene / "images" / f"{cid}.png"
        gt = np.asarray(Image.open(img_path).convert("RGB"), np.float32) / 255.0
        face = np.asarray(Image.open(masks_dir / f"{cid}_face.png")) > 127
        person = np.asarray(Image.open(masks_dir / f"{cid}_person.png")) > 127
        if face.shape != gt.shape[:2] or person.shape != gt.shape[:2]:
            raise SystemExit(f"❌ mask 与图分辨率不一致: {cid}")
        gts.append(gt)
        face_ms.append(face)
        person_ms.append(person)
    w, h = gts[0].shape[1], gts[0].shape[0]

    # 背景灰度：与 14 同口径（GT 四边中位数）
    bg = float(np.median(np.concatenate(
        [np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]], 0) for g in gts], 0)))

    # PLY 路径：P0 = 原生 gaussians.ply（回退即恒等），其余 = 各臂导出
    ply_of = {}
    for a in arms:
        if a == "p0":
            ply_of[a] = res / "gaussians.ply"
        else:
            p = res / "face_opt" / a / f"gaussians_{a}_{args.iters}.ply"
            if not p.is_file():
                raise SystemExit(f"❌ 缺回退 PLY: {p}")
            ply_of[a] = p

    # LPIPS（可选，推进门用 target 脸 LPIPS 相对 P0 变化）
    lpips_fn = None
    if args.lpips:
        try:
            import lpips as lpips_mod
            lpips_fn = lpips_mod.LPIPS(net="alex", verbose=False).to(dev)
        except Exception as e:
            print(f"⚠️ lpips 不可用，跳过 LPIPS: {type(e).__name__}: {e}")

    # ---------- 逐臂渲染 + 评分 ----------
    empty_sel = np.array([], dtype=np.int64)
    roi_boxes = [m14.roi_box(f, 2.3, w, h) for f in face_ms]
    rows = []
    renders = {a: [] for a in arms}
    for a in arms:
        g = m14.load_ply_raw(ply_of[a])
        R = m14.Renderer(g, np.stack(c2w), np.stack(K), w, h, empty_sel, bg, device=dev)
        t0 = time.time()
        with torch.no_grad():
            rd, _ = R.render({}, list(range(len(ids))))
        rd_np = rd.cpu().numpy()
        renders[a] = rd_np
        for i, cid in enumerate(ids):
            gt, r9, fm = gts[i], rd_np[i], face_ms[i]
            x0, y0, x1, y1 = roi_boxes[i]
            fdi = cv2.dilate(fm.astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
                             ).astype(bool)
            person_only = person_ms[i] & ~fdi
            sh_gt = m14.sharp_np(gt, fm)
            sh_rd = m14.sharp_np(r9, fm)
            row = dict(arm=a, target=cid,
                       ply_sha256_16=hashlib.sha256(
                           ply_of[a].read_bytes()).hexdigest()[:16] if a == "p0" else None,
                       psnr_full=m14.psnr_np(gt, r9),
                       ssim_full=ssim_np(m14, gt, r9, dev),
                       psnr_face=m14.psnr_np(gt[y0:y1, x0:x1], r9[y0:y1, x0:x1]),
                       ssim_face=ssim_np(m14, gt[y0:y1, x0:x1], r9[y0:y1, x0:x1], dev),
                       sharp_ratio=(sh_rd / sh_gt) if sh_gt > 0 else None,
                       psnr_person=(m14.psnr_np(gt[person_only], r9[person_only])
                                    if person_only.sum() > 256 else None))
            if lpips_fn is not None:
                crop_gt = torch.from_numpy(
                    gt[y0:y1, x0:x1]).permute(2, 0, 1)[None].to(dev) * 2 - 1
                crop_rd = torch.from_numpy(
                    r9[y0:y1, x0:x1]).permute(2, 0, 1)[None].to(dev) * 2 - 1
                with torch.no_grad():
                    row["lpips_face"] = float(lpips_fn(crop_rd, crop_gt))
            rows.append(row)
        del R
        torch.cuda.empty_cache()
        print(f"✅ {a}: render+eval {len(ids)} target × {time.time()-t0:.1f}s")

    # ---------- 汇总：Δ vs P0（逐 target） ----------
    by = {}
    for r in rows:
        by.setdefault(r["target"], {})[r["arm"]] = r
    summary = []
    keys = ["psnr_full", "ssim_full", "psnr_face", "ssim_face", "sharp_ratio",
            "psnr_person"] + (["lpips_face"] if lpips_fn is not None else [])
    for cid, d in by.items():
        line = dict(target=cid)
        for a in arms:
            for k in keys:
                line[f"{k}.{a}"] = d[a][k]
        for k in keys:
            base = d["p0"][k]
            for a in arms:
                if a != "p0" and d[a][k] is not None and base is not None:
                    line[f"d_{k}.{a}"] = d[a][k] - base
        summary.append(line)

    with open(out_dir / "target_metrics.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(args=vars(args), bg=bg, targets=ids,
                       sim3=dict(s=s, R=R_sim3.tolist(), T=T_sim3.tolist(),
                                 **sim3_info),
                       pairwise_note="target 与 source 帧同属 82 帧环绕序列，人物姿态/曝光差异单列",
                       rows=rows, summary_vs_p0=summary),
                  f, indent=2, ensure_ascii=False)

    # ---------- 目检裁图：每 target 一行的 ROI 条 ----------
    for i, cid in enumerate(ids):
        x0, y0, x1, y1 = roi_boxes[i]
        tiles = [np.ascontiguousarray((gts[i][y0:y1, x0:x1] * 255).astype(np.uint8))]
        for a in arms:
            tiles.append(np.ascontiguousarray(
                (renders[a][i][y0:y1, x0:x1] * 255).astype(np.uint8)))
        strip = np.concatenate(tiles, 1)
        Image.fromarray(strip).save(out_dir / f"target_{cid}_roi_GT_{'_'.join(arms)}.png")

    # ---------- 打印 ----------
    print("\n📊 target 评估汇总（Δ = arm − P0）")
    hdr = f"{'target':<22}" + "".join(f"{a:>16}" for a in arms)
    print(hdr + "   (face PSNR / face LPIPS / full PSNR / person PSNR)")
    for cid, d in by.items():
        cells = []
        for a in arms:
            lp = (f"{d[a]['lpips_face']:.4f}" if d[a].get("lpips_face") is not None
                  else "n/a")
            cells.append(f"{d[a]['psnr_face']:6.2f}/{lp}/{d[a]['psnr_full']:6.2f}/"
                         f"{d[a]['psnr_person'] if d[a]['psnr_person'] else float('nan'):6.2f}")
        print(f"{cid:<22}" + "".join(f"{c:>16}" for c in cells))
    print(f"\n🎉 Done: {out_dir}")


if __name__ == "__main__":
    main()
