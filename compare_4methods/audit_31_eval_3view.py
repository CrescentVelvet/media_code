#!/usr/bin/env python3
"""三图输入（N=3）五方法评测：统一渲染 32 个 GT 相机位姿，对照 32v 轮的 GT 图。

口径（对齐 32v 轮，FINDINGS §2 / audit_24）
--------------------------------------------
· 渲染相机 = 32v GT cameras.npz 的全部 32 个位姿（w2c34，OpenCV 约定）；
  内参统一取 GT 的，等比缩放到 GT 图分辨率（视场一致，跨方法逐像素可比）。
· 对照 ref = ~/output/da3_results/_captures/佳佳抱枕/images（32 张，与 32v 轮同一套）。
· 跨方法对齐 = 各方法自己的 3v 预测相机 Umeyama 到 GT 32 相机中
  「与 3 个输入帧（帧 0/40/81）GT 位姿最接近的 3 个」（audit_24 同款判据：
  相机中心残差/半径 < 3%、视线夹角 < 5°），PLY 用 s/R/t 搬到 GT 系后渲染。
  三对点解 7 DoF 相似变换是下限配置，判据不过会直接拒渲（宁缺毋滥）。

分层报告
--------
· all32   —— 32 个 GT 相机全平均（跨方法主指标，相机集与 32v 轮相同）
· input3  —— 最靠近 3 个输入帧的 3 个 GT 相机（近似自洽；三图输入帧 [0,40,81] 里
  帧 40 不在 32v 抽帧集合，只能用「最近邻」近似，见 FINDINGS §7 抽帧口径）
· holdout —— 其余 29 个（真外推）

用法（WSL，querysplat env：cv2/skimage/gsplat 齐全，经 94_render.sh 同款环境包装）：
  python compare_4methods/audit_31_eval_3view.py
  OUT=~/output/batch_4methods/eval_3view_jiajia.json ONLY=querysplat python ...
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_24_five_way_video as a24  # read_ply / read_cams / umeyama / warp

O = os.path.expanduser("~/output")
# 注意 .sh 用 OUT="${OUT:-}" 传参，空串要回落到默认值（get 不挡空串）
GT_CAMS = os.environ.get("GT_CAMS") or O + "/da3_results/_captures/佳佳抱枕/cameras.npz"
GT_IMAGES = os.environ.get("GT_IMAGES") or O + "/da3_results/_captures/佳佳抱枕/images"
BASE3 = os.environ.get("BASE3") or O + "/infinisplat_v2_results/_captures/佳佳抱枕_3v/cameras.npz"
OUT = os.environ.get("OUT") or O + "/batch_4methods/eval_3view_jiajia.json"
ONLY = os.environ.get("ONLY", "").strip()

CASES = [
    dict(name="DA3-GS",
         ply=O + "/da3_gs_results/佳佳抱枕_3v/gs_ply_full/0000.ply",
         cams=("w2c34", O + "/da3_gs_results/佳佳抱枕_3v_ff/cameras.npz")),
    dict(name="QuerySplat",
         ply=O + "/querysplat_results/佳佳抱枕_3v/gaussians.ply",
         cams=("c2w", O + "/querysplat_results/佳佳抱枕_3v/predicted_input_cameras.npz")),
    dict(name="InfiniSplat v2",
         ply=O + "/infinisplat_v2_results/佳佳抱枕_3v/scene.ply",
         cams=("c2w", O + "/infinisplat_v2_results/佳佳抱枕_3v/cameras.npz")),
    dict(name="VGGT-Ω+3DGS",
         ply=O + "/vggt_omega_results/佳佳抱枕_3v/04_model_3dgs/point_cloud/iteration_30000/point_cloud.ply",
         cams=("json", O + "/vggt_omega_results/佳佳抱枕_3v/04_model_3dgs/cameras.json")),
    dict(name="DA3+3DGS",
         ply=O + "/da3_results/佳佳抱枕_3v/04_model_3dgs/point_cloud/iteration_30000/point_cloud.ply",
         cams=("json", O + "/da3_results/佳佳抱枕_3v/04_model_3dgs/cameras.json")),
]
if ONLY:
    CASES = [c for c in CASES if ONLY.lower() in c["name"].lower()]


def log(*a):
    print(*a, flush=True)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def sharpness(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    from gsplat import rasterization

    # ── GT 侧：32 相机（c2w，07_capture_to_npz 产物，K 已是原图分辨率）+ ref 图 ──
    z = np.load(GT_CAMS)
    c2w = z["c2w"].astype(np.float64)
    Cref, dref, upref = a24.read_cams("c2w", GT_CAMS)
    refs = sorted([p for p in Path(GT_IMAGES).iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if len(refs) != len(Cref):
        raise SystemExit(f"❌ ref 图数 {len(refs)} != 相机数 {len(Cref)}")
    img0 = cv2.imread(str(refs[0]))
    H, W = img0.shape[:2]
    K = z["intrinsics"].astype(np.float64)[0]
    # K 的分辨率 sanity：cx≈宽/2 且 fx 与宽同量级，否则说明 K 与 ref 不同分辨率
    if not (0.4 * W < K[0, 2] < 0.6 * W and 0.4 * H < K[1, 2] < 0.6 * H):
        raise SystemExit(f"❌ K 主点 ({K[0,2]:.0f},{K[1,2]:.0f}) 与 ref 尺寸 {W}×{H} 不匹配")
    W2C = np.linalg.inv(c2w)
    log(f"📷 GT: {len(Cref)} 相机（c2w）｜ ref {W}×{H}×{len(refs)} ｜ fx={K[0,0]:.0f} fy={K[1,1]:.0f}")

    # ── 输入帧 GT 位姿（基准 3 帧）→ 在 32v 集合里找最近邻 ──
    Cb, _, _ = a24.read_cams("c2w", BASE3)
    idx3 = [int(np.argmin(((Cref - c) ** 2).sum(1))) for c in Cb]
    if len(set(idx3)) != 3:
        raise SystemExit(f"❌ 3 个输入帧在 32v 集合里的最近邻去重后不足 3 个: {idx3}")
    nn_dist = [float(np.linalg.norm(Cref[i] - c)) for i, c in zip(idx3, Cb)]
    log(f"🎯 输入帧最近邻: {idx3}（中心距 {['%.3f' % d for d in nn_dist]} m）"
        f" → input3 分层 = 这些索引，其余 {len(Cref) - 3} 个 = holdout")

    dev = "cuda"
    # 预检：与另一个重度 GPU 任务（如 19_orbit 视频渲染）共存时，WSL2 vGPU 上
    # gsplat rasterization 会退化成 100% 利用率活锁、单帧数小时无进展（2026-11-10 实测）。
    # 显存余量 < 阈值直接拒跑，避免再 silent 挂一晚上。
    free_b, total_b = torch.cuda.mem_get_info()
    log(f"🩺 GPU 预检: 空闲 {free_b/2**30:.1f} / {total_b/2**30:.1f} GiB")
    if free_b < 4 * 2**30:
        raise SystemExit("❌ GPU 空闲显存 < 4 GiB——可能有其它任务在跑（或其残留），"
                         "活锁风险高，拒跑。等其退出或 wsl --shutdown 清残留后再试。")
    Kt = torch.from_numpy(K).float().to(dev)[None]
    bg = torch.ones(3, device=dev)
    refs_rgb = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in refs]
    refs_sharp = [sharpness(cv2.imread(str(p))) for p in refs]

    result = {"gt": {"cams": str(GT_CAMS), "images": str(GT_IMAGES), "size": [W, H],
                     "input3_idx": idx3, "input3_nn_dist_m": nn_dist},
              "methods": {}}
    # 每张 ref 的锐度（供「渲染锐度/ref 锐度」比例）
    result["ref_sharpness"] = refs_sharp

    for c in CASES:
        if not Path(c["ply"]).exists():
            log(f"⏭️  {c['name']}: 缺产物 {c['ply']}（对应步骤未跑完？）")
            continue
        g = a24.read_ply(c["ply"])
        n_pts = g["n"]
        Cx, dx, upx = a24.read_cams(*c["cams"])
        if len(Cx) != 3:
            log(f"⚠️  {c['name']}: 相机数 {len(Cx)} != 3（预期三图预测相机）")
        s, R, t = a24.umeyama(Cx, Cref[idx3])
        C2 = (s * (R @ Cx.T)).T + t
        rms = float(np.sqrt(((C2 - Cref[idx3]) ** 2).sum(1).mean()))
        rel = rms / float(np.median(np.linalg.norm(Cref - Cref.mean(0), axis=1)))
        d2 = (R @ dx.T).T
        ang = float(np.median(np.degrees(np.arccos(np.clip(
            np.sum(d2 * dref[idx3], 1), -1, 1)))))
        flag = "OK" if (rel < 0.03 and ang < 5) else "FAIL"
        log(f"\n🔧 {c['name']}: {g['n']:,} 点 SH{g['deg']} ｜ s={s:.4f} 残差 {rel*100:.2f}% "
            f"视线 {ang:.2f}° [{flag}]")
        if flag != "OK":
            log(f"❌ {c['name']} 对齐没过判据，跳过（宁缺毋滥）")
            result["methods"][c["name"]] = {"align": "FAIL", "rel": rel, "ang_deg": ang}
            continue
        g = a24.warp(g, s, R, t)
        G = {k: torch.from_numpy(g[k]).to(dev)
             for k in ("means", "quats", "scales", "opac", "sh")}
        deg = g["deg"]
        del g

        rows, amed = [], 0.0
        for i in range(len(Cref)):
            with torch.no_grad():
                img, alpha, _ = rasterization(
                    G["means"], G["quats"], G["scales"], G["opac"], G["sh"],
                    torch.from_numpy(W2C[i]).float().to(dev)[None], Kt,
                    W, H, sh_degree=deg, packed=False,
                    backgrounds=bg[None], render_mode="RGB")
            rgb = img[0].clamp(0, 1).mul(255).byte().cpu().numpy()   # RGB
            amed = float(alpha.median())
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rows.append(dict(view=i, psnr=psnr(rgb, refs_rgb[i]),
                             ssim=float(ssim(refs_rgb[i], rgb, channel_axis=2, data_range=255)),
                             sharp=sharpness(bgr)))
            if (i + 1) % 16 == 0:
                log(f"   frame {i+1}/{len(Cref)}  α中位 {amed:.3f}")
            elif i < 4:  # 每方法开头几帧逐帧报——活锁会在几分钟内暴露而不是挂一晚上
                log(f"   frame {i+1}/{len(Cref)}  α中位 {amed:.3f}")
        del G
        torch.cuda.empty_cache()

        def agg(idxs):
            ps = [rows[i]["psnr"] for i in idxs]
            ss = [rows[i]["ssim"] for i in idxs]
            sh = [rows[i]["sharp"] for i in idxs]
            rsh = [refs_sharp[i] for i in idxs]
            return dict(psnr=float(np.mean(ps)), ssim=float(np.mean(ss)),
                        sharp=float(np.mean(sh)),
                        sharp_ratio=float(np.mean([a / b for a, b in zip(sh, rsh)])))
        holdout_idx = [i for i in range(len(Cref)) if i not in idx3]
        result["methods"][c["name"]] = dict(
            ply=c["ply"], n_pts=n_pts,
            align="OK", scale=s, rel=rel, ang_deg=ang,
            all32=agg(range(len(Cref))),
            input3=agg(idx3), holdout=agg(holdout_idx),
            per_view=rows)
        log(f"   all32 PSNR {result['methods'][c['name']]['all32']['psnr']:.2f} dB "
            f"｜ input3 {result['methods'][c['name']]['input3']['psnr']:.2f} "
            f"｜ holdout {result['methods'][c['name']]['holdout']['psnr']:.2f} "
            f"｜ 锐度比 {result['methods'][c['name']]['all32']['sharp_ratio']:.2f}")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    log(f"\n✅ 写出 {OUT}")


if __name__ == "__main__":
    main()
