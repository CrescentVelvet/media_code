#!/usr/bin/env python3
"""20_f3_depth_audit.py — F3 网络侧：教师深度/融合后脸层 vs GT 深度审计（rosemary33v）。

主计划 F3 原文：「先用同一静态人物数据明确教师原始深度/融合前各脸层与 GT 的误差；
若确认中心深度错误，冻结教师深度的学生 LoRA 不应承担全部修复目标……；若深度正确
但属性/merge 引入问题，优先学生/属性。」

本脚本纯 CPU 回读（gsplat 渲染 expected depth 用 GPU），不训练、不改 12~15。

口径：
  GT 深度      human_fbx_gt/rosemary_face_v3/depth/{id:06d}.exr，通道 "depth.V"（1024²，
               相机系线性 z 米，背景=1e10）；resize 到 756² 与 SA 对齐，valid=d<1e8。
  教师深度     02 推理导出的 depth/{idx}_{id}.npy（756²，pose 模式下为度量深度）。
  融合渲染深度 gaussians.ply + cameras.json 渲 RGB+ED 第 4 通道（模型系直用）。
  脸层高斯     face_opt/select/selected_ids.npy 投影进视角，z_g 与 GT/融合深度比较，
               按相对融合期望深度分 front/at/behind 三层（<0.9 / ≤1.1 / >1.1）。

判读指标：
  ①教师深度在脸区 bias（符号系统偏移？）+ 逐视角尺度比 s=median(gt/t) 的稳定性
    （尺度错 vs 形状错分离：校正 s 后残差）；
  ②融合渲染深度脸区误差（merge/footprint 是否引入额外误差）；
  ③脸簇高斯中心深度 vs GT（几何层是否放对位置、哪一层错）。

用法（splatanything env，WSL）：
  python 20_f3_depth_audit.py [--result ~/output/splatanything_results/pose_rosemary/rosemary33v] \
      [--gt-pkg /mnt/d/dataset/human_fbx_gt/rosemary_face_v3] \
      [--target-scene ~/output/splatanything_results/_scenes/rosemary3_t2]
产物：face_opt/depth_audit/f3_metrics.json + 每视角 *_teacher_err.png / *_fused_err.png / *_layers.png
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _load_mod(name, fname):
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).parent / fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M03 = _load_mod("render_eval03", "03_render_eval.py")
M13 = _load_mod("face_select13", "13_face_select.py")


def load_gt_depth(pkg, view_id, size):
    """读 GT EXR（通道 depth.V）→ resize 到 size，返回 (z, valid)。"""
    import OpenEXR
    f = Path(pkg) / "depth" / f"{int(view_id):06d}.exr"
    e = OpenEXR.File(str(f))
    z = e.channels()["depth.V"].pixels.astype(np.float32)
    valid = z < 1e8
    zi = Image.fromarray(np.where(valid, z, 0.0))
    zi = zi.resize(size, Image.BILINEAR)
    vi = Image.fromarray((valid * 255).astype(np.uint8))
    vi = vi.resize(size, Image.NEAREST)
    return np.asarray(zi, np.float32), np.asarray(vi, np.uint8) > 127


def load_mask(path, size):
    im = Image.open(path).convert("L").resize(size, Image.NEAREST)
    return np.asarray(im, np.uint8) > 127


def stats(err, mask):
    e = err[mask]
    if e.size == 0:
        return None
    return {
        "n": int(e.size),
        "mae_mm": round(float(np.abs(e).mean()) * 1000, 2),
        "rmse_mm": round(float(np.sqrt((e ** 2).mean())) * 1000, 2),
        "bias_mm": round(float(e.mean()) * 1000, 2),
        "p90_abs_mm": round(float(np.percentile(np.abs(e), 90)) * 1000, 2),
    }


def err_overlay(rgb, err, base_mask, vmax_mm=60.0):
    """误差热力叠加：蓝=近 0，红=|err|>=vmax；只画 base_mask 内。"""
    img = np.asarray(rgb if isinstance(rgb, Image.Image)
                     else Image.open(rgb).convert("RGB")).copy().astype(np.float32)
    a = np.clip(np.abs(err) / (vmax_mm / 1000.0), 0, 1)
    a[~base_mask] = 0.0
    heat = np.zeros_like(img)
    heat[..., 0] = 255 * a
    heat[..., 2] = 255 * (1 - a)
    out = (0.55 * img + 0.45 * heat).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--gt-pkg", required=True)
    ap.add_argument("--target-scene", required=True)
    ap.add_argument("--source-scene", default=None,
                    help="source 场景（含 prior_camera.json 的 pkg 相机）；"
                         "默认 = --result 同名场景（_scenes/<result.name>）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    res = Path(args.result)
    outdir = res / "face_opt" / "depth_audit"
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 相机：source 来自 cameras.json，target 来自 t2 场景 prior_camera.json ----
    c2w_all, K_all, ids_all = M03.load_cameras(str(res / "cameras.json"))
    tcam = json.load(open(Path(args.target_scene) / "prior_camera.json"))
    c2w_t = np.array([np.array(c["matrix"], np.float64) for c in tcam["extrinsics"]])
    ids_t = [c["camera_id"] for c in tcam["extrinsics"]]
    K_t = K_all[0]  # 同一内参
    c2w_all = list(c2w_all) + list(c2w_t)
    ids_all = list(ids_all) + ids_t
    K_all = list(K_all) + [K_t] * len(ids_t)

    g = M03.load_ply(str(res / "gaussians.ply"))
    sel = np.load(res / "face_opt" / "select" / "selected_ids.npy")

    # ---- 渲染相机：15 已判决的唯一正确链 = S @ pkg c2w（source 与 target 一致）----
    # 决定性实测（_tmp_f3_decisive）：S@pkg 脸深 bias +6.3mm；cameras.json +441mm；
    # raw pkg −623mm。即 PLY 对应相机是 S@pkg，cameras.json 的 c2w 并不对应该 PLY。
    # A_v = c2w_res·inv(c2w_pkg) 虽每视角精确，但只复现 cameras.json 那条错链，弃用。
    src_scene = Path(args.source_scene) if args.source_scene else \
        (Path(args.target_scene).parent / res.name)
    prior = json.load(open(src_scene / "prior_camera.json"))
    pkg_by_id = {c["camera_id"]: np.array(c["matrix"], np.float64)
                 for c in prior["extrinsics"]}
    _s3 = json.load(open(res / "face_opt" / "target_eval" / "target_metrics.json"))["sim3"]
    S = np.eye(4)
    S[:3, :3] = np.array(_s3["R"], np.float64) * float(_s3["s"])
    S[:3, 3] = np.array(_s3["T"], np.float64)
    c2w_all = [S @ pkg_by_id[cid] if cid in pkg_by_id else S @ np.asarray(c, np.float64)
               for c, cid in zip(c2w_all, ids_all)]
    print(f"🔧 渲染相机 = S@pkg c2w（s={_s3['s']:.4f}，source/target 统一）")

    metrics = {"views": {}, "device": args.device,
               "frame": {"camera": "S@pkg c2w", "s": round(float(_s3["s"]), 4)}}
    H = W = 756

    for vi, (c2w, K, cid) in enumerate(zip(c2w_all, K_all, ids_all)):
        view_id = int(cid.split("_")[-1])
        is_src = vi < len(ids_all) - len(ids_t)
        sub = "masks" if is_src else "masks_target"
        face_m = load_mask(res / "face_opt" / sub / f"{cid}_face.png", (W, H))
        person_m = load_mask(res / "face_opt" / sub / f"{cid}_person.png", (W, H))
        z_gt, valid = load_gt_depth(args.gt_pkg, view_id, (W, H))
        # GT EXR 与渲染输出同在 pkg 相机帧（S@pkg 链实测 bias +6mm），无需 warp。

        # ---- 融合渲染 expected depth（全高斯；c2w 已统一为 S@pkg）----
        z_exp, _ = M13.render_expected_depth(
            g, np.asarray(c2w), np.asarray(K), W, H, device=args.device)
        z_exp = np.asarray(z_exp).reshape(H, W)  # render 返回带 batch 维

        vm = {
            "view": cid, "is_source": is_src,
            "gt_valid_px": int(valid.sum()),
            "fused_depth": {
                "face": stats(z_exp - z_gt, face_m & valid),
                "person": stats(z_exp - z_gt, person_m & valid & ~face_m),
                "full": stats(z_exp - z_gt, valid),
            },
        }

        def view_rgb():
            p = res / "rgb" / f"{cid}.png"
            if not p.exists():  # target：SA 只导出 source rgb，从 t2 场景读 GT 帧
                p = Path(args.target_scene) / "images" / f"{cid}.png"
            im = Image.open(p).convert("RGB")
            return im.resize((W, H), Image.BILINEAR) if im.size != (W, H) else im

        err_overlay(view_rgb(), (z_exp - z_gt) * 1000,
                    person_m & valid, vmax_mm=20.0).save(
            outdir / f"{cid}_fused_err.png")

        # ---- 教师深度（仅 source 有导出）----
        tpath = res / "depth" / f"{cid}.npy"
        if tpath.exists():
            t = np.load(tpath).astype(np.float32)
            tvalid = valid & np.isfinite(t) & (t > 1e-3)
            s = float(np.median(z_gt[tvalid] / t[tvalid]))
            vm["teacher_depth"] = {
                "scale_median_ratio": round(s, 4),
                "raw": {
                    "face": stats(t - z_gt, face_m & tvalid),
                    "person": stats(t - z_gt, person_m & tvalid & ~face_m),
                    "full": stats(t - z_gt, tvalid),
                },
                "scale_corrected": {  # 分离尺度错 vs 形状错
                    "face": stats(t * s - z_gt, face_m & tvalid),
                    "full": stats(t * s - z_gt, tvalid),
                },
            }
            # 教师误差图用尺度校正后误差（raw 是 ~6.3% 恒定偏移，会整体饱和误导）；
            # vmax 收紧到 20mm 以显示脸区形状残差。
            err_overlay(view_rgb(), (t * s - z_gt) * 1000,
                        person_m & tvalid, vmax_mm=20.0).save(
                outdir / f"{cid}_teacher_err.png")

        # ---- 脸簇高斯层 vs GT ----
        u, v, z = M13.project_points(g["means"], np.asarray(c2w), np.asarray(K))
        in_img = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        sel_mask = np.zeros(len(z), bool)
        sel_mask[np.asarray(sel)] = True
        # 脸区内逐高斯（全体高斯，不只选中）分箱
        ui, vi2 = np.clip(u, 0, W - 1).astype(int), np.clip(v, 0, H - 1).astype(int)
        px_face = face_m[vi2, ui] & in_img & valid[vi2, ui]
        rows = []
        for name, msk in (("all", px_face), ("selected", px_face & sel_mask)):
            zg = z[msk]
            zx = z_exp[vi2[msk], ui[msk]]
            zg_gt = z_gt[vi2[msk], ui[msk]]
            for binname, b in (
                ("front", zg < 0.9 * zx), ("at", (zg >= 0.9 * zx) & (zg <= 1.1 * zx)),
                ("behind", zg > 1.1 * zx),
            ):
                if b.sum() == 0:
                    rows.append({"group": name, "bin": binname, "n": 0})
                    continue
                e = zg[b] - zg_gt[b]
                rows.append({
                    "group": name, "bin": binname, "n": int(b.sum()),
                    "gauss_z_m": round(float(np.median(zg[b])), 4),
                    "gt_z_m": round(float(np.median(zg_gt[b])), 4),
                    "bias_mm": round(float(e.mean()) * 1000, 2),
                    "mae_mm": round(float(np.abs(e).mean()) * 1000, 2),
                })
        vm["face_layers"] = rows

        # 分层叠加图（脸 ROI）
        rgb = np.asarray(view_rgb()).copy()
        ys, xs = np.where(face_m)
        if len(xs):
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            pad = 30
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
            crop = rgb[y0:y1, x0:x1]
            # selected 高斯点按 bias 着色：绿=0±5mm，红=GT 前(负 bias 大)，蓝=GT 后
            msk = px_face & sel_mask
            for zz, uu, vv in zip(z[msk], ui[msk], vi2[msk]):
                e_mm = (zz - z_gt[vv, uu]) * 1000
                col = (0, 220, 0) if abs(e_mm) < 5 else (
                    (255, 60, 60) if e_mm < -5 else (60, 120, 255))
                cv, cu = vv - y0, uu - x0
                if 0 <= cv < crop.shape[0] and 0 <= cu < crop.shape[1]:
                    crop[max(0, cv - 1):cv + 2, max(0, cu - 1):cu + 2] = col
            Image.fromarray(crop).resize(
                ((x1 - x0) * 2, (y1 - y0) * 2), Image.NEAREST).save(
                outdir / f"{cid}_layers.png")
        metrics["views"][cid] = vm
        print(f"[{cid}] fused face bias "
              f"{vm['fused_depth']['face']['bias_mm'] if vm['fused_depth']['face'] else 'n/a'}mm"
              + (f" | teacher s={vm['teacher_depth']['scale_median_ratio']}"
                 if "teacher_depth" in vm else ""), flush=True)

    # ---- 跨视角汇总 ----
    srcs = [m for c, m in metrics["views"].items() if m["is_source"]]
    scales = [m["teacher_depth"]["scale_median_ratio"] for m in srcs
              if "teacher_depth" in m]
    fbiases = [m["fused_depth"]["face"]["bias_mm"] for m in metrics["views"].values()
               if m["fused_depth"]["face"]]
    sel_at = [r for m in metrics["views"].values() for r in m["face_layers"]
              if r["group"] == "selected" and r["bin"] == "at" and r["n"] > 0]
    metrics["summary"] = {
        "teacher_scale_ratio": {
            "per_view": scales,
            "mean": round(float(np.mean(scales)), 4) if scales else None,
            "spread_maxmin": round(float(np.max(scales) - np.min(scales)), 4)
            if scales else None,
        },
        "fused_face_bias_mm": {
            "per_view": fbiases,
            "mean": round(float(np.mean(fbiases)), 2) if fbiases else None,
        },
        "selected_at_layer": {
            "median_bias_mm": round(float(np.median([r["bias_mm"] for r in sel_at])), 2)
            if sel_at else None,
            "views": len(sel_at),
        },
    }
    with open(outdir / "f3_metrics.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics["summary"], ensure_ascii=False, indent=1))
    print(f"🎉 F3 audit done -> {outdir}")


if __name__ == "__main__":
    main()
