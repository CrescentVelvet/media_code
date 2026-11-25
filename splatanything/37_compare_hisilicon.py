#!/usr/bin/env python3
"""海思采集包：竞品与 SplatAnything 的同机位、同画布渲染对照。

相机由输入 OpenGL c2w 转 OpenCV。
相邻 source 的相机中心方向与距离插值，始终注视随包 bounds.center。
竞品读取原生 SH3（f_rest 按 channel-major 布局）；SA 使用导出 DC。
SA 只用 source 相机对齐 X_capture=s R X_model+T，不用竞品 PLY 或渲染调参。
竞品直接按随包相机回放出现近距离局部画面，且没有配套查看器/尺度说明；
它只作为同协议回放，不是经校准的质量排名。--sa-only 不跳过这个边界。
轨迹由相邻 source 相机球面插值，再用 bounds 的范围限制，不做无依据 360° 旋转。
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData

ROOT = Path(__file__).resolve().parent
FLIP_CV = np.diag([1., -1., -1.])
FLIP_HOM = np.diag([1., -1., -1., 1.])


def load_mod(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dump(path, obj):
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
        json.dump(convert(obj), f, ensure_ascii=False, indent=2)


def font(size, bold=False):
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    path = Path("/mnt/c/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def input_cameras(case, camera_path):
    data = json.loads(camera_path.read_text(encoding="utf-8"))
    gl = np.tile(np.eye(4), (len(data["views"]), 1, 1))
    for i, v in enumerate(data["views"]):
        gl[i, :3, :3] = np.asarray(v["rotation3x3"], np.float64).reshape(3, 3)
        gl[i, :3, 3] = np.asarray(v["position"], np.float64)
    cv = gl @ FLIP_HOM
    K = np.array([[[v["fx"], 0, v["cx"]], [0, v["fy"], v["cy"]], [0, 0, 1]]
                  for v in data["views"]], np.float32)
    return cv, K, data


def exported_cameras(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [x["camera_id"] for x in data["extrinsics"]]
    extr = {x["camera_id"]: np.asarray(x["matrix"], np.float64) for x in data["extrinsics"]}
    intr = {x["camera_id"]: np.asarray(x["matrix"], np.float32) for x in data["intrinsics"]}
    return ids, np.stack([extr[i] for i in ids]), np.stack([intr[i] for i in ids])


def fit_source_frame(gt_c2w, pred_c2w, gt_ids, pred_ids):
    """只用 source 求 R_exp=M R_gt，C_exp=s M C_gt+u，不拟合 target/竞品。"""
    index = {k: i for i, k in enumerate(gt_ids)}
    gt = np.stack([gt_c2w[index[k]] for k in pred_ids])
    transforms = pred_c2w[:, :3, :3] @ np.transpose(gt[:, :3, :3], (0, 2, 1))
    u, _, vt = np.linalg.svd(transforms.mean(0))
    M = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    rot_residual = float(np.abs(transforms - M).max())
    a = gt[:, :3, 3] @ M.T
    n = len(a)
    design = np.concatenate([a.reshape(-1, 1), np.tile(np.eye(3), (n, 1))], axis=1)
    solution = np.linalg.lstsq(design, pred_c2w[:, :3, 3].reshape(-1), rcond=None)[0]
    s, t = float(solution[0]), solution[1:]
    residuals = s * a + t - pred_c2w[:, :3, 3]
    pair_scale = [np.linalg.norm(pred_c2w[i, :3, 3] - pred_c2w[j, :3, 3]) /
                  np.linalg.norm(a[i] - a[j]) for i in range(n) for j in range(i + 1, n)
                  if np.linalg.norm(a[i] - a[j]) > 1e-6]
    report = {"R_model_from_capture": M, "scale_model_from_capture": s,
              "translation_model_from_capture": t,
              "rotation_max_abs": rot_residual, "center_max_abs": float(np.abs(residuals).max()),
              "pair_scale_median": float(np.median(pair_scale)),
              "pair_scale_spread": float(np.ptp(pair_scale)), "source_only": True}
    if not (s > 0 and rot_residual < 2e-4 and np.abs(residuals).max() < 2e-3 and
            abs(s - np.median(pair_scale)) < 1e-3):
        raise RuntimeError(f"Source camera frame gate failed: {report}")
    return report


def source_camera_probe(sa, model_cv, model_K, common_model_cv, K, size, root):
    """兼容单场景诊断：保留原生回放和采集画布结果，不将它冒充 NVS 分数。"""
    native_full, _ = render(sa, model_cv, model_K, 756, 756)
    native_input = np.stack([np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
                             for p in sorted((root / "rgb").glob("*.png"))])
    native_common, native_common_alpha = render(sa, model_cv, model_K, size, size)
    common_rgb, common_alpha = render(sa, common_model_cv, K, size, size)
    native_input_mae = [float(np.abs(native_full[i] - native_input[i]).mean())
                        for i in range(len(native_full))]
    return native_common, native_common_alpha, common_rgb, common_alpha, native_input_mae


def model_camera(capture_c2w, frame):
    M, s, t = (np.asarray(frame[k]) for k in
               ("R_model_from_capture", "scale_model_from_capture", "translation_model_from_capture"))
    out = np.tile(np.eye(4), (len(capture_c2w), 1, 1))
    out[:, :3, :3] = M @ capture_c2w[:, :3, :3]
    out[:, :3, 3] = s * (M @ capture_c2w[:, :3, 3].T).T + t
    return out


def load_gaussians(path, quaternion="passthrough"):
    import scipy.special
    p = PlyData.read(str(path))["vertex"]
    names = p.data.dtype.names
    n_rest = sum(n.startswith("f_rest_") for n in names)
    degree = int(math.sqrt(1 + n_rest // 3)) - 1
    if n_rest % 3 or (degree + 1) ** 2 != 1 + n_rest // 3:
        raise RuntimeError("Invalid SH layout")
    sh = np.empty((len(p), 1 + n_rest // 3, 3), np.float32)
    sh[:, 0] = np.stack([p[f"f_dc_{i}"] for i in range(3)], axis=1)
    if n_rest:
        sh[:, 1:] = np.stack([p[f"f_rest_{i}"] for i in range(n_rest)], axis=1).reshape(
            len(p), 3, n_rest // 3).transpose(0, 2, 1)
    quat = np.stack([p[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    if quaternion in ("wxyz", "passthrough"):
        pass
    elif quaternion == "xyzw":
        quat = quat[:, [3, 0, 1, 2]].copy()
    else:
        raise ValueError(quaternion)
    return {"means": np.stack([p[k] for k in ("x", "y", "z")], 1).astype(np.float32),
            "scales": np.exp(np.stack([p[f"scale_{i}"] for i in range(3)], 1)).astype(np.float32),
            "opacities": scipy.special.expit(p["opacity"].astype(np.float32)).astype(np.float32),
            "quats": quat, "sh": sh, "degree": degree, "point_count": len(p)}


def render(g, c2w, K, width, height, background=(.96, .96, .96), batch=1):
    """同一 gsplat 光栅器，直接传原生 SH，不自行重算 +0.5。"""
    import torch
    from gsplat import rasterization
    out_all, alpha_all = [], []
    device = "cuda"
    keys = ("means", "scales", "opacities", "quats", "sh")
    gpu = {k: torch.from_numpy(g[k]).to(device) for k in keys}
    for start in range(0, len(c2w), batch):
        n = min(batch, len(c2w) - start)
        mats = torch.from_numpy(np.linalg.inv(c2w[start:start + n]).astype(np.float32)).to(device)
        intr = torch.from_numpy(K[start:start + n].astype(np.float32)).to(device)
        bg = torch.tensor(np.tile(np.asarray(background, np.float32), (n, 1)), device=device)
        rgb, alpha, _ = rasterization(
            means=gpu["means"], quats=gpu["quats"], scales=gpu["scales"],
            opacities=gpu["opacities"], colors=gpu["sh"][None].expand(n, -1, -1, -1).contiguous(),
            viewmats=mats, Ks=intr, width=width, height=height,
            sh_degree=int(g["degree"]), eps2d=.01, near_plane=.001, far_plane=100.,
            backgrounds=bg, render_mode="RGB")
        out_all.append(rgb.clamp(0, 1).float().cpu().numpy())
        alpha_all.append(alpha.float().cpu().numpy())
    return np.concatenate(out_all), np.concatenate(alpha_all)


def look_at_gl(eye, target):
    z = np.asarray(eye, np.float64) - np.asarray(target, np.float64)  # OpenGL +Z 朝相机后方
    z /= max(np.linalg.norm(z), 1e-12)
    x = np.cross(np.array([0., 1., 0.]), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1., 0., 0.])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = np.stack([x, y, z], axis=1), eye
    return out


def orbit(bounds, count, source_gl):
    """沿相邻 source 的球面位置/注视方向插值，不用另一坐标约定的 initYaw 造轨。"""
    source_gl = np.asarray(source_gl, np.float64)
    centers = source_gl[:, :3, 3]
    target = np.array([bounds["centerX"], bounds["centerY"], bounds["centerZ"]], np.float64)
    forward = -source_gl[:, :3, 2]
    rel = centers - target
    distance = np.linalg.norm(rel, axis=1)
    unit = rel / distance[:, None]
    segments = np.linspace(0, len(source_gl) - 1, count)
    lower = np.floor(segments).astype(int)
    upper = np.minimum(lower + 1, len(source_gl) - 1)
    fraction = (segments - lower)[:, None]
    # 十台相机之间的平滑插值；非整圈的回看段不虚构未观测的后侧。
    direction = unit[lower] * (1 - fraction) + unit[upper] * fraction
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radii = distance[lower] * (1 - fraction[:, 0]) + distance[upper] * fraction[:, 0]
    radii = np.clip(radii, bounds["minDistance"], bounds["maxDistance"])
    eyes = target + direction * radii[:, None]
    gl = np.stack([look_at_gl(eye, target) for eye in eyes])
    yaw = np.degrees(np.arctan2(direction[:, 0], direction[:, 2]))
    pitch = np.degrees(np.arcsin(np.clip(direction[:, 1], -1, 1)))
    return gl @ FLIP_HOM, {"yaw_deg": yaw, "pitch_deg": pitch, "distance": radii,
                          "target": target, "source_first_camera": source_gl[0],
                          "source_interpolation_fraction": fraction[:, 0],
                          "source_pair_lower": lower, "source_pair_upper": upper,
                          "rule": "linear interpolation of unit camera-center directions and distance between adjacent supplied source cameras; look-at supplied bounds center"}


def overlay(rgb, title, subtitle, size):
    canvas = Image.new("RGB", (size, size + 72), (18, 25, 34))
    canvas.paste(Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)).resize((size, size)), (0, 72))
    d = ImageDraw.Draw(canvas)
    d.text((18, 10), title, fill=(242, 247, 251), font=font(28, True))
    d.text((18, 45), subtitle, fill=(148, 174, 194), font=font(16))
    return np.asarray(canvas)


def save_video(frames, path, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    temp = path.with_name(path.stem + "_encoding.avi")
    vw = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"Cannot create video: {temp}")
    for frame in frames:
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vw.release()
    encoder = os.environ.get("FFMPEG", "").strip()
    if encoder and Path(encoder).is_file():
        subprocess.run([encoder, "-hide_banner", "-loglevel", "error", "-y", "-i", str(temp),
                        "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(path)], check=True)
        temp.unlink()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(frames):
            raise RuntimeError(f"H264 frame-count validation failed: {path}")
        cap.release()
        print(f"✅ H.264 video: {path}")
    else:
        # 无 ffmpeg 时仍生成完整可读视频，但明确不伪称 H.264。
        fallback = path.with_name(path.stem + "_MJPG.avi")
        if fallback != temp:
            temp.replace(fallback)
        print(f"⚠️  FFMPEG 未设置，保留 Motion JPEG AVI: {fallback}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--size", type=int, default=704)
    ap.add_argument("--probe-only", action="store_true", help="只渲首源及轨迹起点，检查方向")
    ap.add_argument("--sa-only", action="store_true", help="仅输出 SA；仍需保留竞品回放失配的边界说明")
    ap.add_argument("--skip-existing-video", action="store_true", help="已有 H264 视频时不重渲")
    ap.add_argument("--competitor-quat", choices=["passthrough", "wxyz", "xyzw"], default="passthrough")
    args = ap.parse_args()
    case = args.case.resolve()
    root = case / "splatanything"
    result = root / "inference" / "splatanything"
    camera_path = next(case.glob("*cameras.json"))
    bounds_path = next(case.glob("*bounds.json"))
    competitor_path = next(p for p in case.glob("*.ply") if p.name != "splatanything.ply")
    out = root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    capture_cv, K, metadata = input_cameras(case, camera_path)
    bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
    competitor = load_gaussians(competitor_path, args.competitor_quat)
    reference_files = sorted((root / "images").glob("*.png"))
    first = Image.open(reference_files[0]).convert("RGB")
    gt = np.asarray(first.resize((args.size, args.size)), np.float32) / 255.
    probe_comp, alpha_comp = render(competitor, capture_cv[:1], K[:1], args.size, args.size)
    sa = None
    frame = None
    if (result / "gaussians.ply").exists():
        sa = load_gaussians(result / "gaussians.ply")
        ids, model_cv, model_K = exported_cameras(result / "cameras.json")
        expected = [f"view_{i:02d}_source_{int(v['sourceIndex']):03d}" for i, v in enumerate(metadata["views"])]
        if ids != expected:
            raise RuntimeError(f"Input/export camera order mismatch: {ids} vs {expected}")
        frame = fit_source_frame(capture_cv, model_cv, expected, ids)
        model_sources = model_camera(capture_cv, frame)
        model_first = model_sources[:1]
        probe_sa, alpha_sa = render(sa, model_first, K[:1], args.size, args.size)
        # 原生导出相机/756² K 与共同采集画布分开；两者不混算色差。
        native_full, _ = render(sa, model_cv, model_K, 756, 756)
        if len(model_K) != 10 or np.max(np.abs(model_K[:, :2, 2] - 378)) > 2.0:
            raise RuntimeError("Expected exported K on the official 756-square model canvas")
        native_input = np.stack([np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
                                 for p in sorted((result / "rgb").glob("*.png"))])
        native_input_mae = [float(np.abs(native_full[i] - native_input[i]).mean())
                            for i in range(len(native_full))]
        native_common, native_common_alpha = render(sa, model_cv, model_K, args.size, args.size)
        common_rgb, common_alpha = render(sa, model_sources, K, args.size, args.size)
        native_metrics = [{"view": i,
                           "model_canvas_size": [756, 756],
                           "common_capture_canvas_size": [args.size, args.size],
                           "rgb_mae": float(np.abs(native_common[i] - common_rgb[i]).mean()),
                           "alpha_mae": float(np.abs(native_common_alpha[i] - common_alpha[i]).mean()),
                           "ply_vs_actual_input_rgb_mae": native_input_mae[i],
                           "native_replay_gate": "diagnostic_only_no_threshold"}
                          for i in range(len(expected))]
        dump(result / "alignment.json", {**frame, "native_vs_common_source_camera": native_metrics,
                                        "note": "Native exported K fits the preprocessed model canvas; common rendering uses original acquisition K. Different canvas/intrinsics, not an NVS metric."})
    else:
        probe_sa, alpha_sa = np.zeros_like(probe_comp), np.zeros_like(alpha_comp)
    # 首源图是输入自洽检查，不把其 PSNR 当 NVS 成绩。
    panels = [overlay(gt, "输入参考", "view 00 / sourceIndex 0", args.size),
              overlay(probe_comp[0], "竞品 PLY", f"SH3 / {competitor['point_count']:,} Gaussians", args.size)]
    if sa is not None:
        panels.append(overlay(probe_sa[0], "SplatAnything", f"DC / {sa['point_count']:,} Gaussians", args.size))
    Image.fromarray(np.concatenate(panels, axis=1)).save(out / "source_camera_probe.jpg", quality=95)
    if sa is None or args.probe_only:
        dump(out / "probe.json", {"competitor": competitor["point_count"], "sa_ready": sa is not None,
                                 "source_alignment": frame, "sh_degree": {"competitor": competitor["degree"]}})
        print(f"✅ Camera probe: {out / 'source_camera_probe.jpg'}")
        return
    orbit_cv, trajectory = orbit(bounds, args.frames, capture_cv @ FLIP_HOM)
    orbit_model = model_camera(orbit_cv, frame)
    orbit_K = np.repeat(K[:1], args.frames, axis=0)
    sa_rgb, sa_a = render(sa, orbit_model, orbit_K, args.size, args.size)
    if args.sa_only:
        comp_rgb = np.zeros_like(sa_rgb)
        comp_a = np.zeros_like(sa_a)
    else:
        comp_rgb, comp_a = render(competitor, orbit_cv, orbit_K, args.size, args.size)
    if not args.sa_only:
        save_video([overlay(comp_rgb[i], "竞品 PLY | 相机契约待确认",
                            f"source interpolate {i:03d} / native SH3", args.size)
                    for i in range(args.frames)], root / "competitor.mp4", args.fps)
        save_video([np.concatenate([
                        overlay(comp_rgb[i], "竞品 PLY | 相机契约待确认", "同协议回放，非校准优劣", args.size),
                        overlay(sa_rgb[i], "SplatAnything", "十图 + 已知相机 / native DC", args.size)], axis=1)
                    for i in range(args.frames)], root / "comparison.mp4", args.fps)
    save_video([overlay(sa_rgb[i], "SplatAnything", f"source interpolate {i:03d} / native DC", args.size)
                for i in range(args.frames)], root / "splatanything.mp4", args.fps)
    preview_indices = sorted(set(int(x) for x in np.linspace(0, args.frames - 1, 4)))
    preview = [np.concatenate([
                   overlay(comp_rgb[i], "竞品 PLY | 契约待确认", f"frame {i:03d}", args.size),
                   overlay(sa_rgb[i], "SplatAnything", f"frame {i:03d}", args.size)], axis=1)
               for i in preview_indices]
    Image.fromarray(np.concatenate(preview, axis=0)).resize((1408, 4 * 388)).save(
        root / "comparison_contact_sheet.jpg", quality=95)
    dump(root / "comparison.json", {
        "status": "SA rendered; competitor replay not certified for algorithm ranking",
        "protocol_version": 2,
        "camera_alignment": frame, "trajectory": trajectory,
        "protocol": "same capture-derived source-interpolation cameras for both; no competitor fit or scale tuning; supplied bounds center and distance range; native SH3 vs SA DC; no 360 extrapolation",
        "competitor": {"ply": competitor_path.name, "points": competitor["point_count"],
                       "sh_degree": competitor["degree"], "quaternion": args.competitor_quat,
                       "replay_limitation": "Given cameras produce a close-up, locally blurred rendering; no viewer/coordinate/scale documentation supplied. No quality winner inferred.",
                       "full_ply_preserved": True, "diagnostic_filter_used_in_video": False,
                       "rendered": not args.sa_only},
        "splatanything": {"ply": "splatanything.ply", "points": sa["point_count"],
                          "sh_degree": sa["degree"], "quaternion": "passthrough",
                          "source_camera_alignment_is_not_nvs": True,
                          "native_vs_common_source_camera": native_metrics},
        "frames": args.frames, "fps": args.fps, "image_size": args.size,
        "video_encoding": "H264 yuv420p with frame-count validation when FFMPEG names an existing executable; otherwise MJPG AVI",
        "source_view_metrics_not_nvs": True})
    print(f"🎉 Done: {root / 'comparison_contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
