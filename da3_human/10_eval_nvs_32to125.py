#!/usr/bin/env python3
"""10_eval_nvs_32to125.py — 稀疏输入 NVS 口径评估：N 个视角喂 DA3，渲染全部 125 个视角。

为什么需要这个脚本：
  07_eval_da3_gs.py 只能渲染「与高斯同一次前向」的相机，也就是 in-sample。
  而 DA3 的定位是 NVS——喂少量视角，合成没见过的视角。要把 32 视角的高斯渲染到
  125 个视角，必须先把高斯从「DA3 世界系」搬到「COLMAP 世界系」（两者差一个 Sim(3)）。

对齐链：
  DA3 的 N 个相机中心  --Umeyama Sim(3)-->  COLMAP 的同一批相机中心
  means' = s·R·means + t ；scales' = s·scales ；quats' = q_R ⊗ quats
  （SH 与视角方向绑定，旋转世界系必须同步旋转 SH；本脚本先只支持 DC-only PLY，
    以免在 SH 旋转上引入未验证的误差——full-SH 版本等 DC 口径确认后再加。）

Env:
  PLY        高斯 PLY（DC-only 优先）
  D3NPZ      32 视角那次前向的 cameras.npz（含 extrinsics/intrinsics）
  COLMAP_DIR COLMAP sparse/0（提供 125 视角相机与世界系）
  IMAGES_DIR GT 图像目录
  N_INPUT    当时喂给 DA3 的视角数（默认 32），用于重算抽帧索引
  W,H        渲染分辨率（默认 1440x1920）
"""
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------- PLY
def load_ply(path):
    with open(path, "rb") as f:
        raw = f.read(1 << 20)
        end = raw.index(b"end_header") + len(b"end_header") + 1
        head = raw[:end].decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", head).group(1))
        props = [ln.split()[2] for ln in head.splitlines() if ln.startswith("property")]
        f.seek(end)
        arr = np.fromfile(f, dtype=np.float32).reshape(n, len(props))
    nm = {k: i for i, k in enumerate(props)}
    means = arr[:, [nm["x"], nm["y"], nm["z"]]]
    op = arr[:, nm["opacity"]]
    sc = arr[:, [nm["scale_0"], nm["scale_1"], nm["scale_2"]]]
    q = arr[:, [nm["rot_0"], nm["rot_1"], nm["rot_2"], nm["rot_3"]]]
    dc = arr[:, [nm["f_dc_0"], nm["f_dc_1"], nm["f_dc_2"]]]
    rest = sorted((int(k.split("_")[-1]), i) for k, i in nm.items() if k.startswith("f_rest_"))
    if rest:
        r = arr[:, [i for _, i in rest]]
        k = r.shape[1] // 3
        r = r.reshape(n, 3, k).transpose(0, 2, 1)  # channel-major → (N,K,3)
        sh = np.concatenate([dc[:, None, :], r], axis=1)
    else:
        sh = dc[:, None, :]
    return means, sh, op, sc, q


# ---------------------------------------------------------------- Sim(3)
def umeyama_sim3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc_, dc_ = src - mu_s, dst - mu_d
    cov = (dc_.T @ sc_) / len(src)
    u, sv, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    diag = np.diag([1.0, 1.0, d])
    r = u @ diag @ vt
    var_s = (sc_ ** 2).sum() / len(src)
    s = float(np.trace(diag @ np.diag(sv)) / var_s)
    t = mu_d - s * (r @ mu_s)
    return s, r, t


def rot2quat_wxyz(r):
    from scipy.spatial.transform import Rotation
    xyzw = Rotation.from_matrix(r).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], np.float32)


def quat_mul_wxyz(a, b):
    """Hamilton product，a 在左（世界旋转左乘）。a/b: (...,4) wxyz"""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


# ---------------------------------------------------------------- COLMAP
def read_colmap(colmap_dir):
    """返回 {name: (w2c 4x4, K 3x3)}；支持 txt/bin 两种 sparse 格式。"""
    sys.path.insert(0, os.path.expanduser("~/repos/gaussian-splatting"))
    from scene import colmap_loader as CL  # noqa: E402

    d = Path(colmap_dir)
    if (d / "cameras.bin").exists():
        cams = CL.read_intrinsics_binary(str(d / "cameras.bin"))
        imgs = CL.read_extrinsics_binary(str(d / "images.bin"))
    else:
        cams = CL.read_intrinsics_text(str(d / "cameras.txt"))
        imgs = CL.read_extrinsics_text(str(d / "images.txt"))

    out = {}
    for im in imgs.values():
        cam = cams[im.camera_id]
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = CL.qvec2rotmat(im.qvec).astype(np.float32)
        w2c[:3, 3] = np.asarray(im.tvec, np.float32)
        if cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params[:4]
        elif cam.model in ("SIMPLE_RADIAL", "RADIAL"):
            f, cx, cy = cam.params[:3]
            fx = fy = f
        elif cam.model in ("OPENCV", "OPENCV_FISHEYE"):
            fx, fy, cx, cy = cam.params[:4]
        elif cam.model == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params[:3]
            fx = fy = f
        else:
            raise SystemExit(f"❌ 未支持的 COLMAP 相机模型: {cam.model}")
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
        out[Path(im.name).name] = (w2c, K)
    return out


def centers_of(w2c):
    r = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    return np.einsum("nij,nj->ni", -np.transpose(r, (0, 2, 1)), t)


def main():
    from gsplat import rasterization

    ply = os.environ["PLY"]
    d3npz = os.environ["D3NPZ"]
    colmap_dir = os.environ["COLMAP_DIR"]
    images_dir = os.path.expanduser(os.environ["IMAGES_DIR"])
    n_input = int(os.environ.get("N_INPUT", "32"))
    W, H = int(os.environ.get("W", "1440")), int(os.environ.get("H", "1920"))
    dev = "cuda"

    means_np, sh_np, op_np, sc_np, q_np = load_ply(ply)
    sh_deg = int(round(sh_np.shape[1] ** 0.5)) - 1
    log(f"📦 PLY {len(means_np):,} 高斯 | SH degree {sh_deg}")

    all_imgs = []
    for e in ("*.jpg", "*.jpeg", "*.png"):
        all_imgs.extend(sorted(Path(images_dir).glob(e)))
    all_imgs = sorted(all_imgs)
    idx = np.linspace(0, len(all_imgs) - 1, n_input).round().astype(int)
    input_names = [all_imgs[i].name for i in idx]
    log(f"🖼️ 全部 {len(all_imgs)} 视角，其中 {len(input_names)} 个喂给 DA3")

    d3 = np.load(d3npz)
    ext = d3["extrinsics"]
    assert len(ext) == n_input, f"DA3 相机数 {len(ext)} != N_INPUT {n_input}"
    d3_w2c = np.concatenate(
        [ext, np.tile(np.array([[[0, 0, 0, 1]]], np.float32), (len(ext), 1, 1))], axis=1)

    cm = read_colmap(colmap_dir)
    log(f"📷 COLMAP {len(cm)} 相机")
    missing = [n for n in input_names if n not in cm]
    if missing:
        raise SystemExit(f"❌ COLMAP 缺视角: {missing[:5]}")

    c_d3 = centers_of(d3_w2c)
    c_cm = centers_of(np.stack([cm[n][0] for n in input_names]))
    s, r, t = umeyama_sim3(c_d3, c_cm)
    resid = np.linalg.norm(s * (c_d3 @ r.T) + t - c_cm, axis=1)
    log(f"🎯 Sim(3) s={s:.4f} | 对齐残差 mean {resid.mean()*1000:.1f}mm "
        f"p95 {np.percentile(resid,95)*1000:.1f}mm max {resid.max()*1000:.1f}mm")

    # ---- 高斯变换到 COLMAP 世界系
    # ⚠️ PLY 里的 scale_* 是 log 尺度，必须先 exp 再乘 s（漏 exp 会得到负尺度 →
    #    gsplat 投影半径异常 → isect_tiles 申请数百 GiB 直接 OOM）
    means = s * (means_np.astype(np.float64) @ r.T) + t
    scales = np.exp(sc_np.astype(np.float64)) * s
    q_r = rot2quat_wxyz(r)
    quats = quat_mul_wxyz(np.broadcast_to(q_r, q_np.shape), q_np)

    means = torch.from_numpy(means.astype(np.float32)).to(dev)
    scales = torch.from_numpy(scales.astype(np.float32)).to(dev)
    quats = torch.from_numpy(quats.astype(np.float32)).to(dev)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    opac = torch.sigmoid(torch.from_numpy(op_np).to(dev))
    sh = torch.from_numpy(sh_np).to(dev)

    names_cm = list(cm.keys())
    input_set = set(input_names)
    ps, ps_seen, ps_unseen = [], [], []
    for name in names_cm:
        w2c, K = cm[name]
        p = Path(images_dir) / name
        g = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt = torch.from_numpy(g).permute(2, 0, 1).to(dev)
        if (gt.shape[1], gt.shape[2]) != (H, W):
            gt = torch.nn.functional.interpolate(gt[None], size=(H, W), mode="area")[0]
        Kk = K.copy()
        if H != 1920:
            Kk[0, :] *= W / 1440.0
            Kk[1, :] *= H / 1920.0
        with torch.no_grad():
            img, _, _ = rasterization(
                means, quats, scales, opac, sh,
                torch.from_numpy(w2c).float().to(dev)[None],
                torch.from_numpy(Kk).float().to(dev)[None],
                W, H, sh_degree=sh_deg)
        r_ = img[0].clamp(0, 1).permute(2, 0, 1)
        mse = ((r_ - gt) ** 2).mean().item()
        v = -10 * math.log10(max(mse, 1e-12))
        ps.append(v)
        (ps_seen if name in input_set else ps_unseen).append(v)

    log(f"\n📊 NVS 口径（{len(ps)} 视角，{W}x{H}）")
    log(f"  全部     PSNR {np.mean(ps):.2f}")
    log(f"  seen({len(ps_seen):3d})  PSNR {np.mean(ps_seen):.2f}")
    log(f"  unseen({len(ps_unseen):3d}) PSNR {np.mean(ps_unseen):.2f}")
    log(f"  per-view p10 {np.percentile(ps,10):.1f} p90 {np.percentile(ps,90):.1f}")


if __name__ == "__main__":
    main()
