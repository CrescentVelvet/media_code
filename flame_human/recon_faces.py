#!/usr/bin/env python3
"""recon_faces.py — 阶段三：各向同性裁剪 → 468 点 landmark + DECA/PnP 初值。

三件事：
  1. 各向同性 square crop：以 bbox 中心为心、边长 L = max(w,h) × scale_exp 的正方形，
     越界先 pad 整图（edge replicate，不用黑边——黑边会在 crop 里造出强边缘），
     再 resize 到目标尺寸。因为源是正方形，x/y 缩放比相等，**657/468 点回全图无畸变**。
  2. MediaPipe FaceMesh 出 468 点 2D 观测（阶段四的主观测）。
     DECA 只输出 68 点，解不了 100 维 exp（见设计文档可解性表），所以观测必须 468。
  3. 初值：DECA 给 shape/expr/pose（可选，加载失败自动退回）；
     PnP 用 FLAME mean face 的 468 个 3D 点 + 2D 观测解每帧位姿，
     作为阶段 4.1 的 local_SRT 初值（这一步不依赖 DECA，一定能算）。

越界帧打 `flag_outbound=True`，阶段 4.1 的 reject 会优先剔除（设计文档 §3 遗留项 2）。

Env:
  MATCH_JSON     阶段二输出
  FACE_BOXES     阶段一输出
  SOURCE_DIR     COLMAP 场景（内参从 sparse 读）
  IMAGES_DIR     图像目录（默认 $SOURCE_DIR/images）
  OUT_JSON       输出（默认 $RESULTS_DIR/04_recon/face_recon.json）
  CROP_SCALE_LM  landmark 裁剪倍率（默认 1.6，与 vggt_human PAD_RATIO=0.6 等价）
  CROP_SCALE_DECA DECA 裁剪倍率（默认 1.25，对齐 DECA 官方 demo）
  LM_SIZE        landmark crop 输出尺寸（默认 512）
  USE_DECA       是否加载 DECA（默认 1，失败自动退回 PnP）
"""
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_cameras, view_index  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
N_LM = 468
N_SHAPE = 300   # FLAME 2020 identity 维度
N_EXPR = 100    # FLAME 2020 expression 维度


def _pad_to(a, n):
    """DECA 默认 n_shape=100 / n_exp=50，FLAME 2020 是 300 / 100 → 补零对齐。"""
    a = np.asarray(a, dtype=np.float64).ravel()
    if len(a) >= n:
        return a[:n].tolist()
    return np.pad(a, (0, n - len(a))).tolist()


def log(m):
    print(m, flush=True)


def square_crop(img, bbox, scale_exp, out_size):
    """各向同性正方形裁剪。

    返回 (crop, meta)。meta 含反变换所需的 cx/cy/L 与 flag_outbound。
    逆变换：u_full = (u_crop / out_size) * L + cx - L/2   （与 pad 量无关）
    """
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in bbox]
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    L = max(w, h) * float(scale_exp)
    if L < 8:
        return None, None

    half = L * 0.5
    # 越界判定用「未 pad 时的方形窗口」
    flag_outbound = bool(cx - half < 0 or cy - half < 0 or
                         cx + half > W or cy + half > H)

    px0, py0 = cx - half, cy - half
    # pad 整图（edge replicate）再裁，保证一定是 L×L 且没有黑边
    pad_l = max(0, int(np.ceil(-px0)))
    pad_t = max(0, int(np.ceil(-py0)))
    pad_r = max(0, int(np.ceil(px0 + L - W)))
    pad_b = max(0, int(np.ceil(py0 + L - H)))
    if pad_l or pad_t or pad_r or pad_b:
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r,
                                 cv2.BORDER_REPLICATE)
    ax0, ay0 = int(round(px0 + pad_l)), int(round(py0 + pad_t))
    crop = img[ay0:ay0 + int(round(L)), ax0:ax0 + int(round(L))]
    if crop.size == 0:
        return None, None
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    meta = {
        "cx": float(cx), "cy": float(cy), "L": float(L),
        "size": int(out_size), "scale_exp": float(scale_exp),
        "flag_outbound": flag_outbound,
    }
    return crop, meta


def crop_to_full(uv_crop, meta):
    """crop 坐标 → 全图坐标（纯标量缩放 + 平移，各向同性）。"""
    s = meta["L"] / float(meta["size"])
    off = np.array([meta["cx"] - meta["L"] * 0.5,
                    meta["cy"] - meta["L"] * 0.5], dtype=np.float64)
    return np.asarray(uv_crop, dtype=np.float64) * s + off


def init_deca():
    """加载 DECA。失败返回 None（不致命：退回 PnP + 零初值）。"""
    deca_dir = os.environ.get("DECA_DIR", "")
    ckpt = os.environ.get("DECA_CKPT", "")
    if not deca_dir or not Path(deca_dir).is_dir():
        log("  ⚠️  DECA_DIR 不存在，跳过 DECA（用 PnP + 零初值）")
        return None
    if not ckpt or not Path(ckpt).exists():
        log(f"  ⚠️  DECA 权重不存在: {ckpt}，跳过 DECA")
        return None
    try:
        sys.path.insert(0, deca_dir)
        from decalib.deca import DECA
        from decalib.utils.config import cfg as deca_cfg
        # DECA config 真实属性是小写 flame_model_path（decalib/utils/config.py:27），
        # 指向 generic_model.pkl 文件路径。FLAME_MODEL 语义一致，直接传。
        flame_model = os.environ.get("FLAME_MODEL", "")
        if flame_model:
            deca_cfg.model.flame_model_path = flame_model
        # 只要 shape/expr/pose 初值，不需要纹理（省 1.2GB albedo 加载）
        deca_cfg.model.use_tex = False
        m = DECA(config=deca_cfg, device="cuda" if _has_cuda() else "cpu")
        m.eval()
        log("  ✅ DECA 加载成功")
        return m
    except Exception as e:
        log(f"  ⚠️  DECA 加载失败（{type(e).__name__}: {e}），退回 PnP + 零初值")
        return None


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main():
    source_dir = os.environ.get("SOURCE_DIR", "")
    results_dir = os.environ.get("RESULTS_DIR", "")
    match_json = Path(os.environ.get(
        "MATCH_JSON", f"{results_dir}/03_match/face_person_match.json"))
    images_dir = Path(os.environ.get("IMAGES_DIR", f"{source_dir}/images"))
    out_json = Path(os.environ.get(
        "OUT_JSON", f"{results_dir}/04_recon/face_recon.json"))
    crop_scale_lm = float(os.environ.get("CROP_SCALE_LM", "1.6"))
    crop_scale_deca = float(os.environ.get("CROP_SCALE_DECA", "1.25"))
    lm_size = int(os.environ.get("LM_SIZE", "512"))
    use_deca = int(os.environ.get("USE_DECA", "1"))
    min_det_conf = float(os.environ.get("MIN_DET_CONF", "0.5"))

    for p in (match_json,):
        if not p.exists():
            sys.exit(f"❌ 缺少输入: {p}")

    log("🧩 [阶段三] 人脸重建输入准备")

    # FLAME mean face 上 468 点的 3D 位置（PnP 要用）
    emb_path = os.environ.get("FLAME_LM468_EMBEDDING", "")
    lm3d_mean = None
    if emb_path and Path(emb_path).exists():
        z = np.load(emb_path)
        verts = z["verts_mean"]
        lm3d_mean = (verts[z["lm_verts"]] * z["lm_bary"][:, :, None]).sum(1)
        log(f"  📌 lm468 嵌入: {emb_path} ({lm3d_mean.shape})")
    else:
        log(f"  ⚠️  无 lm468 嵌入 ({emb_path})，PnP 位姿初值不可用"
            f"（03 → bash 01b_build_lm468_embedding.sh）")

    views = view_index(read_cameras(source_dir))
    log(f"  📷 相机: {len(views)} views")

    match = json.loads(match_json.read_text())
    frames = match.get("frames", {})

    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=False,   # 468 点（开 refine 会多 10 个虹膜点）
        min_detection_confidence=min_det_conf)

    deca = init_deca() if use_deca else None

    out_frames = {}
    stats = {"ok": 0, "no_image": 0, "crop_failed": 0,
             "no_landmarks": 0, "outbound": 0}
    t0 = time.time()
    stems = sorted(frames.keys())
    for i, stem in enumerate(stems):
        img_path = None
        for ext in IMG_EXTS:
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            stats["no_image"] += 1
            out_frames[stem] = {"persons": {}, "reason": "no_image"}
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        rec = {"image": img_path.name, "persons": {}}
        view = views.get(stem)

        for oid, m in frames[stem].get("persons", {}).items():
            r = {"ok": False, "reason": ""}
            bbox = m["bbox"]
            crop_lm, meta_lm = square_crop(img, bbox, crop_scale_lm, lm_size)
            if crop_lm is None:
                r["reason"] = "crop_failed"
                stats["crop_failed"] += 1
                rec["persons"][oid] = r
                continue
            r["crop_lm"] = meta_lm
            if meta_lm["flag_outbound"]:
                stats["outbound"] += 1

            res = face_mesh.process(crop_lm)
            if not res.multi_face_landmarks:
                r["reason"] = "no_landmarks"
                stats["no_landmarks"] += 1
                rec["persons"][oid] = r
                continue
            Hc, Wc = crop_lm.shape[:2]
            uv = np.array([[lm.x * Wc, lm.y * Hc]
                           for lm in res.multi_face_landmarks[0].landmark],
                          dtype=np.float64)
            if uv.shape[0] != N_LM:
                r["reason"] = f"lm_count_{uv.shape[0]}"
                stats["no_landmarks"] += 1
                rec["persons"][oid] = r
                continue
            lm_full = crop_to_full(uv, meta_lm)
            r["lm468"] = [[float(a), float(b)] for a, b in lm_full]

            # ── PnP 位姿初值（不依赖 DECA）────────────────────────────
            if lm3d_mean is not None and view is not None:
                K = np.array([[view["fx"], 0, view["cx"]],
                              [0, view["fy"], view["cy"]],
                              [0, 0, 1.0]], dtype=np.float64)
                ok, rvec, tvec, inl = cv2.solvePnPRansac(
                    lm3d_mean.astype(np.float64),
                    lm_full.astype(np.float64), K, None,
                    iterationsCount=100, reprojectionError=8.0,
                    confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
                if ok:
                    R, _ = cv2.Rodrigues(rvec)
                    proj = (K @ (R @ lm3d_mean.T + tvec)).T
                    uvp = proj[:, :2] / np.clip(proj[:, 2:3], 1e-9, None)
                    rms = float(np.sqrt(((uvp - lm_full) ** 2).sum(1).mean()))
                    r["pnp"] = {"R": R.ravel().tolist(),
                                "t": tvec.ravel().tolist(), "rms": rms}

            # ── DECA 初值（可选）───────────────────────────────────────
            # DECA 默认 n_shape=100 / n_exp=50，而 FLAME 2020 是 300 / 100，
            # 不足的补零（多余的主成分本来也接近零均值）。
            if deca is not None:
                try:
                    crop_d, _ = square_crop(img, bbox, crop_scale_deca, 224)
                    if crop_d is not None:
                        import torch
                        # DECA 的 E_flame 不做归一化，期望输入已是 [-1,1]
                        # （与其 datasets 里的 Normalize(0.5, 0.5) 一致）
                        t = (torch.from_numpy(crop_d).permute(2, 0, 1)
                             .float().unsqueeze(0) / 255.0) * 2.0 - 1.0
                        with torch.no_grad():
                            code = deca.encode(t)
                        sh = np.asarray(code["shape"].detach().cpu()).ravel()
                        ex = np.asarray(code["exp"].detach().cpu()).ravel()
                        po = np.asarray(code["pose"].detach().cpu()).ravel()
                        r["deca"] = {
                            "shape": _pad_to(sh, N_SHAPE),
                            "exp": _pad_to(ex, N_EXPR),
                            # pose: [global_rot(3), jaw(3)]；
                            # 头部姿态以 PnP 为准（多视角一致），这里只留 jaw 备用
                            "jaw": po[3:6].tolist() if len(po) >= 6 else [],
                        }
                except Exception as e:
                    log(f"  ⚠️  DECA 推理失败 {stem}/{oid}: "
                        f"{type(e).__name__}: {e}")

            r["ok"] = True
            stats["ok"] += 1
            rec["persons"][oid] = r

        out_frames[stem] = rec
        if (i + 1) % 20 == 0 or i == len(stems) - 1:
            el = time.time() - t0
            log(f"  … {i+1}/{len(stems)} 帧 ({el:.0f}s) {stats}")

    if face_mesh is not None:
        face_mesh.close()

    out = {
        "meta": {
            "match_json": str(match_json),
            "images_dir": str(images_dir),
            "crop_scale_lm": crop_scale_lm,
            "crop_scale_deca": crop_scale_deca,
            "lm_size": lm_size,
            "use_deca": bool(deca is not None),
            "n_frames": len(out_frames),
            "stats": stats,
        },
        "frames": out_frames,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out))
    log(f"💾 {out_json}")
    log(f"📊 ok={stats['ok']} no_landmarks={stats['no_landmarks']} "
        f"outbound={stats['outbound']}")
    log("🎉 阶段三完成。下一步：bash 05_align_3dmm.sh")


if __name__ == "__main__":
    main()
