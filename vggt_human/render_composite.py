#!/usr/bin/env python3
"""三模型合成渲染验证：scene + bodies + heads(warped) 合成 vs GT。

每个模型单独渲染（各自独立 ply），head 按逐帧 warp，然后 alpha 合成：
  C = head_rgb + (1-head_a)·[body_rgb + (1-body_a)·scene_rgb]
合成前提：三区域 mask 互斥（100% 验证过），跨模型遮挡由 alpha 链自然处理。
输出：合成 mosaic + 全帧 PSNR + 头部区域内 PSNR。
"""
import os, sys, json, math
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as _Rot

os.environ.setdefault("OMP_NUM_THREADS", "8")

RESULTS = os.environ.get("RESULTS_DIR", "/mnt/d/output/vggt_human_ms")
ITERS = os.environ.get("ITERS", "4000")                      # 默认（兼容旧用法）
HEAD_ITERS = os.environ.get("HEAD_ITERS", ITERS)             # head 各自 iteration
BODY_ITERS = os.environ.get("BODY_ITERS", ITERS)             # body 各自 iteration
SCENE_ITERS = os.environ.get("SCENE_ITERS", ITERS)           # scene iteration
WITH_BASELINE = os.environ.get("BASELINE", "0") == "1"       # 同帧加载 04b 基线对比
CLAMP = os.environ.get("CLAMP", "1") == "1"                  # mask 约束合成（切除越界 alpha）
CLAMP_DILATE = int(os.environ.get("CLAMP_DILATE", "10"))     # person mask 膨胀 px（软边容忍）
HEAD_CROP = os.environ.get("HEAD_CROP", "0") == "1"          # 输出 head 区高分辨率裁剪对比
N_VIS = int(os.environ.get("N_VIS", "6"))
OUT_DIR = f"{RESULTS}/03i_composite_vis"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/repos/gaussian-splatting"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from render_head_compare import load_gs, MiniGS, warp_gs, restore_gs, qvec2rotmat_np
from gaussian_renderer import render
from scene.cameras import Camera
from argparse import Namespace
from face_center_3d import parse_colmap_cameras
from pathlib import Path

PIDS = ["00", "01", "02"]


def make_camera(v, W, H):
    FoVx = 2 * math.atan(W / (2 * v["fx"]))
    FoVy = 2 * math.atan(H / (2 * v["fy"]))
    return Camera(resolution=(W, H), colmap_id=0,
                  R=qvec2rotmat_np(v["qvec"]).T, T=np.asarray(v["tvec"]),
                  FoVx=FoVx, FoVy=FoVy, depth_params=None,
                  image=Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)),
                  invdepthmap=None, image_name=v["stem"], uid=0,
                  data_device="cuda")


def render_rgb_alpha(g, cam, pipe, bg):
    """渲染两次：正常 RGB + 染白 alpha。

    ⚠️ alpha pass 只染白（SH_DC=1.7724539 → color=1），**保留原 opacity**：
    渲染值 = Σ α_i T_i = 该层真实累积 alpha，与 RGB pass 一致，合成才是
    数学正确的 over 操作。若强制 opacity=10（几何覆盖），densify 后大量
    半透明高斯被夸大成不透明 → body 层过度遮挡 scene 层，全帧 PSNR 反降
    （实测 30k body vs 4k body：12.02→11.04dB）。
    FORCE_OPACITY=1 可切回旧模式（仅对照用）。
    """
    force = os.environ.get("FORCE_OPACITY", "0") == "1"
    with torch.no_grad():
        rgb = render(cam, g, pipe, bg)["render"].clamp(0, 1)
        dc0 = g._features_dc.clone()
        g._features_dc = torch.full_like(dc0, 1.7724539)
        op0 = None
        if force:
            op0 = g._opacity.clone()
            g._opacity = torch.full_like(op0, 10.0)
        a = render(cam, g, pipe, bg)["render"].clamp(0, 1).mean(0, keepdim=True)
        g._features_dc = dc0
        if force:
            g._opacity = op0
    return rgb, a


def main():
    fit = json.loads(open(f"{RESULTS}/03e_head_3dmm/head_fit.json").read())
    views = {v["stem"]: v for v in parse_colmap_cameras(f"{RESULTS}/03b_source_ba")}

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                     antialiasing=False, debug=False)
    bg = torch.zeros(3, device="cuda")

    # 加载 7 个模型（head/body/scene 可各自指定 iteration）
    # BODY_ITERS_P00/P01/P02 可单独覆盖对应人（默认 fallback BODY_ITERS）
    models = {}
    g_scene = MiniGS(load_gs(f"{RESULTS}/03i_scene/point_cloud/iteration_{SCENE_ITERS}/point_cloud.ply"))
    models["scene"] = g_scene
    heads = {}
    body_iters_desc = []
    for pid in PIDS:
        _bi = os.environ.get(f"BODY_ITERS_P{pid}", BODY_ITERS)
        body_iters_desc.append(f"p{pid}={_bi}")
        heads[pid] = MiniGS(load_gs(f"{RESULTS}/03i_head_p{pid}/point_cloud/iteration_{HEAD_ITERS}/point_cloud.ply"))
        models[f"body{pid}"] = MiniGS(load_gs(f"{RESULTS}/03i_body_p{pid}/point_cloud/iteration_{_bi}/point_cloud.ply"))
    print(f"models loaded: {len(models)+len(heads)}  (head={HEAD_ITERS} body[{','.join(body_iters_desc)}] scene={SCENE_ITERS})")

    g_base = None
    if WITH_BASELINE:
        _p = f"{RESULTS}/04b_model_3dgs_ba/point_cloud/iteration_30000/point_cloud.ply"
        if os.path.isfile(_p):
            g_base = MiniGS(load_gs(_p))
            print(f"baseline loaded: 04b iter30000 ({len(g_base._xyz)} gaussians)")

    stems_all = sorted(views.keys())
    step = max(1, len(stems_all) // N_VIS)
    picked = stems_all[::step][:N_VIS]

    os.makedirs(OUT_DIR, exist_ok=True)
    psnrs_full, psnrs_head, psnrs_body, psnrs_base = [], [], [], []
    rows = []
    for stem in picked:
        v = views[stem]
        W, H = int(v["W"]), int(v["H"])
        gt_files = list(Path(f"{RESULTS}/03b_source_ba/images").glob(f"{stem}.*"))
        if not gt_files: continue
        gt = np.asarray(Image.open(gt_files[0]).convert("RGB"), dtype=np.float32) / 255.0
        cam = make_camera(v, W, H)

        # scene 渲染（最底层）
        c_scene, _ = render_rgb_alpha(g_scene, cam, pipe, bg)
        comp = c_scene.clone()
        acc_a = torch.zeros(1, H, W, device="cuda")

        # mask 约束：body 层 alpha 限制在自己 body mask（膨胀）内，切除越界高斯
        # （body 30k densify 后高斯漂移出界，不切会污染 scene 区，实测 +6dB 收益）
        body_masks, head_masks = {}, {}
        if CLAMP:
            import cv2
            k = np.ones((CLAMP_DILATE*2+1, CLAMP_DILATE*2+1), np.uint8)
            for pid in PIDS:
                bpath = f"{RESULTS}/03i_region_masks/{stem}.p{pid}.body.png"
                hpath = f"{RESULTS}/03i_region_masks/{stem}.p{pid}.head.png"
                if os.path.isfile(bpath):
                    m = np.asarray(Image.open(bpath).convert("L"), dtype=np.float32) / 255.0
                    if m.max() >= 0.1:
                        m = cv2.dilate((m > 0.3).astype(np.uint8), k).astype(np.float32)
                        body_masks[pid] = torch.tensor(m, device="cuda")[None]
                if os.path.isfile(hpath):
                    m = np.asarray(Image.open(hpath).convert("L"), dtype=np.float32) / 255.0
                    if m.max() >= 0.1:
                        m = cv2.dilate((m > 0.3).astype(np.uint8), k).astype(np.float32)
                        head_masks[pid] = torch.tensor(m, device="cuda")[None]

        # body 渲染叠加（中间层），记录每人 clamp 后 alpha 供 head 遮挡计算
        body_alpha = {}
        for pid in PIDS:
            cb, ab = render_rgb_alpha(models[f"body{pid}"], cam, pipe, bg)
            if CLAMP and pid in body_masks:
                ab = ab * body_masks[pid]          # 切除 body mask 外越界 alpha
            body_alpha[pid] = ab
            comp = comp * (1 - ab * (1 - acc_a)) + cb * (ab * (1 - acc_a))
            acc_a = torch.clamp(acc_a + ab * (1 - acc_a), 0, 1)

        # head 渲染叠加（最上层，逐帧 warp）
        # ⚠️ over 算子方向：head 是顶层，其 alpha 不能被【自己 body】的累积 alpha
        # 衰减（自己身体永远在自己头后面）。旧式 a_h=ah*(1-acc_a) 把 acc_a(含自身
        # body, 实测 head 区内 0.84) 乘进去 → head 实际贡献仅 0.16 → 脸区露 body
        # 垃圾（斑点噪声）。正确：只被【其他人 body】衰减（跨人前置遮挡近似）。
        for pid in PIDS:
            fr = fit["persons"].get(pid)
            if fr is None: continue
            pf = fr["per_frame"].get(stem)
            if pf is None:
                continue
            R_ref = np.asarray(fr["R_ref"]); t_ref = np.asarray(fr["t_ref"])
            R_f = np.asarray(pf["R"]).reshape(3, 3)
            t_f = np.asarray(pf["t"]).reshape(3)
            A = R_f @ R_ref.T
            b = t_f - A @ t_ref
            g = heads[pid]
            x0, r0 = warp_gs(g, A, b)
            ch, ah = render_rgb_alpha(g, cam, pipe, bg)
            restore_gs(g, x0, r0)
            if CLAMP and pid in head_masks:
                ah = ah * head_masks[pid]          # head 层只约束到自己 head 区域
            # 其他人 body 的累积 alpha（跨人遮挡近似；不含自身 body）
            trans_other = torch.ones(1, H, W, device="cuda")
            for j in PIDS:
                if j != pid and j in body_alpha:
                    trans_other = trans_other * (1 - body_alpha[j])
            a_h = ah * trans_other                 # 只被其他人 body 遮挡衰减
            comp = comp * (1 - a_h) + ch * a_h
            acc_a = torch.clamp(acc_a + a_h * (1 - acc_a), 0, 1)

        # 全帧 PSNR
        gt_t = torch.tensor(gt, device="cuda").permute(2, 0, 1)
        mse = ((comp - gt_t) ** 2).mean()
        p_full = float(-10 * np.log10(mse.item() + 1e-8))
        psnrs_full.append(p_full)

        # body 区域内 PSNR（p00 为主，三人 body mask 并集）
        best_body_psnr = None
        for pid in PIDS:
            mpath = f"{RESULTS}/03i_region_masks/{stem}.p{pid}.body.png"
            if not os.path.isfile(mpath): continue
            m = np.asarray(Image.open(mpath).convert("L"), dtype=np.float32) / 255.0
            if m.max() < 0.1: continue
            m_t = torch.tensor(m, device="cuda")[None]
            mse_b = (((comp - gt_t) ** 2) * m_t).sum() / (m_t.sum() * 3 + 1e-8)
            p_b = float(-10 * np.log10(mse_b.item() + 1e-8))
            best_body_psnr = p_b if best_body_psnr is None else max(best_body_psnr, p_b)
        if best_body_psnr is not None:
            psnrs_body.append(best_body_psnr)

        # 04b 基线同帧全帧 PSNR
        if g_base is not None:
            with torch.no_grad():
                base_img = render(cam, g_base, pipe, bg)["render"].clamp(0, 1)
            mse_b = ((base_img - gt_t) ** 2).mean()
            psnrs_base.append(float(-10 * np.log10(mse_b.item() + 1e-8)))

        # head 区域内 PSNR（任一 head mask）
        best_head_psnr = None
        for pid in PIDS:
            mpath = f"{RESULTS}/03i_region_masks/{stem}.p{pid}.head.png"
            if not os.path.isfile(mpath): continue
            m = np.asarray(Image.open(mpath).convert("L"), dtype=np.float32) / 255.0
            if m.max() < 0.1: continue
            m_t = torch.tensor(m, device="cuda")[None]
            mse_h = (((comp - gt_t) ** 2) * m_t).sum() / (m_t.sum() * 3 + 1e-8)
            p_h = float(-10 * np.log10(mse_h.item() + 1e-8))
            best_head_psnr = p_h if best_head_psnr is None else max(best_head_psnr, p_h)
        if best_head_psnr is not None:
            psnrs_head.append(best_head_psnr)

        comp_np = comp.permute(1, 2, 0).cpu().numpy()

        # head 区高分辨率裁剪对比（GT | composite | 04b）+ Laplacian 锐度
        if HEAD_CROP:
            import cv2
            base_np = base_img.permute(1, 2, 0).cpu().numpy() if g_base is not None else None
            for pid in PIDS:
                mpath = f"{RESULTS}/03i_region_masks/{stem}.p{pid}.head.png"
                if not os.path.isfile(mpath): continue
                m = np.asarray(Image.open(mpath).convert("L"), dtype=np.float32) / 255.0
                if m.max() < 0.1: continue
                ys, xs = np.where(m > 0.5)
                if len(ys) == 0: continue
                pad = 40
                y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
                x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
                tiles = [gt[y0:y1, x0:x1], comp_np[y0:y1, x0:x1]]
                names = ["GT", "comp"]
                if base_np is not None:
                    tiles.append(base_np[y0:y1, x0:x1]); names.append("04b")
                def sharp(c):
                    g8 = (cv2.cvtColor((np.clip(c, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
                    return cv2.Laplacian(g8, cv2.CV_64F).var()
                sh = [sharp(t) for t in tiles]
                rowimg = np.concatenate(tiles, axis=1)
                Image.fromarray((np.clip(rowimg, 0, 1) * 255).astype(np.uint8)).save(
                    f"{OUT_DIR}/headcrop_{stem}_p{pid}.png")
                print(f"  ✂️ headcrop p{pid} {stem}: " + "  ".join(f"{n} sharp={s:.0f}" for n, s in zip(names, sh)))

        side = np.concatenate([gt, comp_np], axis=1)
        rows.append((stem, side, p_full))

    psnrs_full = np.array(psnrs_full)
    psnrs_head = np.array(psnrs_head)
    print(f"\n=== 三模型合成渲染（{len(rows)} 帧）===")
    print(f"  全帧 PSNR : med={np.median(psnrs_full):.2f}  mean={psnrs_full.mean():.2f}  min={psnrs_full.min():.2f}")
    if len(psnrs_body):
        print(f"  body 区 PSNR: med={np.median(psnrs_body):.2f}  mean={np.array(psnrs_body).mean():.2f}")
    print(f"  head 区 PSNR: med={np.median(psnrs_head):.2f}  mean={psnrs_head.mean():.2f}" if len(psnrs_head) else "  head 区: n/a")
    if psnrs_base:
        pb = np.array(psnrs_base)
        print(f"  04b 基线同帧: med={np.median(pb):.2f}  mean={pb.mean():.2f}  (Δ={psnrs_full.mean()-pb.mean():+.2f} dB)")

    if rows:
        th = min(r.shape[0] for _, r, _ in rows)
        mosaic = np.concatenate([r[:th] for _, r, _ in rows], axis=0)
        img = Image.fromarray((np.clip(mosaic, 0, 1) * 255).astype(np.uint8))
        total = Image.new("RGB", (mosaic.shape[1], mosaic.shape[0] + 34), (10, 10, 10))
        total.paste(img, (0, 34))
        dr = ImageDraw.Draw(total)
        dr.text((10, 8), f"composite [GT | render]  full-PSNR med={np.median(psnrs_full):.2f} dB", fill=(255, 255, 255))
        out = f"{OUT_DIR}/composite_mosaic.png"
        total.save(out)
        print(f"\n🖼️ {out}")
        for s, _, p in rows:
            print(f"  {s}: {p:.2f} dB")


if __name__ == "__main__":
    main()
