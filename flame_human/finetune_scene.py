#!/usr/bin/env python3
"""finetune_scene.py — 阶段 08d：scene 高斯定向 finetune（治大 yaw 欠拟合）。

背景（composite 诊断，2026-09-10）：
  帧 24-33（相机 yaw -31°~-42°）composite 仅 9-11dB，读图确认整幅雾噪
  （背景货架糊成一片）。根因 = 上游 nn 模型在大 yaw 段欠拟合：该视角
  相机覆盖少、densify 不足，scene 高斯 opacity p50 仅 0.104（偏透明）。
  HYPIR 是 2D 人脸增强，作用域够不到 scene 的 3D 拟合问题——必须直接
  在监督下继续优化 scene 高斯 + densify 补点。

做法（镜像 08b body finetune 的教训）：
  1. 全属性自由 3DGS 优化（复用 finetune_body.BodyGS）；
  2. **监督区域 = person mask 之外**（scene 职责＝背景；body/head 区域
     不参与 loss，防止 scene 高斯被拉去拟合人）——这是 08b「监督区域
     必须与分支职责边界严格一致」教训的直接应用；
  3. **mask 内泄漏惩罚**（LAMBDA_INSIDE）：scene-only 渲染在 person 区
     域本应接近黑，惩罚其亮度防止 scene 高斯长进人物区造成 composite
     重影（镜像 08b 的 LAMBDA_OUTSIDE）；
  4. densify 标准流程 + MAX_GAUSS 上限（scene 起点 747k，防显存爆）。

Env: SCENE_PLY / MASKS_DIR / ALIGN_JSON / SOURCE_DIR / IMAGES_DIR /
     OUT_DIR / PID / EPOCHS / LR / DENSIFY_* / LAMBDA_DSSIM /
     LAMBDA_INSIDE / MAX_GAUSS
"""
import os
import sys
import json
import time
import random
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_cameras  # noqa: E402
from finetune_body import BodyGS, log  # noqa: E402  复用自由3DGS骨架


def main():
    results_dir = Path(os.environ.get("RESULTS_DIR", ""))
    scene_ply = Path(os.environ.get(
        "SCENE_PLY", f"{results_dir}/07_body_gs_src/scene_gs.ply"))
    masks_dir = Path(os.environ.get(
        "MASKS_DIR", f"{results_dir}/01c_sam3_person_masks"))
    align_json = Path(os.environ.get(
        "ALIGN_JSON", f"{results_dir}/05_align/head_align.json"))
    source_dir = os.environ.get("SOURCE_DIR", "")
    images_dir = os.environ.get("IMAGES_DIR", f"{source_dir}/images")
    out_dir = Path(os.environ.get(
        "OUT_DIR", f"{results_dir}/08d_finetune_scene"))
    pid = os.environ.get("PID", "0")
    epochs = int(os.environ.get("EPOCHS", "30"))
    lr = float(os.environ.get("LR", "1e-3"))
    densify_from = int(os.environ.get("DENSIFY_FROM", "3"))
    densify_until = int(os.environ.get("DENSIFY_UNTIL", "20"))
    densify_every = int(os.environ.get("DENSIFY_EVERY", "3"))
    grad_thresh = float(os.environ.get("DENSIFY_GRAD", "2e-4"))
    min_opacity = float(os.environ.get("MIN_OPACITY", "0.005"))
    lam_dssim = float(os.environ.get("LAMBDA_DSSIM", "0.2"))
    lam_inside = float(os.environ.get("LAMBDA_INSIDE", "0.3"))
    max_gauss = int(os.environ.get("MAX_GAUSS", "1500000"))
    sh_every = int(os.environ.get("SH_EVERY", "8"))
    seed = int(os.environ.get("SEED", "0"))

    if not scene_ply.exists():
        sys.exit(f"❌ 缺少 scene ply: {scene_ply}")
    if not align_json.exists():
        sys.exit(f"❌ 缺少 05 对齐结果: {align_json}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"🏋️  [08d] scene finetune  device={dev}")
    log(f"  📦 {scene_ply.name}")

    model = BodyGS(scene_ply, dev)
    log(f"  🔢 {len(model._xyz):,} 高斯, SH{model.max_sh}")

    sys.path.insert(0, os.environ.get("GS_DIR", ""))
    try:
        from gaussian_renderer import render
        from utils.loss_utils import l1_loss, ssim
        from scene.cameras import Camera as GSCamera
    except Exception as e:
        sys.exit(f"❌ 无法导入 gaussian-splatting: {e}")
    from types import SimpleNamespace
    pipe = SimpleNamespace(compute_cov3D_python=False,
                           convert_SHs_python=False, debug=False,
                           antialiasing=False)

    views = read_cameras(source_dir)
    log(f"  📷 {len(views)} views")

    # person mask（p0 alpha）→ scene 监督区域 = 其补集
    from PIL import Image
    pid2 = pid.zfill(2)
    mask_cache = {}
    n_mask = 0
    for v in views:
        s = v["stem"]
        mp = masks_dir / f"{s}.p{pid2}.alpha.png"
        if not mp.exists():
            mp = masks_dir / f"{s}.p{pid2}.mask.png"
        if not mp.exists():
            continue
        m = Image.open(mp).convert("L")
        if m.size != (v["W"], v["H"]):
            m = m.resize((v["W"], v["H"]), Image.LANCZOS)
        m = np.asarray(m, dtype=np.float32) / 255.0
        # scene 监督区域 = 1 - person_mask（背景才是 scene 的职责）
        mask_cache[s] = torch.from_numpy(1.0 - m).unsqueeze(0)
        n_mask += 1
    log(f"  🎭 scene 监督区（mask 补集）: {n_mask}/{len(views)} 帧")
    if n_mask == 0:
        sys.exit("❌ 无 person mask，无法界定 scene 监督区域")

    align = json.loads(align_json.read_text())
    ap = align["persons"][pid]
    stems = ap["frames"]
    view_by = {v["stem"]: v for v in views}
    cams = []
    for s in stems:
        v = view_by.get(s)
        if v is None:
            continue
        cams.append((s, v))
    log(f"  🎬 训练相机 {len(cams)} 帧")

    def build_params():
        return [
            {"params": [model._xyz], "lr": lr},
            {"params": [model._opacity], "lr": lr},
            {"params": [model._scaling], "lr": lr},
            {"params": [model._rotation], "lr": lr},
            {"params": [model._features], "lr": lr},
        ]

    opt = torch.optim.Adam(build_params(), lr=lr)

    bg = torch.zeros(3, device=dev)
    _xyz_np = model._xyz.detach().cpu().numpy()
    extent = float(np.linalg.norm(_xyz_np.max(0) - _xyz_np.min(0)))
    log(f"  📐 extent={extent:.3f}")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.active_sh = min(model.max_sh, (epoch - 1) // max(sh_every, 1))
        order = list(range(len(cams)))
        random.shuffle(order)
        running = 0.0
        for ci in order:
            stem, v = cams[ci]
            img_path = Path(images_dir) / f"{stem}.png"
            if not img_path.exists():
                img_path = Path(images_dir) / f"{stem}.jpg"
            if not img_path.exists():
                continue
            gt_pil = Image.open(img_path).convert("RGB").resize(
                (v["W"], v["H"]))
            gt = torch.tensor(np.asarray(gt_pil), dtype=torch.float32,
                              device=dev) / 255.0
            gt = gt.permute(2, 0, 1)
            cam = GSCamera(resolution=(v["W"], v["H"]), colmap_id=0,
                           R=v["R"].T, T=v["T"],
                           FoVx=2 * np.arctan(v["W"] / (2 * v["fx"])),
                           FoVy=2 * np.arctan(v["H"] / (2 * v["fy"])),
                           depth_params=None, invdepthmap=None,
                           image=gt_pil, image_name=stem, uid=ci,
                           data_device=dev)
            pkg = render(cam, model, pipe, bg, 1.0)
            m = mask_cache.get(stem)
            if m is not None:
                m_d = m.to(dev)                      # =1 背景(scene职责)
                area = m_d.sum() * 3 + 1e-6
                img_m = pkg["render"] * m_d + gt.detach() * (1.0 - m_d)
                Ll1 = (torch.abs(img_m - gt) * m_d).sum() / area
                loss = (1 - lam_dssim) * Ll1 + \
                    lam_dssim * (1.0 - ssim(img_m, gt))
                # 🚧 mask 内泄漏惩罚：scene-only 渲染在 person 区域本应近黑，
                # 惩罚其亮度防止 scene 高斯长进人物区（composite 重影）。
                if lam_inside > 0:
                    in_area = (1.0 - m_d).sum() * 3 + 1e-6
                    L_in = (pkg["render"] * (1.0 - m_d)).sum() / in_area
                    loss = loss + lam_inside * L_in
            else:
                loss = (1 - lam_dssim) * l1_loss(pkg["render"], gt) + \
                    lam_dssim * (1.0 - ssim(pkg["render"], gt))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            running += float(loss.item())
            vp = pkg.get("viewspace_points", None)
            vf = pkg.get("visibility_filter", None)
            if (vp is not None and vp.grad is not None and vf is not None
                    and len(vf) > 0 and len(vf) <= model.grad_accum.shape[0]):
                with torch.no_grad():
                    g = torch.norm(vp.grad[vf, :2], dim=-1)
                    model.grad_accum[vf] += g
                    model.denom[vf] += 1.0

        avg = running / max(len(cams), 1)
        do_den = (densify_from <= epoch <= densify_until and
                   (epoch - densify_from) % densify_every == 0 and
                   len(model._xyz) < max_gauss)
        msg = f"  epoch {epoch:3d}/{epochs}  loss={avg:.5f}  N={len(model._xyz):,}"
        if do_den:
            kept, added = model.densify(grad_thresh, min_opacity, extent)
            msg += f"  densify: kept={kept:,} +{added:,}"
            # ⚠️ densify 把 model._xyz 等替换成新 nn.Parameter，Adam 仍持
            # 旧引用 → 活参数收不到更新、loss 单调爬升（首轮发散根因）。
            # 必须重建 optimizer 重挂新参数（镜像 finetune_body.py:396）。
            opt = torch.optim.Adam(build_params(), lr=lr)
        if epoch % 5 == 0 or epoch == 1:
            log(msg + f"  ({time.time()-t0:.0f}s)")

    out_ply = out_dir / f"scene_ft_p{pid}.ply"
    model.save(out_ply)
    log(f"💾 saved {out_ply}  N={len(model._xyz):,}")
    (out_dir / "scene_ft_stats.json").write_text(json.dumps({
        "n_gauss": len(model._xyz), "epochs": epochs, "lr": lr,
        "lam_inside": lam_inside, "lam_dssim": lam_dssim,
        "max_gauss": max_gauss, "final_loss": avg,
    }, indent=1))
    log("✅ [08d] done")


if __name__ == "__main__":
    main()
