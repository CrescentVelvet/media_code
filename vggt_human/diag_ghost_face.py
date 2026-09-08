#!/usr/bin/env python3
"""diag_ghost_face.py — 诊断合成图中某人脸部重影/模糊的来源。

MODE=stats : 不需要 GPU。打印每人 per_frame 数 / warp 幅度 / head mask 质心
             （质心 y 最小 = 站得最高 = 站立者），以及 N_VIS=8 的选帧列表。
MODE=frame : STEM=xxx PID=xx 需要 GPU。按 render_composite 同一路径逐层分解，
             打印该人 head 内部区各层 alpha 统计，并存分解裁剪图：
             [GT | comp | head-only | under(scene+body) | ah | own-body-a | trans_other]
"""
import os, sys, json
import numpy as np
from PIL import Image

os.environ.setdefault("OMP_NUM_THREADS", "8")
RESULTS = os.environ.get("RESULTS_DIR", "/mnt/d/output/vggt_human_ms")
MODE = os.environ.get("MODE", "stats")
STEM = os.environ.get("STEM", "")
PID = os.environ.get("PID", "01")
HEAD_ITERS = os.environ.get("HEAD_ITERS", "10000")
BODY_ITERS = os.environ.get("BODY_ITERS", "30000")
SCENE_ITERS = os.environ.get("SCENE_ITERS", "4000")
CLAMP_DILATE = int(os.environ.get("CLAMP_DILATE", "10"))
CLAMP_FEATHER = int(os.environ.get("CLAMP_FEATHER", "20"))
N_VIS = int(os.environ.get("N_VIS", "8"))
PIDS = ["00", "01", "02"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def soft(mm):
    import cv2
    k = np.ones((CLAMP_DILATE * 2 + 1, CLAMP_DILATE * 2 + 1), np.uint8)
    hard = cv2.dilate((mm > 0.3).astype(np.uint8), k).astype(np.float32)
    if CLAMP_FEATHER > 0:
        ks = CLAMP_FEATHER * 2 + 1
        hard = np.minimum(cv2.GaussianBlur(hard, (ks, ks), 0), hard)
    return hard


def stats():
    fit = json.loads(open(f"{RESULTS}/03e_head_3dmm/head_fit.json").read())
    print("per_frame 数 / warp 幅度（相对 ref 位姿）:")
    for pid, fr in fit["persons"].items():
        pf = fr["per_frame"]
        R_ref = np.asarray(fr["R_ref"]); t_ref = np.asarray(fr["t_ref"])
        tn, an = [], []
        for s, p in pf.items():
            R = np.asarray(p["R"]).reshape(3, 3); t = np.asarray(p["t"]).reshape(3)
            A = R @ R_ref.T; b = t - A @ t_ref
            tn.append(np.linalg.norm(b))
            an.append(np.degrees(2 * np.arccos(np.clip((np.trace(A) - 1) / 2, -1, 1))))
        tn, an = np.array(tn), np.array(an)
        print(f"  p{pid}: n={len(pf)}  |t| med={np.median(tn):.3f} p90={np.percentile(tn, 90):.3f}"
              f"  ang med={np.median(an):.1f} deg p90={np.percentile(an, 90):.1f} deg")
    k0 = list(fit["persons"][PIDS[0]]["per_frame"].values())[0]
    print("per_frame keys:", sorted(k0.keys()))

    from face_center_3d import parse_colmap_cameras
    views = {v["stem"]: v for v in parse_colmap_cameras(f"{RESULTS}/03b_source_ba")}
    stems_all = sorted(views.keys())
    step = max(1, len(stems_all) // N_VIS)
    picked = stems_all[::step][:N_VIS]
    print(f"picked stems (N_VIS={N_VIS}): {picked}")

    print("head mask 质心（cy 最小 = 站立者）@ 中间帧:")
    s0 = stems_all[len(stems_all) // 2]
    for pid in PIDS:
        hp = f"{RESULTS}/03i_region_masks/{s0}.p{pid}.head.png"
        if not os.path.isfile(hp):
            print(f"  p{pid}: no mask"); continue
        m = np.asarray(Image.open(hp).convert("L"), np.float32) / 255
        ys, xs = np.where(m > 0.5)
        if len(ys) == 0:
            print(f"  p{pid}: empty"); continue
        print(f"  p{pid}: cy={ys.mean():.0f} cx={xs.mean():.0f} area={len(ys)}  @{s0}")


def frame():
    import torch, cv2
    import render_composite as RC
    from face_center_3d import parse_colmap_cameras
    from pathlib import Path

    views = {v["stem"]: v for v in parse_colmap_cameras(f"{RESULTS}/03b_source_ba")}
    v = views[STEM]; W, H = int(v["W"]), int(v["H"])
    gt = np.asarray(Image.open(list(Path(f"{RESULTS}/03b_source_ba/images").glob(f"{STEM}.*"))[0])
                    .convert("RGB"), np.float32) / 255
    pipe = RC.Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                        antialiasing=False, debug=False)
    bg = torch.zeros(3, device="cuda")
    cam = RC.make_camera(v, W, H)
    g_scene = RC.MiniGS(RC.load_gs(f"{RESULTS}/03i_scene/point_cloud/iteration_{SCENE_ITERS}/point_cloud.ply"))
    heads = {p: RC.MiniGS(RC.load_gs(f"{RESULTS}/03i_head_p{p}/point_cloud/iteration_{HEAD_ITERS}/point_cloud.ply")) for p in PIDS}
    bodies = {p: RC.MiniGS(RC.load_gs(f"{RESULTS}/03i_body_p{p}/point_cloud/iteration_{BODY_ITERS}/point_cloud.ply")) for p in PIDS}
    fit = json.loads(open(f"{RESULTS}/03e_head_3dmm/head_fit.json").read())

    body_masks, head_masks, body_raw, head_raw = {}, {}, {}, {}
    for p in PIDS:
        bp = f"{RESULTS}/03i_region_masks/{STEM}.p{p}.body.png"
        hp = f"{RESULTS}/03i_region_masks/{STEM}.p{p}.head.png"
        if os.path.isfile(bp):
            m = np.asarray(Image.open(bp).convert("L"), np.float32) / 255
            body_raw[p] = m; body_masks[p] = torch.tensor(soft(m), device="cuda")[None]
        if os.path.isfile(hp):
            m = np.asarray(Image.open(hp).convert("L"), np.float32) / 255
            head_raw[p] = m; head_masks[p] = torch.tensor(soft(m), device="cuda")[None]

    c_scene, a_scene = RC.render_rgb_alpha(g_scene, cam, pipe, bg)
    comp = c_scene.clone(); acc = torch.zeros(1, H, W, device="cuda")
    body_alpha = {}
    for p in PIDS:
        cb, ab = RC.render_rgb_alpha(bodies[p], cam, pipe, bg)
        if p in body_masks:
            ab = ab * body_masks[p]
        body_alpha[p] = ab
        comp = comp * (1 - ab * (1 - acc)) + cb * (ab * (1 - acc))
        acc = torch.clamp(acc + ab * (1 - acc), 0, 1)
    under_head = comp.clone()   # head 层之前的底层（scene+body）

    head_info = {}
    for p in PIDS:
        fr = fit["persons"].get(p)
        pf = fr["per_frame"].get(STEM) if fr else None
        if pf is None:
            head_info[p] = None; continue
        R_ref = np.asarray(fr["R_ref"]); t_ref = np.asarray(fr["t_ref"])
        R_f = np.asarray(pf["R"]).reshape(3, 3); t_f = np.asarray(pf["t"]).reshape(3)
        A = R_f @ R_ref.T; b = t_f - A @ t_ref
        g = heads[p]; x0, r0 = RC.warp_gs(g, A, b)
        ch, ah = RC.render_rgb_alpha(g, cam, pipe, bg)
        RC.restore_gs(g, x0, r0)
        ah_raw = ah
        if p in head_masks:
            ah = ah * head_masks[p]
        trans_other = torch.ones(1, H, W, device="cuda")
        for j in PIDS:
            if j != p and j in body_alpha:
                trans_other = trans_other * (1 - body_alpha[j])
        a_h = ah * trans_other
        head_info[p] = (ch, ah_raw, ah, a_h, trans_other)
        comp = comp * (1 - a_h) + ch * a_h

    p = PID
    info = head_info[p]; hm = head_raw.get(p)
    if info is None or hm is None:
        print(f"no head/mask for p{p} @{STEM}"); return
    ch, ah_raw, ah, a_h, trans_other = info
    interior = (soft(hm) >= 0.999)
    ni = int(interior.sum())
    gt_t = torch.tensor(gt, device="cuda").permute(2, 0, 1)
    int_t = torch.tensor(interior, device="cuda")[None]

    def st(name, t):
        tt = t[0].cpu().numpy() if torch.is_tensor(t) else t
        print(f"  {name}: mean={tt[interior].mean():.3f} p5={np.percentile(tt[interior], 5):.3f} min={tt[interior].min():.3f}")

    print(f"=== {STEM} p{p} head interior px={ni} ===")
    st("ah(clamped)", ah)
    st("ah(raw render)", ah_raw)
    st("own body alpha", body_alpha[p])
    st("trans_other", trans_other)
    st("a_h effective", a_h)
    for j in PIDS:
        if j != p and head_info[j]:
            st(f"head{j} alpha overlap", head_info[j][2])
    mse = (((comp - gt_t) ** 2) * int_t).sum() / (ni * 3 + 1e-8)
    print(f"  comp vs GT interior PSNR={-10 * np.log10(mse.item() + 1e-8):.2f}")
    vis = (ah_raw[0].cpu().numpy() > 0.9) & interior
    if vis.sum() > 100:
        mseh = (((ch - gt_t) ** 2) * torch.tensor(vis, device="cuda")[None]).sum() / (vis.sum() * 3 + 1e-8)
        print(f"  head-only vs GT (ah>0.9, n={int(vis.sum())}) PSNR={-10 * np.log10(mseh.item() + 1e-8):.2f}")
    ahn = ah[0].cpu().numpy()
    print(f"  interior ah<0.9 frac={(ahn[interior] < 0.9).mean():.3f}  ah<0.5 frac={(ahn[interior] < 0.5).mean():.3f}")

    ys, xs = np.where(hm > 0.5)
    pad = 40
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)

    def g3(t):
        a = t if isinstance(t, np.ndarray) else t[0].cpu().numpy()
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        return np.clip(a, 0, 1)[y0:y1, x0:x1]

    tiles = [gt[y0:y1, x0:x1],
             comp.permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1],
             ch.permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1],
             under_head.permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1],
             g3(ah), g3(body_alpha[p]), g3(trans_other)]
    row = np.concatenate(tiles, axis=1)
    out = f"{RESULTS}/03i_composite_vis/ghost_{STEM}_p{p}.png"
    Image.fromarray((np.clip(row, 0, 1) * 255).astype(np.uint8)).save(out)
    print(f"  saved {out}")
    print("  tiles: [GT | comp | head-only | under(scene+body) | ah | own-body-a | trans_other]")


def scan():
    """逐帧扫描某人 head：head-only PSNR / alpha 洞比例 / warp 幅度 的相关性。"""
    import torch
    import render_composite as RC
    from face_center_3d import parse_colmap_cameras
    from pathlib import Path

    views = {v["stem"]: v for v in parse_colmap_cameras(f"{RESULTS}/03b_source_ba")}
    fit = json.loads(open(f"{RESULTS}/03e_head_3dmm/head_fit.json").read())
    fr = fit["persons"][PID]
    R_ref = np.asarray(fr["R_ref"]); t_ref = np.asarray(fr["t_ref"])
    pipe = RC.Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                        antialiasing=False, debug=False)
    bg = torch.zeros(3, device="cuda")
    g = RC.MiniGS(RC.load_gs(f"{RESULTS}/03i_head_p{PID}/point_cloud/iteration_{HEAD_ITERS}/point_cloud.ply"))
    gb = RC.MiniGS(RC.load_gs(f"{RESULTS}/03i_body_p{PID}/point_cloud/iteration_{BODY_ITERS}/point_cloud.ply"))
    stems_all = sorted(views.keys())
    step = max(1, len(stems_all) // N_VIS)
    picked = stems_all[::step][:N_VIS]
    print(f"stem            warp_ang  ah_hole%  headOnlyPSNR  bodyGhostPSNR")
    for stem in picked:
        pf = fr["per_frame"].get(stem)
        hp = f"{RESULTS}/03i_region_masks/{stem}.p{PID}.head.png"
        if pf is None or not os.path.isfile(hp):
            continue
        v = views[stem]; W, H = int(v["W"]), int(v["H"])
        gt = np.asarray(Image.open(list(Path(f"{RESULTS}/03b_source_ba/images").glob(f"{stem}.*"))[0])
                        .convert("RGB"), np.float32) / 255
        gt_t = torch.tensor(gt, device="cuda").permute(2, 0, 1)
        cam = RC.make_camera(v, W, H)
        m = np.asarray(Image.open(hp).convert("L"), np.float32) / 255
        interior = soft(m) >= 0.999
        int_t = torch.tensor(interior, device="cuda")[None]
        R = np.asarray(pf["R"]).reshape(3, 3); t = np.asarray(pf["t"]).reshape(3)
        A = R @ R_ref.T; b = t - A @ t_ref
        ang = np.degrees(2 * np.arccos(np.clip((np.trace(A) - 1) / 2, -1, 1)))
        x0, r0 = RC.warp_gs(g, A, b)
        ch, ah = RC.render_rgb_alpha(g, cam, pipe, bg)
        RC.restore_gs(g, x0, r0)
        cb, ab = RC.render_rgb_alpha(gb, cam, pipe, bg)
        ahn = ah[0].cpu().numpy()
        hole = (ahn < 0.9) & interior
        vis = (ahn > 0.9) & interior
        mse = (((ch - gt_t) ** 2) * torch.tensor(vis, device="cuda")[None]).sum() / (max(vis.sum(), 1) * 3 + 1e-8)
        mseb = (((cb - gt_t) ** 2) * torch.tensor(hole, device="cuda")[None]).sum() / (max(hole.sum(), 1) * 3 + 1e-8)
        print(f"{stem}  {ang:6.1f}deg  {hole.mean()*100:6.2f}%  "
              f"{-10*np.log10(mse.item()+1e-8):8.2f}    {-10*np.log10(mseb.item()+1e-8):8.2f}")


if __name__ == "__main__":
    if MODE == "stats":
        stats()
    elif MODE == "scan":
        scan()
    else:
        frame()
