#!/usr/bin/env python3
"""18_flame_fit.py — F2 上半段：三视角联合拟合共享 FLAME 粗脸 + 审计门。

对应 da3_human/EXPERIMENT_PLAN.md F2：
  「三张 source 联合拟合一个共享身份/形状的 FLAME 粗脸，使用置信筛选的鼻尖/眼角等
   对应；FLAME 先单独过三视角轮廓/关键点/鼻部深度顺序审计；失败就停在此」。

本脚本只做「拟合 + 审计」，不绑高斯（绑定在 F2 下半段，须本审计通过后才做）。
在 rosemary_face_v3（静态合成包，GT 相机 + GT 深度 EXR）上做：
相机直接用 prior_camera.json（GT 世界系，**不需要 Sim(3)**）；GT 深度做鼻部深度审计。

对应点（置信筛选，轮廓点不进硬损失）：
  mediapipe 468 ↔ iBUG51（DECA landmark_embedding 的 51 序，即 iBUG68 去掉轮廓 0-16）
    MP 4(鼻尖)→iBUG30→51#13 | MP 6(鼻梁)→iBUG27→#10
    MP 33(右眼外)→iBUG36→#19 | MP 133(右眼内)→iBUG39→#22
    MP 362(左眼内)→iBUG42→#25 | MP 263(左眼外)→iBUG45→#28
    MP 61(右嘴角)→iBUG48→#31 | MP 291(左嘴角)→iBUG54→#37
    MP 105(右眉中)→iBUG20→#3 | MP 334(左眉中)→iBUG25→#8

审计门（全过才算 FLAME 可作为约束先验）：
  G1 关键点重投影 RMSE < 3px（1024²）
  G2 鼻部深度：鼻尖 FLAME 深度 vs GT 深度 |Δz| < 15mm，且鼻尖 z < 眼角 z（顺序正确）
  G3 轮廓 IoU（FLAME 顶点 5px splat 投影 vs 脸 mask）> 0.6

用法（flame_human env）：
  python 18_flame_fit.py --scene <07式source场景> --landmarks <json> \
      --pkg <human_fbx_gt 包> --out <目录>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

DECA = Path.home() / "repos" / "DECA"
sys.path.insert(0, str(DECA))
sys.path.insert(0, str(DECA / "decalib"))

# MP index -> iBUG68 index（landmarks2d = 17动态轮廓 + 51静态，序号即 iBUG68）
PAIRS = {4: 30, 6: 27, 33: 36, 133: 39, 362: 42, 263: 45,
         61: 48, 291: 54, 105: 20, 334: 25}
MP_FOR_PAIR = {"nose_tip": 4, "nose_bridge": 6, "eye_r_out": 33,
               "eye_r_in": 133, "eye_l_in": 362, "eye_l_out": 263,
               "mouth_r": 61, "mouth_l": 291, "brow_r": 105, "brow_l": 334}


def load_depth_npy(audit_prep, fid):
    p = Path(audit_prep) / f"depth_{fid:06d}.npy"
    return np.load(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="07 式 source 场景")
    ap.add_argument("--pkg", required=True, help="human_fbx_gt 包（GT 深度来源）")
    ap.add_argument("--landmarks", required=True,
                    help="FaceMesh 地标 json（18 的探针产物，1024² 原生）")
    ap.add_argument("--depth-cache", required=True, help="EXR→npy 预转换目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from decalib.models.FLAME import FLAME, Struct
    cfg = Struct(flame_model_path=str(DECA / "data" / "generic_model.pkl"),
                 flame_lmk_embedding_path=str(DECA / "data" / "landmark_embedding.npy"),
                 n_shape=30, n_exp=0)
    flame = FLAME(cfg).to(dev)

    # ---------- 相机（GT 世界系，直接用）----------
    sc = json.loads((Path(args.scene) / "prior_camera.json").read_text())
    extr = {e["camera_id"]: np.asarray(e["matrix"], np.float64)
            for e in sc["extrinsics"]}
    intr = {e["camera_id"]: np.asarray(e["matrix"], np.float64)
            for e in sc["intrinsics"]}
    meta = json.loads((Path(args.scene) / "meta.json").read_text())
    view_ids = meta["view_ids"]          # 包内 id，如 [20,30,38]
    stems = list(extr.keys())            # 000_000020 等
    lm_all = json.loads(Path(args.landmarks).read_text())

    views = []
    for stem, fid in zip(stems, view_ids):
        lmv = lm_all["native"].get(stem)
        if lmv is None:
            raise SystemExit(f"❌ 缺地标: {stem}")
        img = np.asarray(Image.open(
            Path(args.pkg) / "rgb" / f"{fid + 1:06d}.png").convert("RGB"))
        dep = load_depth_npy(args.depth_cache, fid + 1)
        views.append(dict(stem=stem, fid=fid, lm=lmv["pts"],
                          c2w=extr[stem], K=intr[stem], img=img, depth=dep))
    H, W = views[0]["img"].shape[:2]

    # ---------- 待优化参数（共享身份/形状 + 共享头姿；静态包）----------
    rot = torch.zeros(3, device=dev, requires_grad=True)
    trans = torch.tensor([0.165, 0.370, 1.30], device=dev, requires_grad=True)
    log_s = torch.tensor([np.log(1.0)], device=dev, requires_grad=True)
    beta = torch.zeros(1, 30, device=dev, requires_grad=True)

    def verts_lmks():
        v, lmk, _ = flame(shape_params=beta,
                          expression_params=torch.zeros(1, 0, device=dev),
                          pose_params=torch.cat([rot[None],
                                                 torch.zeros(1, 3, device=dev)],
                                                1))
        s = torch.exp(log_s)[0]
        X = v[0] * s + trans
        L = lmk[0] * s + trans
        return X, L

    def project(X, vw):
        """OpenCV 投影：w2c = inv(c2w)。"""
        c2w = torch.tensor(vw["c2w"], dtype=torch.float32, device=dev)
        Rc = c2w[:3, :3]
        t = c2w[:3, 3]
        Xc = (X - t) @ Rc
        K = torch.tensor(vw["K"], dtype=torch.float32, device=dev)
        z = Xc[:, 2].clamp_min(1e-6)
        px = torch.stack([K[0, 0] * Xc[:, 0] / z + K[0, 2],
                          K[1, 1] * Xc[:, 1] / z + K[1, 2]], 1)
        return px, z

    def loss_all():
        _, L = verts_lmks()
        tot = 0.0
        for vw in views:
            px, _ = project(L, vw)
            for name, mp_i in MP_FOR_PAIR.items():
                i51 = PAIRS[mp_i]
                gt = torch.tensor(vw["lm"][name], dtype=torch.float32,
                                  device=dev)
                tot = tot + ((px[i51] - gt) ** 2).sum()
        return tot / (len(views) * len(PAIRS)) + 1e-3 * (beta ** 2).sum()

    # ---------- 初值搜索：yaw 网格（FLAME 规范系 +y 上、脸朝 +z）----------
    best = None
    for yaw_deg in range(0, 360, 30):
        th = float(np.radians(yaw_deg))
        # 绕 +y 的旋转作为 global orient 初值
        with torch.no_grad():
            rot.data = torch.tensor([0.0, th, 0.0], dtype=torch.float32,
                                    device=dev)
        l = loss_all()
        if best is None or l < best[0]:
            best = (float(l), yaw_deg)
    with torch.no_grad():
        rot.data = torch.tensor([0.0, float(np.radians(best[1])), 0.0],
                                dtype=torch.float32, device=dev)
    print(f"初值 yaw={best[1]}° loss0={best[0]:.4f}")

    opt = torch.optim.Adam([{"params": [rot, trans, log_s], "lr": 1e-2},
                            {"params": [beta], "lr": 1e-2}])
    hist = []
    for it in range(args.iters):
        opt.zero_grad()
        l = loss_all()
        l.backward()
        opt.step()
        if it in (50, 100, 200, 300, args.iters - 1):
            hist.append((it, float(l)))
            print(f"  it{it:4d} loss {float(l):.5f}")

    # ---------- 审计 ----------
    X, L = verts_lmks()
    X_np = X.detach().cpu().numpy()
    L_np = L.detach().cpu().numpy()
    gates = {}
    per_view = []
    for vw in views:
        px, z = project(L, vw)
        px_np = px.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        errs = []
        for name, mp_i in MP_FOR_PAIR.items():
            errs = np.linalg.norm(px_np[PAIRS[mp_i]] - np.array(vw["lm"][name]))
            per_view.append(dict(view=vw["stem"], lm=name,
                                 err_px=float(errs)))
        # 鼻部深度审计：FLAME 鼻尖/眼角 深度 vs GT EXR
        nose_i, eL_i, eR_i = PAIRS[4], PAIRS[263], PAIRS[33]
        npx = np.clip(np.round(px_np[nose_i]).astype(int), [0, 0], [W - 1, H - 1])
        z_gt = float(vw["depth"][npx[1], npx[0]])
        z_flame_nose = float(z_np[nose_i])
        order_ok = bool(z_np[nose_i] < z_np[eL_i] and z_np[nose_i] < z_np[eR_i])
        gates.setdefault("nose_dz_mm", []).append(
            abs(z_flame_nose - z_gt) * 1000)
        gates.setdefault("nose_order_ok", []).append(order_ok)
        # 轮廓 IoU：顶点 5px splat vs FaceMesh 脸凸包 mask
        pxv, _ = project(X, vw)
        pxv = pxv.detach().cpu().numpy()
        splat = np.zeros((H, W), np.uint8)
        pv = np.round(pxv).astype(int)
        ok = (pv[:, 0] >= 2) & (pv[:, 0] < W - 2) & (pv[:, 1] >= 2) & \
             (pv[:, 1] < H - 2)
        pv = pv[ok]
        for x, y in pv:
            splat[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = 1
        import cv2
        splat = cv2.morphologyEx(splat, cv2.MORPH_CLOSE,
                                 np.ones((9, 9), np.uint8))
        fmask = np.zeros((H, W), np.uint8)
        hull_pts = np.array([vw["lm"][k] for k in
                             ("chin", "forehead", "eye_r_out", "eye_l_out")],
                            np.float32)
        # 简化：用 bbox 近似 GT 脸区（审计级足够）
        ys = [vw["lm"][k][1] for k in vw["lm"]]
        xs = [vw["lm"][k][0] for k in vw["lm"]]
        fmask[int(min(ys)):int(max(ys)) + 1, int(min(xs)):int(max(xs)) + 1] = 1
        inter = float((splat & fmask).sum())
        union = float(((splat | fmask)).sum())
        gates.setdefault("iou_bbox_approx", []).append(inter / max(union, 1))
        # 叠加图
        ov = vw["img"].copy()
        d = ImageDraw.Draw(Image.fromarray(ov))
        im = Image.fromarray(ov)
        dr = ImageDraw.Draw(im)
        for name, mp_i in MP_FOR_PAIR.items():
            gx, gy = vw["lm"][name]
            fx, fy = px_np[PAIRS[mp_i]]
            dr.ellipse([gx - 3, gy - 3, gx + 3, gy + 3], outline=(0, 255, 0),
                       width=2)
            dr.ellipse([fx - 2, fy - 2, fx + 2, fy + 2], outline=(255, 40, 40),
                       width=2)
        cnt = splat.copy() * 255
        edge = cv2.Canny(cnt, 50, 150) > 0
        arr = np.asarray(im).copy()
        arr[edge] = [255, 200, 0]
        Image.fromarray(arr).save(out / f"flame_fit_{vw['stem']}.png")
    errs_all = [p["err_px"] for p in per_view]
    rmse = float(np.sqrt(np.mean(np.square(errs_all))))
    gate = dict(
        G1_reproj_rmse_px=rmse,
        G1_pass=rmse < 3.0,
        G2_nose_dz_mm_max=float(np.max(gates["nose_dz_mm"])),
        G2_nose_dz_mm=float(np.mean(gates["nose_dz_mm"])),
        G2_pass=(max(gates["nose_dz_mm"]) < 15.0 and
                 all(gates["nose_order_ok"])),
        G2_nose_order_ok=all(gates["nose_order_ok"]),
        G3_iou_bbox_approx=float(np.mean(gates["iou_bbox_approx"])),
        G3_pass=float(np.mean(gates["iou_bbox_approx"])) > 0.6,
    )
    result = dict(args=vars(args), gates=gate, per_view=per_view,
                  fit=dict(rot=[float(x) for x in rot.detach().cpu()],
                           trans=[float(x) for x in trans.detach().cpu()],
                           scale=float(torch.exp(log_s).detach().cpu()),
                           beta=beta.detach().cpu().numpy().tolist()),
                  hist=hist)
    with open(out / "flame_fit.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(json.dumps(gate, indent=1, ensure_ascii=False))
    print(f"🎉 {out}  G1={'✅' if gate['G1_pass'] else '❌'} "
          f"G2={'✅' if gate['G2_pass'] else '❌'} "
          f"G3={'✅' if gate['G3_pass'] else '❌'}")


if __name__ == "__main__":
    main()
