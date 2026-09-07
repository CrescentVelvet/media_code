#!/usr/bin/env python3
"""head 模型训练前后对比：区域内 PSNR + 渲染叠图。

加载初始 head_gs_p{PID}.ply 与训练后 iteration ply，对有 head mask 的帧
逐帧 warp（与 build_region_masks 同公式）渲染，算 head 区域内 PSNR，
并输出 [gt | 初始 | 训练后] 对比图供人工复检。
"""
import os, sys, json
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as _Rot

os.environ.setdefault("OMP_NUM_THREADS", "8")

RESULTS = os.environ.get("RESULTS_DIR", "/mnt/d/output/vggt_human_ms")
PID = os.environ.get("PID", "00")
HEAD_DIR = f"{RESULTS}/03e_head_3dmm"
TRAIN_DIR = os.environ.get("TRAIN_DIR", f"{RESULTS}/03i_head_p{PID}")
ITERS = os.environ.get("ITERS", "4000")
N_VIS = int(os.environ.get("N_VIS", "6"))
OUT_DIR = f"{TRAIN_DIR}/vis_compare"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/repos/gaussian-splatting"))


def qvec2rotmat_np(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def load_gs(path):
    """读 ply 的高斯参数（不依赖 GaussianModel 完整初始化）。"""
    from plyfile import PlyData
    ply = PlyData.read(path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
    op = v["opacity"].astype(np.float32)[:, None]
    sc = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32)
    return dict(xyz=torch.tensor(xyz), rot=torch.tensor(rot), dc=torch.tensor(dc),
                op=torch.tensor(op), sc=torch.tensor(sc))

class MiniGS:
    """只包装渲染需要的属性。⚠️ PLY 存 raw 值：scale 要 exp、opacity 要 sigmoid。"""
    def __init__(self, d, recolor=None):
        self._xyz = d["xyz"].cuda()
        self._rotation = d["rot"].cuda()
        self._scaling = torch.exp(d["sc"].cuda())          # log → scale
        self._opacity = torch.sigmoid(d["op"].cuda())      # raw → opacity
        if recolor is None:
            self._features_dc = d["dc"].cuda()
        else:
            c = torch.full_like(d["dc"], recolor)   # 染白 +1.772 / 染黑 -1.772
            self._features_dc = c
        self._features_rest = torch.zeros(len(self._xyz), 15, 3, device="cuda")
        self.max_radii2D = torch.empty(0)
        self.active_sh_degree = 0
        self.pretrained_exposures = None

    @property
    def get_xyz(self): return self._xyz
    @property
    def get_rotation(self): return self._rotation
    @property
    def get_scaling(self): return self._scaling
    @property
    def get_opacity(self): return self._opacity
    @property
    def get_features(self):
        return torch.cat([self._features_dc.unsqueeze(1), self._features_rest], dim=1)


def warp_gs(g, A, b):
    """与 build_region_masks.py 完全一致的刚性 warp（就地修改再返回原值）。"""
    xyz0 = g._xyz.detach().clone()
    rot0 = g._rotation.detach().clone()
    g._xyz.data = (xyz0 @ torch.tensor(A.T, dtype=torch.float32, device="cuda")
                   + torch.tensor(b, dtype=torch.float32, device="cuda"))
    qA = _Rot.from_matrix(A)
    q_old = _Rot.from_quat(rot0.cpu().numpy()[:, [1, 2, 3, 0]])
    q_new = (qA * q_old).as_quat()
    rot_new = np.stack([q_new[:, 3], q_new[:, 0], q_new[:, 1], q_new[:, 2]], axis=1)
    g._rotation.data = torch.tensor(rot_new, dtype=torch.float32, device="cuda")
    return xyz0, rot0


def restore_gs(g, xyz0, rot0):
    g._xyz.data = xyz0
    g._rotation.data = rot0


def psnr_region(img, gt, mask, eps=1e-8):
    m = mask[None] if mask.ndim == 2 else mask
    mse = (((img - gt) ** 2) * m).sum() / (m.sum() * 3 + eps)
    return float(-10.0 * np.log10(mse.item() + eps))


def main():
    from gaussian_renderer import render
    from scene import Scene, GaussianModel
    from fit_head_3dmm import parse_colmap_cameras, build_proj_matrices
    from train_face_finetune import RegionMaskData
    from utils.camera_utils import cameraList_from_camInfos
    from argparse import Namespace

    fit = json.loads(open(f"{HEAD_DIR}/head_fit.json").read())
    fr = fit["persons"][PID]
    R_ref = np.asarray(fr["R_ref"]); t_ref = np.asarray(fr["t_ref"])
    pf = fr["per_frame"]

    views = parse_colmap_cameras(f"{RESULTS}/03b_source_ba")
    views = {v["stem"]: v for v in views}

    # 初始与训练后模型
    d_init = load_gs(f"{HEAD_DIR}/head_gs_p{PID}.ply")
    d_trained = load_gs(f"{TRAIN_DIR}/point_cloud/iteration_{ITERS}/point_cloud.ply")
    print(f"init: {len(d_init['xyz'])} gaussians, trained: {len(d_trained['xyz'])} gaussians")
    g_init = MiniGS(d_init)
    g_trained = MiniGS(d_trained)
    # 渲染管线（CPU pipe）
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                     antialiasing=False, debug=False)
    bg = torch.zeros(3, device="cuda")

    # 相机（与 build_region_masks.py 完全一致的构造）
    from scene.cameras import Camera

    def make_camera(v, W, H):
        import math
        FoVx = 2 * math.atan(W / (2 * v["fx"]))
        FoVy = 2 * math.atan(H / (2 * v["fy"]))
        return Camera(resolution=(W, H), colmap_id=0,
                      R=qvec2rotmat_np(v["qvec"]).T, T=np.asarray(v["tvec"]),
                      FoVx=FoVx, FoVy=FoVy, depth_params=None,
                      image=Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)),
                      invdepthmap=None, image_name=v["stem"], uid=0,
                      data_device="cuda")

    os.makedirs(OUT_DIR, exist_ok=True)
    stems = sorted(pf.keys())
    step = max(1, len(stems) // N_VIS)
    picked = stems[::step][:N_VIS]

    psnrs_init, psnrs_trained = [], []
    tiles = []
    from pathlib import Path
    for stem in picked:
        mpath = f"{RESULTS}/03i_region_masks/{stem}.p{PID}.head.png"
        if not os.path.isfile(mpath): continue
        mask = np.asarray(Image.open(mpath).convert("L"), dtype=np.float32) / 255.0
        if mask.max() < 0.1: continue
        # gt 帧
        gt_files = list(Path(f"{RESULTS}/03b_source_ba/images").glob(f"{stem}.*"))
        if not gt_files: continue
        gt = np.asarray(Image.open(gt_files[0]).convert("RGB"), dtype=np.float32) / 255.0
        H, W = gt.shape[:2]

        R_f = np.asarray(pf[stem]["R"]).reshape(3,3)
        t_f = np.asarray(pf[stem]["t"]).reshape(3)
        A = R_f @ R_ref.T
        b = t_f - A @ t_ref

        v = views[stem]
        W, H = int(v["W"]), int(v["H"])
        cam = make_camera(v, W, H)

        outs = []
        for tag, g in [("init", g_init), ("trained", g_trained)]:
            x0, r0 = warp_gs(g, A, b)
            with torch.no_grad():
                img = render(cam, g, pipe, bg)["render"].clamp(0, 1)
            restore_gs(g, x0, r0)
            img_np = img.permute(1, 2, 0).cpu().numpy()
            m_t = torch.tensor(mask, device="cuda")[None]
            p = psnr_region(img, torch.tensor(gt, device="cuda").permute(2, 0, 1), m_t)
            if tag == "init": psnrs_init.append(p)
            else: psnrs_trained.append(p)
            outs.append((tag, img_np, p))

        # 叠图行：gt | init overlay | trained overlay
        ov_init = gt * 0.4 + outs[0][1] * 0.6
        ov_trained = gt * 0.4 + outs[1][1] * 0.6
        row = np.concatenate([gt, ov_init, ov_trained], axis=1)
        tiles.append((stem, row, outs[0][2], outs[1][2]))

    if not tiles:
        print("❌ 没有可评估帧"); return
    psnrs_init = np.array(psnrs_init); psnrs_trained = np.array(psnrs_trained)
    print(f"\n=== head 区域内 PSNR（{len(tiles)} 帧）===")
    print(f"  init   : med={np.median(psnrs_init):.2f}  mean={psnrs_init.mean():.2f}")
    print(f"  trained: med={np.median(psnrs_trained):.2f}  mean={psnrs_trained.mean():.2f}")
    print(f"  delta  : {psnrs_trained.mean()-psnrs_init.mean():+.2f} dB")

    # 拼 mosaic
    th = min(t.shape[0] for _, t, _, _ in tiles)
    tiles_r = [(s, t[:th], a, b) for s, t, a, b in tiles]
    mosaic = np.concatenate([t for _, t, _, _ in tiles_r], axis=0)
    # 标题条
    bar = np.zeros((36, mosaic.shape[1], 3), dtype=np.float32)
    from PIL import ImageDraw
    img = Image.fromarray((mosaic * 255).astype(np.uint8))
    total = Image.new("RGB", (mosaic.shape[1], mosaic.shape[0] + 36), (10, 10, 10))
    total.paste(img, (0, 36))
    dr = ImageDraw.Draw(total)
    dr.text((10, 8), f"p{PID} head  [GT | init overlay | trained(4k iters) overlay]  PSNR init->trained: {psnrs_init.mean():.1f} -> {psnrs_trained.mean():.1f} dB", fill=(255, 255, 255))
    out = f"{OUT_DIR}/head_compare_p{PID}.png"
    total.save(out)
    print(f"\n🖼️ {out}")
    for s, _, a, b in tiles_r:
        print(f"  {s}: {a:.2f} -> {b:.2f} dB")


if __name__ == "__main__":
    main()
