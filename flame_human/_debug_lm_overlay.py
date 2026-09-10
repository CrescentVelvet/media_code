"""_debug_lm_overlay.py — 05 拟合 landmark vs GT(DECA) 叠加诊断。

绿 = DECA 检测的 468 landmarks（uv_gt，监督目标）
红 = FLAME 拟合后投影位置（uv_pred）
读图判断 26.5px 误差的性质：系统性区域偏差（某部位整片偏）还是随机噪声。
同时按 landmark 区域（脸/眉/眼/鼻/嘴/轮廓）分组统计 RMS，定位误差集中处。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_cameras  # noqa: E402
from align_3dmm import FlameLM, forward, reproj_rms  # noqa: E402

# MediaPipe 468 点区域划分（粗略）
REGIONS = {
    "轮廓(0-16)": list(range(0, 17)),
    "右眉(17-21)": list(range(17, 22)),
    "左眉(22-26)": list(range(22, 27)),
    "鼻(27-35)": list(range(27, 36)),
    "右眼(36-41)": list(range(36, 42)),
    "左眼(42-47)": list(range(42, 48)),
    "嘴外(48-59)": list(range(48, 60)),
    "嘴内(60-67)": list(range(60, 68)),
    "虹膜(468+)": [],
}


def main():
    results_dir = Path(os.environ["RESULTS_DIR"])
    source_dir = os.environ["SOURCE_DIR"]
    pid = os.environ.get("PID", "0")
    dev = torch.device("cuda")

    recon = json.loads((results_dir / "04_recon/face_recon.json").read_text())
    al = json.loads(
        (results_dir / "05_align/head_align.json").read_text())["persons"][pid]
    flame = FlameLM(os.environ["FLAME_MODEL"],
                    os.environ["FLAME_LM468_EMBEDDING"], dev)

    views = read_cameras(source_dir)
    view_by = {v["stem"]: v for v in views}
    stems = al["frames"]

    Rc = torch.tensor(np.stack([view_by[s]["R"] for s in stems]),
                      dtype=torch.float32, device=dev)
    tc = torch.tensor(np.stack([view_by[s]["T"] for s in stems]),
                      dtype=torch.float32, device=dev)
    K = torch.tensor(np.stack([
        [[view_by[s]["fx"], 0, view_by[s]["cx"]],
         [0, view_by[s]["fy"], view_by[s]["cy"]],
         [0, 0, 1.0]] for s in stems]), dtype=torch.float32, device=dev)

    uv_gt = []
    for s in stems:
        rec = recon["frames"][s]["persons"][pid]
        uv_gt.append(np.asarray(rec["lm468"], dtype=np.float32))
    uv_gt = torch.tensor(np.stack(uv_gt), dtype=torch.float32, device=dev)

    idc = torch.tensor(al["id_coeff"], dtype=torch.float32, device=dev)
    exp = torch.tensor(np.array(al["exp_coeff"]), dtype=torch.float32,
                       device=dev)
    lq = torch.tensor(np.array(al["local_q"]), dtype=torch.float32, device=dev)
    lt = torch.tensor(np.array(al["local_t"]), dtype=torch.float32, device=dev)
    gq = torch.tensor(al["global_q"], dtype=torch.float32, device=dev)
    gt = torch.tensor(al["global_t"], dtype=torch.float32, device=dev)
    log_s = torch.tensor(np.log(al["scale"]), dtype=torch.float32, device=dev)

    with torch.no_grad():
        uv = forward(flame.lm3d(idc, exp), log_s, gq, gt, lq, lt, Rc, tc, K)
        err = (uv - uv_gt).norm(dim=-1)                 # (F,468)
    e = err.cpu().numpy()
    print(f"总体 RMS 中值 = {np.median(e):.2f} px  (n={e.shape[1]} 点/帧)")
    print("\n按 landmark 区域（跨帧中值 px）:")
    for name, idx in REGIONS.items():
        if not idx:
            continue
        idx = [i for i in idx if i < e.shape[1]]
        print(f"  {name:14s} {np.median(e[:, idx]):6.2f} px")
    if e.shape[1] > 67:
        print(f"  {'其余(68+)':14s} {np.median(e[:, 68:]):6.2f} px")
    # 逐帧
    fr = np.median(e, axis=1)
    order = np.argsort(-fr)
    print("\n最差 8 帧:", [(int(i), round(float(fr[i]), 1)) for i in order[:8]])

    # 叠加图
    from PIL import Image, ImageDraw
    out_dir = results_dir / "lm_overlay"
    out_dir.mkdir(exist_ok=True)
    for i in (0, 15, 29):
        s = stems[i]
        v = view_by[s]
        p = Path(source_dir) / "images" / f"{s}.jpg"
        if not p.exists():
            p = Path(source_dir) / "images" / f"{s}.png"
        img = Image.open(p).convert("RGB").resize((v["W"], v["H"]))
        d = ImageDraw.Draw(img)
        g = uv_gt[i].cpu().numpy()
        r = uv[i].cpu().numpy()
        for k in range(len(g)):
            d.ellipse([g[k, 0] - 1.5, g[k, 1] - 1.5,
                       g[k, 0] + 1.5, g[k, 1] + 1.5], fill=(0, 255, 0))
            d.ellipse([r[k, 0] - 1.5, r[k, 1] - 1.5,
                       r[k, 0] + 1.5, r[k, 1] + 1.5], fill=(255, 40, 40))
        img.save(out_dir / f"lm_{i:03d}_{s}.png")
        print(f"  saved lm_{i:03d}_{s}.png  (rms={err[i].mean():.1f}px)")
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
