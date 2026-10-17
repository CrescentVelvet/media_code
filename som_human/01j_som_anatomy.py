#!/usr/bin/env python3
"""01j_som_anatomy.py — 解剖 SOM 运动基：判断运动分解是否发生（SOM 是否退化）。

判读口径（EXPERIMENTS.md E-01 基线：model_p07 退化水平）：
  基平移 ~0.009 / 基旋转 ~0.26° / 系数场熵 2.10（近均匀） → SOM ≈ 恒等，退化
  若基幅度显著大于该水平且系数场出现分组 → 运动分解生效

用法（WSL 内，vggt_human env）：
    python 01j_som_anatomy.py --model <model_dir> [--save-csv out.csv]
"""
import argparse
import os
import sys

import numpy as np

# train_som.py 同目录（som_motion 所在）+ 官方仓（som_motion 依赖 utils）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_DG_DIR = os.environ.get("DG_DIR", os.path.expanduser("~/repos/Deformable-3D-Gaussians"))
sys.path.insert(0, _DG_DIR)


def load_ckpt(model_dir):
    sd = os.path.join(model_dir, "som_motion")
    if not os.path.isdir(sd):
        sys.exit("no som_motion dir: {}".format(sd))
    iters = [d for d in os.listdir(sd) if d.startswith("iteration_")]
    it = max(int(d.split("_")[1]) for d in iters)
    import torch
    ck = torch.load(os.path.join(sd, "iteration_{}".format(it), "som.pth"),
                    map_location="cpu", weights_only=True)
    return ck, it


def anatomy(model_dir):
    import torch
    from som_motion import SoMMotionModel, rot6d_to_matrix

    ck, it = load_ckpt(model_dir)
    rot6d, transl = ck["rot6d"], ck["transl"]  # [T,B,6] [T,B,3]
    T, B = rot6d.shape[:2]
    print("ckpt: iteration={}  frames={}  bases={}".format(it, T, B))

    tm = transl.norm(dim=-1)
    Rm = rot6d_to_matrix(rot6d)
    tr = Rm[..., 0, 0] + Rm[..., 1, 1] + Rm[..., 2, 2]
    ang = torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1, 1)))

    print("平移: mean={:.4f}  max={:.4f}  (退化基线 0.0087)".format(tm.mean(), tm.max()))
    print("旋转: mean={:.2f}°  max={:.2f}°  (退化基线 0.26°)".format(ang.mean(), ang.max()))

    # 系数场（需要高斯位置）
    import plyfile
    pc_dir = os.path.join(model_dir, "point_cloud", "iteration_{}".format(it))
    ply = plyfile.PlyData.read(os.path.join(pc_dir, "point_cloud.ply"))
    xyz = np.stack([ply["vertex"][k] for k in ("x", "y", "z")], 1)

    net = SoMMotionModel(num_frames=T, num_bases=B)
    net.coef_net.load_state_dict(ck["coef_net"])
    net.coef_net.cuda().eval()
    with torch.no_grad():
        xt = torch.from_numpy(xyz).float().cuda()
        w = torch.softmax(net.coef_net(xt), dim=-1).cpu().numpy()
    ent = -(w * np.log(w + 1e-9)).sum(1)
    dom = w.argmax(1)
    top3 = sorted([(float((dom == b).mean()), b) for b in range(B)], reverse=True)[:3]
    print("系数场: 主导基 top3 = " +
          "  ".join("基{}:{:.0%}".format(b, u) for u, b in top3))
    print("  熵 mean={:.2f}（均匀={:.2f}；退化基线 2.10）".format(ent.mean(), np.log(B)))

    # 判定
    verdict = []
    verdict.append(("平移", float(tm.mean()), 0.03, ">"))   # >3.4x 退化基线
    verdict.append(("旋转°", float(ang.mean()), 1.0, ">"))
    verdict.append(("主导基占比", top3[0][0], 0.5, "<"))
    verdict.append(("熵", float(ent.mean()), 1.8, "<"))
    ok = sum(1 for _, v, t, op in verdict if (v > t if op == ">" else v < t))
    print("\n判定: {}/4 项指示运动分解生效 → {}".format(
        ok, "✅ SOM 学到运动" if ok >= 2 else "❌ SOM 仍退化"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    anatomy(args.model)
