#!/usr/bin/env python3
"""解剖 model_p07 的 SOM 运动基：运动幅度 + 系数场空间分组。"""
import os
import sys
import numpy as np
import torch

DG = "/home/velvet/repos/Deformable-3D-Gaussians"
SOM = "/mnt/c/code/media_code/som_human"
sys.path.insert(0, DG)
sys.path.insert(0, SOM)
os.chdir(DG)

R = "/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou"
MODEL = os.path.join(R, "model_p07")

from som_motion import SoMMotionModel, rot6d_to_matrix

# 找最新权重
sd = os.path.join(MODEL, "som_motion")
iters = [d for d in os.listdir(sd) if d.startswith("iteration_")]
it = max(int(d.split("_")[1]) for d in iters)
ck = torch.load(os.path.join(sd, "iteration_{}".format(it), "som.pth"),
                map_location="cpu", weights_only=True)
print("ckpt: iteration={}, num_frames={}, num_bases={}".format(
    it, ck["num_frames"], ck["num_bases"]))

rot6d = ck["rot6d"]   # [T,B,6]
transl = ck["transl"]  # [T,B,3]
T, B = rot6d.shape[:2]

# 1) 每条基的平移幅度（相对 identity）
tm = transl.norm(dim=-1)  # [T,B]
print("\n平移幅度 (单位=场景坐标):")
print("  每基均值: " + " ".join("{:.4f}".format(tm[:, b].mean()) for b in range(B)))
print("  全局: mean={:.4f} max={:.4f}".format(tm.mean(), tm.max()))

# 2) 每条基的旋转幅度（转轴角，度）
Rm = rot6d_to_matrix(rot6d)  # [T,B,3,3]
tr = Rm[..., 0, 0] + Rm[..., 1, 1] + Rm[..., 2, 2]
ang = torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1, 1)))  # [T,B]
print("\n旋转幅度 (度):")
print("  每基均值: " + " ".join("{:.2f}".format(ang[:, b].mean()) for b in range(B)))
print("  全局: mean={:.2f} max={:.2f}".format(ang.mean(), ang.max()))

# 3) 系数场：采样空间网格，看系数的空间分布是否有"分区"
net = SoMMotionModel(num_frames=T, num_bases=B)
net.coef_net.load_state_dict(ck["coef_net"])
net.coef_net.cuda().eval()

# 读高斯位置，取手部附近区域（mask 里的点）
import plyfile
ply = plyfile.PlyData.read(os.path.join(MODEL, "point_cloud",
                                        "iteration_{}".format(it), "point_cloud.ply"))
xyz = np.stack([ply["vertex"][k] for k in ("x", "y", "z")], 1)
print("\n高斯数: {}".format(len(xyz)))
print("xyz 范围: {}".format(np.round([xyz.min(0), xyz.max(0)], 3).tolist()))

with torch.no_grad():
    xt = torch.from_numpy(xyz).float().cuda()
    w = torch.softmax(net.coef_net(xt), dim=-1).cpu().numpy()  # [N,B]
ent = -(w * np.log(w + 1e-9)).sum(1)  # 系数熵：低=明确分组，高=混合
dom = w.argmax(1)
print("\n系数场:")
print("  主导基分布: " + " ".join("{}:{:.0%}".format(b, (dom == b).mean()) for b in range(B)))
print("  系数熵: mean={:.2f} (max={:.2f}=均匀)".format(ent.mean(), np.log(B)))
# 相关性：运动幅度大的基是否被更多点使用
usage = np.array([(dom == b).mean() for b in range(B)])
motion = tm.mean(0).numpy()
print("  基运动幅度 vs 使用率 corr: {:.2f}".format(np.corrcoef(motion, usage)[0, 1]))
