"""反算 studio_human 实拍相机的半径/高度分布，给 Blender 相机布置做基准。"""
import numpy as np
z = np.load("/home/velvet/output/da3_human_results/bench_first_125v/cameras.npz")
ext = z["extrinsics"].astype(np.float64)
r, t = ext[:, :3, :3], ext[:, :3, 3]
c = np.einsum("nij,nj->ni", -np.transpose(r, (0, 2, 1)), t)   # 相机中心（世界系）
o = c.mean(0)
d = np.linalg.norm(c - o, axis=1)
print(f"相机数 {len(c)} | 世界中心 {np.round(o,3)}")
print(f"半径: min {d.min():.3f} / p25 {np.percentile(d,25):.3f} / med {np.median(d):.3f} / p75 {np.percentile(d,75):.3f} / max {d.max():.3f} m")
print(f"高度(y): min {c[:,1].min():.3f} / med {np.median(c[:,1]):.3f} / max {c[:,1].max():.3f} (中心 y={o[1]:.3f})")
az = np.degrees(np.arctan2(c[:,0]-o[0], c[:,2]-o[2]))
print(f"方位角覆盖: {az.min():.1f} ~ {az.max():.1f} deg")
K = z["intrinsics"]
print(f"fx/W 分布: min {K[:,0,0].min()/1440:.4f} max {K[:,0,0].max()/1440:.4f}")
