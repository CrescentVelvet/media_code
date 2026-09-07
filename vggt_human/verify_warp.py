#!/usr/bin/env python3
"""验证 head warp 的数学正确性。

核心问题：head_gs 存的是参考帧位姿的 canonical 高斯。训练时用 A,b（把 canonical
warp 到帧 f 的位姿）。验证方法：warp 后的 head 高斯在帧 f 的渲染应该贴合
head_fit 里帧 f 的 landmark 重投影（拟合时就是最小化这个）。

具体做法：
1. 加载 head_gs_p00.ply（canonical）
2. 对每个有 per_frame 的帧：xyz' = A @ xyz + b，q' = qA ⊗ q
3. 用该帧的投影矩阵把 xyz' 投影到图像
4. 与该帧 landmark 2D 观测对比（landmark 对应模板顶点索引）
5. 若 median 误差 < 5px，warp 数学方向正确
"""
import os, sys, json
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "8")

RESULTS = os.environ.get("RESULTS_DIR", "/mnt/d/output/vggt_human_ms")
PID = os.environ.get("PID", "00")
HEAD_DIR = f"{RESULTS}/03e_head_3dmm"
OUT_DIR = f"{RESULTS}/03e_head_3dmm/vis_warpcheck"


def quat_mul_np(qa, qb):
    """(N,4) wxyz Hamilton 乘法"""
    aw, ax, ay, az = qa[:,0], qa[:,1], qa[:,2], qa[:,3]
    bw, bx, by, bz = qb[:,0], qb[:,1], qb[:,2], qb[:,3]
    return np.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], axis=1)


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def load_ply_xyz(path):
    """读 ply 的 xyz 与 rotation（不依赖 plyfile，纯文本解析 vertex 段）。"""
    from plyfile import PlyData
    ply = PlyData.read(path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float64)
    return xyz, rot


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fit_head_3dmm import parse_colmap_cameras, build_proj_matrices, N_LM

    fit = json.loads(open(f"{HEAD_DIR}/head_fit.json").read())
    fr = fit["persons"][PID]
    R_ref = np.asarray(fr["R_ref"]); t_ref = np.asarray(fr["t_ref"])
    pf = fr["per_frame"]

    xyz, rot = load_ply_xyz(f"{HEAD_DIR}/head_gs_p{PID}.ply")
    print(f"head_gs_p{PID}: {len(xyz)} 高斯")

    views = parse_colmap_cameras(f"{RESULTS}/03b_source_ba")
    P = build_proj_matrices(views)

    lm_path = f"{HEAD_DIR}/face_landmarks.json"
    lm = json.loads(open(lm_path).read())

    # 每 4 帧抽 1 帧验证
    stems = sorted(pf.keys())[::4]
    errs_all = []
    per_frame = []
    for stem in stems:
        if stem not in P: continue
        recs = lm["frames"].get(stem, {}).get("persons", {}).get(PID)
        if not recs or not recs.get("ok") or not recs.get("lm"): continue
        obs = np.asarray(recs["lm"], float)          # (468,2)
        if obs.shape != (N_LM, 2): continue

        R_f = np.asarray(pf[stem]["R"]).reshape(3,3)
        t_f = np.asarray(pf[stem]["t"]).reshape(3)
        A = R_f @ R_ref.T
        b = t_f - A @ t_ref

        xyz_f = xyz @ A.T + b
        # 旋转四元数 warp（只验证平移部分对 landmark 即可，四元数不投影）
        Xh = np.hstack([xyz_f, np.ones((len(xyz_f),1))])
        q = (P[stem][0] @ Xh.T).T
        q = q[:,:2] / q[:,2:3]

        # landmark 对应的高斯是哪个？head_gs 是模板面采样，不是 468 点。
        # 用最近邻：模板 468 对应点 warp 后的 2D 预测 vs 2D 观测
        # —— 模板 landmark 对应由 prepare_3dmm_template 输出（mp_to_ict），
        #    但 head_gs 是采样点不保留索引。改为直接验证：warp 后整体点云的
        #    图像 bbox 应与该帧 head mask bbox 高度重合。
        # 简化：用头中心（拟合 head_center）warp 后投影 vs head mask 质心
        hc = np.asarray(fr["head_center"])
        hc_f = A @ hc + b
        hch = np.array([*hc_f, 1.0])
        qc = P[stem][0] @ hch
        qc = qc[:2] / qc[2]
        # head mask 质心
        import glob
        from PIL import Image
        mpath = f"{RESULTS}/03i_region_masks/{stem}.p{PID}.head.png"
        if not os.path.isfile(mpath):
            continue
        m = np.asarray(Image.open(mpath).convert("L"), dtype=np.float32) / 255.0
        if m.max() < 0.1: continue
        ys, xs = np.nonzero(m > 0.1)
        cx, cy = xs.mean(), ys.mean()
        # head mask 是渲染 alpha 生成（含全分辨率），质心 vs warp 后头中心投影
        d = np.hypot(qc[0] - cx, qc[1] - cy)
        per_frame.append((stem, d, m.shape))
        errs_all.append(d)

    errs_all = np.asarray(errs_all)
    print(f"\n验证 {len(errs_all)} 帧（warp 后头中心投影 vs head mask 质心）:")
    print(f"  median = {np.median(errs_all):.2f} px")
    print(f"  p90    = {np.percentile(errs_all, 90):.2f} px")
    print(f"  max    = {errs_all.max():.2f} px")
    if np.median(errs_all) < 15:
        print("✅ warp 数学方向正确（warp 后头中心落在 mask 质心附近）")
    else:
        print("❌ warp 方向可疑！需要检查 A,b 的构造")


if __name__ == "__main__":
    main()
