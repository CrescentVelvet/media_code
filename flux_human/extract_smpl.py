#!/usr/bin/env python3
"""选静止参考帧 + 导出 SMPL 数据供后续渲染/生成。

读 sam_3d_body 输出的每帧 npz, 按 body_pose_params 的窗口内变化选最静止帧
T*, 导出 T* 的 vertices/cam_t/faces/keypoints/focal/bbox + 参考帧原图路径
到 reference.npz。faces 从 SAM 3D Body 模型取 (MHR 固定拓扑), 缓存到
faces.npy 只加载一次。

⚠️ 此脚本需在 sam_3d_body env 跑 (要 import sam_3d_body 取 faces); 由
02_extract_smpl.sh 用 `conda run -n sam_3d_body` 调用。

Env vars:
  RESULTS_DIR, SAM3D_RESULTS, SMPL_OUT, SAM3D_DIR,
  CHECKPOINT_PATH, MHR_PATH, FRAMES_DIR, WINDOW
"""
import os
import sys
import glob
import numpy as np

SAM3D_DIR = os.environ.get("SAM3D_DIR", "../sam-3d-body")
sys.path.insert(0, SAM3D_DIR)  # so `from sam_3d_body import ...` resolves

RESULTS_DIR = os.environ.get("RESULTS_DIR")
SAM3D_RESULTS = os.environ.get("SAM3D_RESULTS", f"{RESULTS_DIR}/sam3d")
SMPL_OUT = os.environ.get("SMPL_OUT", f"{RESULTS_DIR}/smpl")
FRAMES_DIR = os.environ.get("FRAMES_DIR", f"{RESULTS_DIR}/frames")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "")
MHR_PATH = os.environ.get("MHR_PATH", "")
WINDOW = int(os.environ.get("WINDOW", "5"))


def get_faces():
    """Load SAM 3D Body model once to grab the MHR mesh faces (cached)."""
    cache = os.path.join(SMPL_OUT, "faces.npy")
    if os.path.exists(cache):
        faces = np.load(cache)
        print(f"⏭️ faces 已缓存: {cache} ({len(faces)} faces)")
        return faces
    if not CHECKPOINT_PATH:
        sys.exit("❌ CHECKPOINT_PATH not set (needed to load model for faces)")
    print("🏋️ 加载 SAM 3D Body 模型取 faces (仅首次, 之后缓存到 faces.npy) ...")
    try:
        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
        model, model_cfg = load_sam_3d_body(
            CHECKPOINT_PATH, device="cpu", mhr_path=MHR_PATH
        )
        est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg)
        faces = np.asarray(est.faces)
    except Exception as e:
        sys.exit(f"❌ 加载模型取 faces 失败: {e}\n"
                 f"   introspect: python -c \"from sam_3d_body import SAM3DBodyEstimator; print(dir(SAM3DBodyEstimator))\"")
    os.makedirs(SMPL_OUT, exist_ok=True)
    np.save(cache, faces)
    print(f"✅ faces 缓存: {cache} ({len(faces)} faces)")
    return faces


def load_frame(path):
    """Read one frame's npz -> dict of p0 outputs (None if no person)."""
    d = np.load(path)
    if int(d["n_persons"]) == 0:
        return None

    def g(k):
        return d[k] if k in d.files else None

    return {
        "vertices": g("pred_vertices_p0"),
        "cam_t": g("pred_cam_t_p0"),
        "body_pose": g("body_pose_params_p0"),
        "global_rot": g("global_rot_p0"),
        "kp3d": g("pred_keypoints_3d_p0"),
        "kp2d": g("pred_keypoints_2d_p0"),
        "focal": g("focal_length_p0"),
        "bbox": g("bbox_p0"),
        "npz_path": path,
        "stem": os.path.splitext(os.path.basename(path))[0],
    }


def select_reference(frames):
    """Pick the frame with the smallest pose change within a sliding window."""
    poses = []
    idx_map = []
    for i, f in enumerate(frames):
        p = f.get("body_pose")
        if p is None:
            continue
        poses.append(p.flatten())
        idx_map.append(i)
    if not poses:
        sys.exit("❌ no body_pose_params_p0 in any npz (sam_3d_body INFERENCE_TYPE=body 应有)")
    poses = np.stack(poses)
    n = len(poses)
    scores = np.full(n, np.inf)
    half = WINDOW // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = poses[lo:hi]
        if len(seg) >= 2:
            scores[i] = float(np.sum(np.linalg.norm(seg[1:] - seg[:-1], axis=1)))
        else:
            scores[i] = 0.0
    best_local = int(np.argmin(scores))
    best = idx_map[best_local]
    print(f"🎯 参考帧: {frames[best]['stem']} (#{best_local}/{n} in pose-tracked, "
          f"score={scores[best_local]:.4f}, window={WINDOW})")
    return best


def find_ref_image(frame):
    """Locate the original frame image (for IP-Adapter in 04)."""
    stem = frame["stem"]
    for ext in (".jpg", ".jpeg", ".png"):
        cand = os.path.join(FRAMES_DIR, stem + ext)
        if os.path.isfile(cand):
            return cand
    cand = os.path.join(FRAMES_DIR, stem)
    if os.path.isfile(cand):
        return cand
    return ""


def main():
    if not RESULTS_DIR:
        sys.exit("❌ RESULTS_DIR not set")
    npzs = sorted(glob.glob(os.path.join(SAM3D_RESULTS, "npz", "**", "*.npz"), recursive=True))
    if not npzs:
        sys.exit(f"❌ no npz in {SAM3D_RESULTS}/npz — run sam_3d_body inference first")
    print(f"🖼️ {len(npzs)} 帧 npz")
    frames = [load_frame(p) for p in npzs]
    good = [f for f in frames if f is not None]
    if not good:
        sys.exit("❌ no persons detected in any frame")
    print(f"✅ {len(good)}/{len(frames)} 帧含人体")

    best = select_reference(frames)
    T = frames[best]
    faces = get_faces()
    ref_image = find_ref_image(T)
    if ref_image:
        print(f"🖼️ 参考帧原图 (IP-Adapter 用): {ref_image}")
    else:
        print(f"⚠️ 未找到参考帧原图 (stem={T['stem']}, FRAMES_DIR={FRAMES_DIR}); 04 IP-Adapter 将无 ref")

    os.makedirs(SMPL_OUT, exist_ok=True)
    out = os.path.join(SMPL_OUT, "reference.npz")
    np.savez_compressed(
        out,
        vertices=T["vertices"],
        cam_t=T["cam_t"],
        faces=faces,
        kp3d=T["kp3d"],
        kp2d=T["kp2d"],
        focal=T["focal"],
        bbox=T["bbox"],
        frame_idx=best,
        frame_stem=T["stem"],
        frame_path=T["npz_path"],
        ref_image=ref_image,
        n_frames=len(frames),
    )
    print(f"✅ 导出: {out}")
    print(f"   vertices: {None if T['vertices'] is None else T['vertices'].shape}, "
          f"faces: {faces.shape}, cam_t: {None if T['cam_t'] is None else T['cam_t'].shape}")
    print(f"   下一步: GPU=0 bash flux_human/03_render_depth.sh")


if __name__ == "__main__":
    main()
