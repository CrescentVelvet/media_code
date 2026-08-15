#!/usr/bin/env python3
"""渲染 N 视角骨骼深度图 (SMPL mesh -> pyrender depth)。

读 reference.npz 的 vertices+faces, 居中 mesh, 用 orbit 相机 (绕 mesh 一圈)
渲染每视角的深度图, 存 .npy (float32 米) + .png (uint8 可视化, 04 ControlNet
输入) + cameras.npz (每视角 camera-to-world 位姿, 给 05 重建用)。

纯 flux_human env (pyrender + trimesh, 不 import sam_3d_body)。

Env vars (set by 03_render_depth.sh):
  REF_NPZ, DEPTH_DIR, NUM_VIEWS, ELEVATION, IMG_SIZE, CAMERA_DIST, FOV_DEG
"""
import os
import sys
import numpy as np

REF_NPZ = os.environ.get("REF_NPZ")
DEPTH_DIR = os.environ.get("DEPTH_DIR")
NUM_VIEWS = int(os.environ.get("NUM_VIEWS", "24"))
ELEVATION = float(os.environ.get("ELEVATION", "-10"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "768"))
CAMERA_DIST = os.environ.get("CAMERA_DIST", "")
FOV_DEG = float(os.environ.get("FOV_DEG", "35"))


def look_at(eye, target, up):
    """Camera-to-world pose (OpenGL convention: -Z forward)."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    M = np.eye(4)
    M[:3, 0] = right
    M[:3, 1] = new_up
    M[:3, 2] = -forward
    M[:3, 3] = eye
    return M


def main():
    if not REF_NPZ or not DEPTH_DIR:
        sys.exit("❌ REF_NPZ / DEPTH_DIR not set")
    if not os.path.isfile(REF_NPZ):
        sys.exit(f"❌ {REF_NPZ} not found — run 02 first")

    import trimesh
    import pyrender

    d = np.load(REF_NPZ, allow_pickle=True)
    verts = d["vertices"]
    faces = d["faces"]
    if verts is None or faces is None:
        sys.exit("❌ reference.npz 缺 vertices/faces")
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)
    print(f"🏋️ mesh: vertices {verts.shape}, faces {faces.shape}")

    # Center the mesh at origin (orbit camera revolves around origin).
    center = verts.mean(axis=0)
    verts = verts - center
    extent = float(np.abs(verts).max())
    print(f"📐 mesh extent (居中后): {extent:.3f}m")

    # Camera distance: auto from mesh extent + FOV, or override.
    fov = np.radians(FOV_DEG)
    if CAMERA_DIST:
        dist = float(CAMERA_DIST)
    else:
        # distance so mesh fills ~60% of frame: d = extent / tan(fov/2) * factor
        dist = extent / np.tan(fov / 2) * 1.4
    print(f"🎮 camera dist={dist:.3f}m  fov={FOV_DEG}°  size={IMG_SIZE}")

    # Build mesh (light grey). front-facing tweaks: trimesh uses +Z forward by
    # default for some loaders; pyrender wants +Z back (OpenGL). Use the
    # Mesh.from_trimesh with smooth=False (flat normals are fine for depth).
    tmesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    pmesh = pyrender.Mesh.from_trimesh(tmesh, smooth=False)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3])
    scene.add(pmesh)

    cam = pyrender.PerspectiveCamera(
        yfov=fov, aspectRatio=1.0, znear=0.05, zfar=dist * 5
    )

    renderer = pyrender.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    flags = pyrender.RenderFlags.DEPTH_ONLY

    el = np.radians(ELEVATION)
    cam_poses = []
    os.makedirs(DEPTH_DIR, exist_ok=True)
    try:
        from PIL import Image
        has_pil = True
    except ImportError:
        has_pil = False

    print(f"🚀 渲染 {NUM_VIEWS} 视角 (orbit, elevation={ELEVATION}°) ...")
    for i in range(NUM_VIEWS):
        az = 2 * np.pi * i / NUM_VIEWS
        eye = dist * np.array([
            np.cos(el) * np.cos(az),
            np.sin(el),
            np.cos(el) * np.sin(az),
        ])
        pose = look_at(eye, [0, 0, 0], [0, 1, 0])
        cam_poses.append(pose)
        sn = scene.add(cam, pose=pose)
        depth = renderer.render(scene, flags=flags)
        scene.remove_node(sn)

        # depth is float32 in meters (zfar regions = 0 / inf).
        depth = np.asarray(depth, dtype=np.float32)
        np.save(os.path.join(DEPTH_DIR, f"depth_{i:03d}.npy"), depth)

        # uint8 visualization for ControlNet: min-max normalize to 0-255.
        valid = depth[depth > 0]
        if valid.size > 0:
            lo, hi = float(valid.min()), float(valid.max())
            vis = np.zeros_like(depth, dtype=np.uint8)
            mask = depth > 0
            vis[mask] = np.clip((depth[mask] - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8)
            if has_pil:
                Image.fromarray(vis, mode="L").save(os.path.join(DEPTH_DIR, f"depth_{i:03d}.png"))
        print(f"  [{i+1}/{NUM_VIEWS}] az={np.degrees(az):.1f}° -> depth_{i:03d}.npy/png  "
              f"valid={valid.size} ({valid.size/depth.size*100:.0f}%)")
    renderer.delete()

    np.savez(os.path.join(DEPTH_DIR, "cameras.npz"),
             poses=np.stack(cam_poses),
             dist=dist, fov=fov, img_size=IMG_SIZE, elevation=ELEVATION)
    print(f"✅ done. {NUM_VIEWS} 视角 -> {DEPTH_DIR}")
    print(f"   cameras.npz: {len(cam_poses)} poses (给 05 重建用)")
    print(f"   下一步: GPU=0 bash flux_human/04_generate_views.sh")


if __name__ == "__main__":
    main()
