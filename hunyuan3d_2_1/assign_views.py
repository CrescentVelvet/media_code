#!/usr/bin/env python3
"""Assign Hunyuan3D-2mv canonical view tags from VGGT-Omega camera poses.

Why: the mv model feeds each view a learned embedding for its canonical slot
(hy3dshape preprocessors.py: front/left/back/right = 0/90/180/270 deg). Our
inputs are video frames at unknown angles, so tags must come from real poses.

For each object: read VGGT_OUT/<object>/predictions.npz (extrinsic = w2c),
recover camera centers, fit the orbit center (least-squares closest point to
the optical axes), estimate world-up from camera y-axes, compute per-view
azimuth relative to view 0, snap to slots (err <=45 deg, unique slots), and
write PREP_DIR/<object>/views.json. If all views collapse onto one slot
(too-narrow arc), fall back to mode=single with the largest-alpha-area view.

Env vars: PREP_DIR, VGGT_OUT, MAX_SLOT_ERR (deg, default 45)
"""
import json
import os
import sys

import numpy as np
from PIL import Image

PREP_DIR = os.environ["PREP_DIR"]
VGGT_OUT = os.environ["VGGT_OUT"]
MAX_SLOT_ERR = float(os.environ.get("MAX_SLOT_ERR", "45"))

# hy3dshape preprocessors.py: "front, front clockwise 90, back, front clockwise 270"
SLOTS = [(0.0, "front"), (90.0, "left"), (180.0, "back"), (270.0, "right")]


def camera_geometry(extrinsic):
    """w2c [N,3,4] -> centers C [N,3], forward dirs d [N,3], image-down v [N,3]."""
    R = extrinsic[:, :3, :3]
    t = extrinsic[:, :, 3]
    centers = -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)
    fwd = np.einsum("nij,j->ni", np.transpose(R, (0, 2, 1)), np.array([0, 0, 1.0]))
    down = np.einsum("nij,j->ni", np.transpose(R, (0, 2, 1)), np.array([0, 1.0, 0]))
    return centers, fwd, down


def orbit_center(centers, dirs):
    """Least-squares point closest to all optical axes (min sum ||A_i(P-C_i)||)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for C, d in zip(centers, dirs):
        d = d / np.linalg.norm(d)
        Ai = np.eye(3) - np.outer(d, d)
        A += Ai
        b += Ai @ C
    return np.linalg.solve(A + 1e-9 * np.eye(3), b)


def azimuths(centers, up, center):
    """Azimuth of each camera around `center`, relative to camera 0, in [0,360)."""
    up = up / np.linalg.norm(up)
    U = centers - center
    U = U - np.outer(U @ up, up)
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    u0 = U[0]
    out = [0.0]
    for u in U[1:]:
        ang = np.degrees(np.arctan2(np.dot(np.cross(u0, u), up), np.dot(u0, u)))
        out.append(float(ang % 360.0))
    return out


def ang_err(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def assign_slots(azs):
    """Greedy: candidates by ascending slot error; unique slots; err <= MAX_SLOT_ERR."""
    cand = []
    for i, az in enumerate(azs):
        for slot_ang, slot_name in SLOTS:
            cand.append((ang_err(az, slot_ang), i, slot_name))
    cand.sort()
    used_views, used_slots, out = set(), set(), {}
    for err, i, slot_name in cand:
        if i in used_views or slot_name in used_slots or err > MAX_SLOT_ERR:
            continue
        out[slot_name] = i
        used_views.add(i)
        used_slots.add(slot_name)
    return out


def largest_alpha_view(png_paths):
    best, best_area = 0, -1
    for i, p in enumerate(png_paths):
        a = np.asarray(Image.open(p).split()[-1])
        area = int((a > 10).sum())
        if area > best_area:
            best, best_area = i, area
    return best


def main():
    objects = sorted(
        d for d in os.listdir(PREP_DIR) if os.path.isdir(os.path.join(PREP_DIR, d))
    )
    if not objects:
        sys.exit("❌ no object folders in " + PREP_DIR)

    for obj in objects:
        obj_dir = os.path.join(PREP_DIR, obj)
        with open(os.path.join(obj_dir, "prep_manifest.json"), encoding="utf-8") as fp:
            stems = json.load(fp)["images"]
        npz_path = os.path.join(VGGT_OUT, obj, "predictions.npz")
        if not os.path.exists(npz_path):
            print(f"⚠️ {obj}: {npz_path} missing, skipped")
            continue
        extrinsic = np.load(npz_path)["extrinsic"].reshape(len(stems), -1)[:, :12]
        extrinsic = extrinsic.reshape(len(stems), 3, 4)

        centers, fwd, down = camera_geometry(extrinsic)
        up = -down.mean(axis=0)
        center = orbit_center(centers, fwd)
        azs = azimuths(centers, up, center)
        slot_map = assign_slots(azs)

        rgba = [os.path.join(obj_dir, "rgba", s + ".png") for s in stems]
        result = {"azimuths_deg": {stems[i]: round(azs[i], 1) for i in range(len(stems))}}
        if len(slot_map) >= 2:
            result["mode"] = "mv"
            result["views"] = {slot: "rgba/" + stems[i] + ".png" for slot, i in slot_map.items()}
        else:
            best = largest_alpha_view(rgba)
            result["mode"] = "single"
            result["image"] = "rgba/" + stems[best] + ".png"
            result["note"] = "view arc too narrow for multi-view slots; fell back to single"
        with open(os.path.join(obj_dir, "views.json"), "w", encoding="utf-8", newline="\n") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=2)
        if result["mode"] == "mv":
            # views 的 value 是 "rgba/<stem>.png"，要映射回 azs 的整数索引
            idx_of = {os.path.basename(p)[: -len(".png")]: i for i, p in enumerate(rgba)}
            az_str = ", ".join(
                f"{s}={azs[idx_of[os.path.basename(v)[: -len('.png')]]]:.0f}°"
                for s, v in sorted(result["views"].items(), key=lambda kv: kv[1])
            )
        else:
            az_str = stems[int(largest_alpha_view(rgba))]
        print(f"🎯 {obj}: mode={result['mode']}  {az_str}")
    print("🎉 [assign] done")


if __name__ == "__main__":
    main()
