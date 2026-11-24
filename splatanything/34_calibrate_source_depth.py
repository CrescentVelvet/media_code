"""Estimate depth scale and bounded axial offsets from source photographs, then test held-out views."""
import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location("rigid", Path(__file__).with_name("33_rigid_face.py"))
rigid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rigid)
align, trace, ev, cam, S = rigid.align, rigid.trace, rigid.ev, rigid.cam, rigid.S
ARMS = ("raw", "scale", "offset")
METHODS = ("SA", "DP1", "GT")


def source_arrays(pack, rows, pose, ks, scale, capture, single):
    with np.load(capture) as f:
        pixels, means = f["pixel"], f["means"].reshape(-1, 3)
    arrays = []
    for i, row in enumerate(rows):
        frame = next(f for f in pack["frames"] if f["raw_id"] == row["raw"])
        alpha = np.asarray(Image.open(frame["path"]).convert("RGBA"))[:, :, 3]
        fg = ev.M21.remap_nearest(alpha, row["layout"]) == 255
        rgb = ev.M21.official_preprocess_rgb(row["path"], row["layout"])
        source = pixels // S**2 == i
        pix = pixels[source] % S**2
        sa = np.full(fg.shape, np.nan)
        sa.flat[pix] = ((means[source] - pose[i, :3, 3]) @ pose[i, :3, 2]) / scale
        dp = ev.M21.remap_nearest(np.load(single / f"{i:02}.npz")["depth"], row["layout"])
        arrays.append(dict(rgb=rgb, fg=fg, SA=sa, DP1=dp))
    return arrays


def matched_depths(arrays, pose, ks, scale, rows, allow_face_texture=False):
    sift = cv2.SIFT_create(nfeatures=5000)
    features = []
    for a, row in zip(arrays, rows):
        # Use foreground outside the face. Nose and jaw remain independent evaluation points.
        face = cv2.dilate(align.hull_mask(row["hull"]), np.ones((13, 13), np.uint8))
        interior = cv2.erode(a["fg"].astype(np.uint8), np.ones((13, 13), np.uint8))
        mask = np.uint8((interior > 0) & (face == 0)) * 255
        if row["hull"] is None:
            mask.fill(0)
        if allow_face_texture and row["hull"] is not None:
            mask = interior*255
            for idx in (*rigid.TEST["nose"], *rigid.TEST["jaw"]):
                xy = tuple(np.rint(row["hull"][idx]).astype(int))
                cv2.circle(mask, xy, 24, 0, -1)
        gray = cv2.cvtColor(np.uint8(np.clip(a["rgb"]*255, 0, 255)), cv2.COLOR_RGB2GRAY)
        keys, desc = sift.detectAndCompute(gray, mask)
        features.append((np.array([k.pt for k in keys]), desc))
    cams = [k @ np.linalg.inv(p)[:3] for k, p in zip(ks, pose)]
    matcher = cv2.BFMatcher()
    depths = [{} for _ in rows]
    counts = {}
    for i, j in align.PAIRS:
        ki, di = features[i]
        kj, dj = features[j]
        counts[f"{i+1}-{j+1}"] = 0
        if di is None or dj is None or min(len(di), len(dj)) < 2:
            continue
        def ratio(a, b):
            return {(m.queryIdx, m.trainIdx) for m, n in matcher.knnMatch(a, b, k=2)
                    if m.distance < .75*n.distance}
        matches = sorted(ratio(di, dj) & {(b, a) for a, b in ratio(dj, di)})
        for ix, jx in matches:
            u, v = ki[ix], kj[jx]
            h = cv2.triangulatePoints(cams[i], cams[j], u[:, None], v[:, None])[:, 0]
            if abs(h[3]) < 1e-9:
                continue
            world = h[:3]/h[3]
            rays = world - pose[[i, j], :3, 3]
            rays /= np.linalg.norm(rays, axis=1)[:, None]
            angle = np.degrees(np.arccos(np.clip(rays[0] @ rays[1], -1, 1)))
            pair = [trace.project(world[None], pose[n], ks[n]) for n in (i, j)]
            if not (4 <= angle <= 100 and all(z[0] > 0 and np.linalg.norm(uv[0]-obs) <= 1.5
                    for (uv, z), obs in zip(pair, (u, v)))):
                continue
            counts[f"{i+1}-{j+1}"] += 1
            for n, xy, (_, z) in zip((i, j), (u, v), pair):
                q = int(align.pixel_index(xy)[0])
                depths[n].setdefault(q, []).append(z[0]/scale)
    return depths, counts


def calibrate(arrays, anchors, method):
    sources = []
    ratios = []
    residual_data = []
    for a, marks in zip(arrays, anchors):
        pix = np.array(sorted(marks), dtype=int)
        pred = a[method].ravel()[pix]
        ref = np.array([np.median(marks[p]) for p in pix])
        valid = np.isfinite(pred) & (pred > 0) & np.isfinite(ref)
        pix, pred, ref = pix[valid], pred[valid], ref[valid]
        xy = np.c_[pix % S, pix // S]
        spread = np.ptp(xy, axis=0) if len(xy) else np.zeros(2)
        supported = len(pix) >= 8 and spread.min() >= 45
        sources.append(dict(n=len(pix), xy_span_px=spread.tolist(), supported=bool(supported)))
        residual_data.append((pred, ref))
        if supported:
            ratios.append(np.median(ref/pred))
    s = float(np.clip(np.median(ratios), .60, 1.50)) if len(ratios) >= 2 else 1.
    offsets = []
    for log, (pred, ref) in zip(sources, residual_data):
        dz = float(np.clip(np.median(ref-s*pred), -.030, .030)) if log["supported"] and len(ratios)>=2 else 0.
        offsets.append(dz)
        log["anchor_abs_median_mm"] = [float(np.median(abs(z-ref))*1000) for z in
                                       (pred, s*pred, s*pred+dz)] if len(pred) else []
    return dict(scale=s, offset_m=offsets, supported_sources=len(ratios), sources=sources)


def points(a, method, arm, pars, i, pose, k, scale, q):
    z = a[method].ravel()[q]
    if arm != "raw":
        z = pars["scale"] * z
    if arm == "offset":
        z = z + pars["offset_m"][i]
    rays = np.c_[q % S, q // S, np.ones(len(q))] @ np.linalg.inv(k).T
    return (rays * (scale*z[:, None])) @ pose[:3, :3].T + pose[:3, 3]


def measure(arrays, pars, method, src, poses, ks, target, tar_pose, scale, truth, masks, box, folder):
    depth_layers = {}
    mark_stats = {}
    for arm in ARMS:
        depths, colors = [], []
        obs = {k: [] for k in rigid.TEST}
        for i, (a, row) in enumerate(zip(arrays, src)):
            good = a["fg"] & np.isfinite(a["SA"]) & np.isfinite(a[method]) & (a[method] > .031)
            q = np.flatnonzero(good)
            xyz = points(a, method, arm, pars, i, poses[i], ks[i], scale, q)
            d, c = rigid.raster(xyz, a["rgb"].reshape(-1, 3)[q], tar_pose, target["k"])
            depths.append(d/scale)
            colors.append(c)
            if row["hull"] is None or target["hull"] is None:
                continue
            for group, ids in rigid.TEST.items():
                for idx in ids:
                    q, inside = align.pixel_index(row["hull"][idx])
                    if not inside or not good.flat[q]:
                        continue
                    p = points(a, method, arm, pars, i, poses[i], ks[i], scale, np.array([q]))
                    uv, z = trace.project(p, tar_pose, target["k"])
                    if z[0] > 0:
                        obs[group].append(float(np.linalg.norm(uv[0]-target["hull"][idx])))
        mark_stats[arm] = {k: dict(n=len(v), median_px=align.median(v)) for k, v in obs.items()}
        d = np.stack(depths)
        depth_layers[arm] = d
        overlay = np.sum(colors, axis=0) / np.maximum(1, np.isfinite(d).sum(axis=0))[:, :, None]
        trace.write_image(folder / f"{method}_{arm}.png", overlay[box])
    same = np.logical_and.reduce([np.isfinite(d) for d in depth_layers.values()])
    stats = {}
    for region, mask in masks.items():
        use = mask & same.any(axis=0)
        multi = mask & (same.sum(axis=0) >= 2)
        metrics = dict(roi=int(mask.sum()), common=int(use.sum()), multi=int(multi.sum()))
        for arm, d in depth_layers.items():
            near = np.where(same, d, np.inf).min(axis=0)
            span = (np.where(same, d, -np.inf).max(axis=0)[multi]-near[multi])*1000
            errors = abs(near[use]-truth[use])*1000
            metrics[arm] = dict(p90_mm=float(np.percentile(errors, 90)) if len(errors) else None,
                                mae_mm=float(errors.mean()) if len(errors) else None,
                                span_mm=align.median(span), visible=int((mask & np.isfinite(d).any(axis=0)).sum()))
        stats[region] = metrics
    return dict(regions=stats, landmarks=mark_stats)


def run(args, pack):
    name = pack["name"]
    old = args.trace/name
    replay, = (old/"replay").iterdir()
    archive = args.protocol/"models/sa_pose"/replay.name
    scene = Path(cam.read(archive/"run.json")["input_path"])
    src = align.rows(pack, archive, scene, "source", old)
    poses, ks, ids = ev.M03.load_cameras(replay/"cameras.json")
    assert ids == [r["stem"] for r in src]
    fit = ev.M15.estimate_sim3(replay, scene)[:3]
    single = args.previous/name/"dp_input/single_frame/depth"
    arrays = source_arrays(pack, src, poses, ks, fit[0], old/"pre.npz", single)
    anchors, counts = matched_depths(arrays, poses, ks, fit[0], src)
    pars = {m: calibrate(arrays, anchors, m) for m in ("SA", "DP1")}
    trials = dict(outside_face=counts)
    selected = "outside_face"
    if min(p["supported_sources"] for p in pars.values()) < 2:
        extra, extra_counts = matched_depths(arrays, poses, ks, fit[0], src, allow_face_texture=True)
        trials["foreground_excluding_test_marks"] = extra_counts
        candidates = {m: calibrate(arrays, extra, m) for m in pars}
        if min(p["supported_sources"] for p in candidates.values()) >= 2:
            anchors, pars = extra, candidates
            selected = "foreground_excluding_test_marks"
    out = args.out/name
    out.mkdir(parents=True, exist_ok=True)
    cam.save(out/"source_fit.json", dict(source_ids=[r["raw"] for r in src], trials=trials,
                                         selected=selected, models=pars))
    cam.save(out/"source_anchors.json", [
        dict(raw=row["raw"], points=[dict(x=int(q % S), y=int(q // S),
             triangulated_z_m=float(np.median(refs)), pair_observations=len(refs))
             for q, refs in sorted(marks.items())]) for row, marks in zip(src, anchors)])
    # Prediction parameters are frozen before reading any depth ground truth.
    for a, row in zip(arrays, src):
        frame = next(f for f in pack["frames"] if f["raw_id"] == row["raw"])
        z, ok = rigid.geometry(frame, row)
        a["GT"] = np.where(ok, z, np.nan)
    pars["GT"] = calibrate(arrays, anchors, "GT")
    cam.save(out/"gt_control_fit.json", pars["GT"])
    previous = {r["target"]: r for r in cam.read(old/"trace.json")["frames"]}
    results = []
    for row in align.rows(pack, archive, Path(pack["target_scene"]), "target", old):
        raw = row["raw"]
        assert raw not in [r["raw"] for r in src]
        sub = out/str(raw)
        sub.mkdir(exist_ok=True)
        x, y, xx, yy = previous[raw]["box_xyxy"]
        box = np.s_[y:yy, x:xx]
        trace.write_image(sub/"target.png", ev.M21.official_preprocess_rgb(row["path"],row["layout"])[box])
        truth, masks = rigid.masks_and_gt(pack, row)
        pose = ev.M15.apply_sim3(row["pose"], *fit)
        results.append(dict(raw=raw, models={m: measure(arrays, pars[m], m, src, poses, ks,
                            row, pose, fit[0], truth, masks, box, sub) for m in METHODS}))
    cam.save(out/"eval.json", results)
    print(f"{name}: {trials}, selected {selected}, evaluated {len(results)} held-out views", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("protocol", "trace", "previous", "out"):
        ap.add_argument(arg, type=Path)
    args = ap.parse_args()
    packs = cam.read(args.protocol/"manifest.json")["packages"]
    for name in ("rosemary", "daven_holdout"):
        run(args, next(p for p in packs if p["name"] == name))
