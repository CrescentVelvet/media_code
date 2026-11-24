"""Apply frozen source-only depth fits to SA Gaussian centres and evaluate rendered hold-outs."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

spec = importlib.util.spec_from_file_location("rigid", Path(__file__).with_name("33_rigid_face.py"))
rigid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rigid)
align, trace, ev, cam, S = rigid.align, rigid.trace, rigid.ev, rigid.cam, rigid.S


def calibrated_plys(out, raw, cfg, capture, src, poses, pack, unit, pars):
    paths = {"raw": raw}
    if pars["scale"] == 1 and not any(pars["offset_m"]):
        return dict.fromkeys(("raw", "scale", "offset"), raw)
    with np.load(capture) as data:
        g = trace.gaussians(data)
        pixels, weights = data["pixel"], data["weights"]
    foreground = []
    for i, row in enumerate(src):
        frame = next(f for f in pack["frames"] if f["raw_id"] == row["raw"])
        alpha = np.asarray(Image.open(frame["path"]).convert("RGBA"))[:, :, 3]
        fg = ev.M21.remap_nearest(alpha, row["layout"]) == 255
        foreground.append((pixels//S**2 == i) & fg.ravel()[pixels%S**2])
    for arm in ("scale", "offset"):
        means = g["means"].copy()
        for i, use in enumerate(foreground):
            origin = poses[i, :3, 3]
            delta = means[use] - origin
            z = delta @ poses[i, :3, 2]
            gain = pars["scale"]
            if arm == "offset":
                gain = gain + pars["offset_m"][i]*unit/z[:, None]
            means[use] = origin + delta*gain
        merged = align.gaussian_merge(dict(g, means=means), weights, cfg["model"]["gs_voxel_size"])
        keep = merged["opacities"] >= cfg["infer_opacity_thresh"]
        paths[arm] = out/(arm+".ply")
        align.save_ply(paths[arm], {k: v[keep] for k, v in merged.items()})
    return paths


def run(args, pack):
    name = pack["name"]
    old, out = args.trace/name, args.out/name
    log = cam.read(out/"source_fit.json")
    replay, = (old/"replay").iterdir()
    archive = args.protocol/"models/sa_pose"/replay.name
    scene = Path(cam.read(archive/"run.json")["input_path"])
    src = align.rows(pack, archive, scene, "source", old)
    poses, _, ids = ev.M03.load_cameras(replay/"cameras.json")
    assert log["source_ids"] == [r["raw"] for r in src] and ids == [r["stem"] for r in src]
    fit = ev.M15.estimate_sim3(replay, scene)[:3]
    cfg = yaml.safe_load((archive/"config.yaml").read_text())["wrapper"]
    paths = calibrated_plys(out, replay/"gaussians.ply", cfg, old/"pre.npz", src, poses,
                            pack, fit[0], log["models"]["SA"])
    baseline = {r["target"]: r for r in cam.read(old/"trace.json")["frames"]}
    reports = []
    for row in align.rows(pack, archive, Path(pack["target_scene"]), "target", old):
        raw = row["raw"]
        assert raw not in log["source_ids"]
        x, y, xx, yy = baseline[raw]["box_xyxy"]
        box = np.s_[y:yy, x:xx]
        target = ev.M21.official_preprocess_rgb(row["path"], row["layout"])
        pose = ev.M15.apply_sim3(row["pose"], *fit)
        truth, masks = rigid.masks_and_gt(pack, row)
        render_cache = {}
        stats, depths = {}, {}
        for arm, path in paths.items():
            if path not in render_cache:
                g = ev.M03.load_ply(path)
                rgb, alpha = ev.M03.render(g, pose, row["k"], S, S, background=0, device="cuda", eps2d=.01)
                z, a = ev.M21.M13.render_expected_depth(g, pose, row["k"], S, S, device="cuda", eps2d=.01)
                render_cache[path] = rgb[0][box], alpha[0,:,:,0][box], np.where(a[0,:,:,0]>.01, z[0]/fit[0], np.nan)
            crop, alpha_roi, depths[arm] = render_cache[path]
            trace.write_image(out/str(raw)/("render_"+arm+".png"), crop)
            stats[arm] = dict(rgb_mae=float(abs(crop-target[box]).mean()*255),
                              coverage=float((alpha_roi>.5).mean()))
        assert abs(stats["raw"]["rgb_mae"]-baseline[raw]["metrics"]["export"]["rgb_mae_255"]) < .01
        support = np.logical_and.reduce([np.isfinite(z) for z in depths.values()])
        for region, mask in masks.items():
            use = support & mask
            for arm, z in depths.items():
                error = abs(z[use]-truth[use])*1000
                stats[arm][region] = dict(p90_mm=float(np.percentile(error,90)),
                                          mae_mm=float(error.mean()), coverage=float(use.sum()/mask.sum()))
        reports.append(dict(raw=raw, arms=stats))
    cam.save(out/"gaussian_eval.json", reports)
    print(f"{name}: verified original replay; rendered {len(reports)} target cameras", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("protocol", "trace", "out"):
        ap.add_argument(arg, type=Path)
    args = ap.parse_args()
    packs = cam.read(args.protocol/"manifest.json")["packages"]
    for name in ("rosemary", "daven_holdout"):
        run(args, next(p for p in packs if p["name"]==name))
