"""Trace the same held-out camera through lifting, source composition, merge and export."""
import argparse
import importlib.util
import itertools
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location("cam", Path(__file__).with_name("29_camera_probe.py"))
cam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cam)
ev = cam.ev
SIZE = 756
COLORS = np.array([[1, .19, .12], [.04, .85, .25], [.14, .4, 1]], np.float32)


def map_native(xy, k, p):
    xy = np.asarray(xy)
    return (xy - k[:2, 2]) * np.diag(p["K_reference"])[:2] / np.diag(k)[:2] + p["K_reference"][:2, 2]


def observations(pack, archive, scene, split):
    poses, intr, ids = ev.M03.load_cameras(scene / "prior_camera.json")
    if pack is None:
        data = cam.read(archive / "face_opt/audit/landmarks.json")[split]
        raw_ids = cam.read(scene / "meta.json")["picked_idx"]
    elif pack["kind"] == "static_synthetic":
        data = {r["id"]: r["points"] for r in cam.read(Path(pack["landmarks"]))["views"].values()}
    else:
        folder = Path(pack["n1_eval"]).parent.parent / "protocol_v1"
        old = next(p for p in cam.read(folder / "protocol.json")["packages"] if p["scene"] == pack["name"])
        review = cam.read(folder / "roi_review.json")["views"][pack["name"]]
        data = {r["raw_frame_idx"]: r["landmark_qa"]["points"] for r in old["frames"]
                if review[r["camera_id"]]["status"] == "accepted"}
    rows = []
    for i, (stem, k) in enumerate(zip(ids, intr)):
        path = scene / "images" / (stem + ".png")
        layout = ev.evaluation_geometry(k, Image.open(path).size, k)
        if pack is None:
            raw = raw_ids[i]
            lm = data[stem]
            bounds = np.array(lm["bbox"]).reshape(2, 2)
            tip = np.array(lm["pts"]["nose_tip"])
            if split == "target":  # target landmarks were annotated on the original photograph
                tip, bounds = map_native(tip, k, layout), map_native(bounds, k, layout)
        else:
            slot = "target" if split == "target" else "source3"
            raw = next(f["raw_id"] for f in pack["frames"] if f["scene_slots"].get(slot) == stem)
            pts = data.get(raw)
            tip = bounds = None
            if pts is not None and len(pts):
                pts = np.asarray(pts)
                if pack["kind"] == "static_synthetic":
                    pts = map_native(pts, k, layout)
                # FaceMesh 1 is the nose tip. Bound the face including chin and forehead.
                tip = pts[1]
                bounds = np.stack([pts.min(axis=0), pts.max(axis=0)])
        hull = None if pack is None or pts is None or not len(pts) else pts
        rows.append(dict(raw=raw, stem=stem, path=path, layout=layout, k=layout["K_reference"],
                         pose=poses[i], tip=tip, bounds=bounds, hull=hull))
    return rows


def project(points, pose, k):
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    uvz = local @ k.T
    return uvz[:, :2] / uvz[:, 2:], local[:, 2]


def gaussians(data):
    return {k: data[k].astype(np.float32, copy=False).reshape(-1, n) if n > 1
            else data[k].astype(np.float32, copy=False).reshape(-1)
            for k, n in dict(means=3, scales=3, quats=4, sh=3, opacities=1).items()}


def merge_within_sources(g,weights,source,archive):
    from src.models.models.rasterization import GaussianSplatRenderer
    voxel=yaml.safe_load((archive/"config.yaml").read_text())["wrapper"]["model"]["gs_voxel_size"]
    groups=[]
    for i in range(3):
        sel=source==i
        inputs={k:torch.from_numpy(v[sel]).cuda()[None] for k,v in dict(g,weights=weights).items()}
        keep=torch.ones_like(inputs["weights"],dtype=torch.bool)
        group=GaussianSplatRenderer.merge_voxels(SimpleNamespace(voxel_size=voxel),inputs,keep)
        groups.append({k:v[0].cpu().numpy() for k,v in group.items()})
    return {k:np.concatenate([group[k] for group in groups]) for k in g}


def write_image(path, rgb):
    Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255 + .5)).save(path)


def trace(protocol, out, archive, name, prepare_faces=False):
    run = cam.read(archive / "run.json")
    scene = Path(run["input_path"])
    assert run["resolution"] == [SIZE, SIZE]
    pack = None if name == "jiajia" else next(p for p in cam.read(protocol / "manifest.json")["packages"]
                                              if p["name"] == name)
    tar = scene.with_name(scene.name + "_t2") if pack is None else Path(pack["target_scene"])
    sources = observations(pack, archive, scene, "source")
    targets = observations(pack, archive, tar, "target")
    assert not ({r["raw"] for r in sources} & {r["raw"] for r in targets})
    replay = out / "replay" / scene.name
    if prepare_faces:
        requests = []
        for split, frames in [("source",sources),("target",targets)]:
            for row in frames:
                if row["bounds"] is None:
                    continue
                key = f'{split}_{row["raw"]}'
                if split == "source":
                    file = replay/"rgb"/(row["stem"]+".png")
                else:
                    file = out/(key+"_canvas.png")
                    write_image(file,ev.M21.official_preprocess_rgb(row["path"],row["layout"]))
                lo,hi=row["bounds"]
                margin=max(45,int(max(hi-lo)*.6))
                requests.append(dict(key=key, file=str(file), bbox=np.r_[
                    np.maximum(0,np.floor(lo-margin)),np.minimum(SIZE,np.ceil(hi+margin))].astype(int).tolist()))
        cam.save(out/"face_requests.json",requests)
        return
    if pack is None or pack["kind"] != "static_synthetic":
        verified=cam.read(out/"landmarks_canvas.json")
        for split, frames in [("source",sources),("target",targets)]:
            for row in frames:
                if row["bounds"] is None:
                    continue
                points=verified.get(f'{split}_{row["raw"]}')
                if points is None:
                    row.update(tip=None,bounds=None,hull=None)
                    continue
                pts=np.asarray(points)
                row.update(tip=pts[1],bounds=np.stack((pts.min(axis=0),pts.max(axis=0))),hull=pts)
                # Profile-tip QA on the exact replay RGB: the detected point is in the background.
                if split=="source" and (name,row["raw"]) in {("real_gray_0221a3dd",43),("real_black_3a8b3cc7",196)}:
                    row["tip"]=None
    fit = ev.M15.estimate_sim3(replay, scene)
    src_cam, src_k, stems = ev.M03.load_cameras(replay / "cameras.json")
    assert stems == [r["stem"] for r in sources]
    with np.load(out / "pre.npz") as data:
        g = gaussians(data)
        pixel, colors, weights = data["pixel"], data["base_rgb"].astype(np.float32), data["weights"]
    source = pixel // SIZE**2
    within=merge_within_sources(g,weights,source,archive)
    with np.load(out / "merged.npz") as data:
        merged = gaussians(data)
    exported = ev.M03.load_ply(replay / "gaussians.ply")
    archived = ev.M03.load_ply(archive / "gaussians.ply")
    archive_fit = ev.M15.estimate_sim3(archive, scene)
    tips, labels = [], []
    for i, row in enumerate(sources):
        if row["tip"] is None:
            continue
        x, y = np.rint(row["tip"]).astype(int)
        ix = i * SIZE**2 + y * SIZE + x
        index = np.searchsorted(pixel, ix)
        if index >= len(pixel) or pixel[index] != ix:
            continue
        xyz = g["means"][index]
        uv, _ = project(xyz[None], src_cam[i], src_k[i])
        assert np.linalg.norm(uv[0] - [x, y]) < .1, "Source-pixel identity mismatch"
        tips.append(xyz)
        labels.append(i)
        rgb = np.asarray(Image.open(replay / "rgb" / (row["stem"] + ".png")).convert("RGB")).copy()
        cv2.circle(rgb, (x, y), 4, tuple((COLORS[i] * 255).astype(int).tolist()), 1, cv2.LINE_AA)
        lo, hi = np.rint(row["bounds"]).astype(int)
        Image.fromarray(rgb).crop((max(0,lo[0]-8), max(0,lo[1]-8), min(SIZE,hi[0]+9),
                                   min(SIZE,hi[1]+9))).save(out / f"source{i+1}.png")
    report = dict(scene=str(scene), target_scene=str(tar), archive=str(archive), source_ids=[r["raw"] for r in sources],
                  counts=dict(pre=len(pixel), merged=len(merged["means"]), exported=len(exported["means"])),
                  accepted_nose_source_slots=[i+1 for i in labels], frames=[])
    def source_residual(xyz):
        return np.concatenate([project(xyz[None],src_cam[i],src_k[i])[0][0]-sources[i]["tip"]
                               for i in labels])

    common_point = None
    if len(labels)>=2:
        common_point=least_squares(source_residual,np.mean(tips,axis=0)).x
        report["single_static_nose_fit"]=dict(source_slots=[i+1 for i in labels],
            source_residual_px=np.linalg.norm(source_residual(common_point).reshape(-1,2),axis=1).tolist())
    p3path = archive / "face_opt/p3/gaussians_p3_300.ply"
    p3 = ev.M03.load_ply(p3path) if pack is None and p3path.exists() else None
    for row in targets:
        result = dict(target=row["raw"], stem=row["stem"])
        report["frames"].append(result)
        if row["bounds"] is None:
            result["face"] = "N/A: missing accepted landmarks"
            continue
        dest = out / str(row["raw"])
        dest.mkdir(exist_ok=True)
        gt = ev.M21.official_preprocess_rgb(row["path"], row["layout"])
        pose = ev.M15.apply_sim3(row["pose"], *fit[:3])
        original_pose = ev.M15.apply_sim3(row["pose"], *archive_fit[:3])
        k = row["k"]
        lo, hi = row["bounds"]
        margin = max(6, round(max(hi-lo)*.10))
        x, y = np.maximum(0, np.floor(lo).astype(int)-margin)
        xx, yy = np.minimum(SIZE, np.ceil(hi).astype(int)+margin+1)
        box = np.s_[y:yy, x:xx]
        pad=max(xx-x,yy-y)//2
        context=np.s_[max(0,y-pad):min(SIZE,yy+pad),max(0,x-pad):min(SIZE,xx+pad)]
        result.update(box_xyxy=[int(v) for v in (x,y,xx,yy)], gt_tip_xy=row["tip"].tolist(), metrics={})

        def render(label, gs, camera=pose):
            rgb, alpha = ev.M03.render(gs, camera, k, SIZE, SIZE, background=0, device="cuda", eps2d=.01)
            img, a = rgb[0][box], alpha[0,:,:,0][box]
            write_image(dest / (label+".png"), img)
            if label in ("source1","source2","source3","pre","weighted","merged","export","provenance"):
                write_image(dest / ("context_"+label+".png"),rgb[0][context])
            result["metrics"][label] = dict(rgb_mae_255=float(np.abs(img-gt[box]).mean()*255),
                                            coverage=float((a>.5).mean()))
            return img

        write_image(dest / "gt.png", gt[box])
        write_image(dest / "context_gt.png",gt[context])
        pre = render("pre",g)
        weighted = render("weighted",dict(g, opacities=weights))
        same_src = render("within",within)
        post = render("merged",merged)
        final = render("export",exported)
        old = render("archive",archived,original_pose)
        result["comparison_mae_255"] = {label:float(np.abs(a-b).mean()*255) for label,a,b in [
            ("replay_archive",final,old),("pre_weighted",pre,weighted),("weighted_merge",weighted,post),
            ("weighted_within",weighted,same_src),("within_merge",same_src,post),
            ("pre_merge",pre,post),("merge_export",post,final)]}
        if p3 is not None:
            render("p3",p3,original_pose)
        # Per-Gaussian constant source colour in SH0; blending reveals the visible sources.
        render("provenance",dict(g, opacities=weights, sh=(COLORS[source]-.5)/.28209479177387814))
        del result["metrics"]["provenance"]
        buffers = []
        for i in range(3):
            selected = source == i
            render(f"source{i+1}",{key:val[selected] for key,val in g.items()})
            render(f"drop{i+1}",{key:val[~selected] for key,val in g.items()})
            render(f"weighted_drop{i+1}",{key:val[~selected] for key,val in dict(g,opacities=weights).items()})
            # Hard z-buffer of source RGB/centres in the same crop, no Gaussian attributes.
            uv, z = project(g["means"][selected],pose,k)
            u,v = np.rint(uv).astype(int).T
            keep = (z>.01)&(u>=x)&(v>=y)&(u<xx)&(v<yy)
            idx = (v[keep]-y)*(xx-x)+(u[keep]-x)
            order = np.lexsort((z[keep],idx))
            flat, first = np.unique(idx[order],return_index=True)
            win = order[first]
            buf = np.full((yy-y,xx-x),np.inf)
            buf.flat[flat] = z[keep][win]/fit[0]
            rgb = np.zeros((yy-y,xx-x,3),np.float32)
            rgb.reshape(-1,3)[flat] = colors[selected][keep][win]
            write_image(dest / f"points{i+1}.png",rgb)
            # Count a ray only when its frontmost source point originates inside the
            # source face. Otherwise background mixed with a nose can give metre-scale gaps.
            face = np.zeros((SIZE,SIZE),np.uint8)
            if sources[i]["hull"] is not None:
                cv2.fillConvexPoly(face,cv2.convexHull(np.rint(sources[i]["hull"]).astype(np.int32)),1)
            front_face = face.ravel()[(pixel[selected][keep][win] % SIZE**2)].astype(bool)
            buf.flat[flat[~front_face]] = np.inf
            buffers.append(buf)
        uv, z = project(np.asarray(tips).reshape(-1,3),pose,k)
        in_view = (z>.01)&(uv.min(axis=1)>=0)&(uv.max(axis=1)<SIZE)
        pairs = list(itertools.combinations(np.flatnonzero(in_view),2))
        result["nose"] = dict(sources=[sources[i]["raw"] for i in labels],xy=uv.tolist(),z_m=(z/fit[0]).tolist(),
                              in_front_and_in_frame=in_view.tolist(),
                              pairs_px={f"{labels[a]+1}-{labels[b]+1}":float(np.linalg.norm(uv[a]-uv[b]))
                                        for a,b in pairs})
        if common_point is not None:
            common_xy=project(common_point[None],pose,k)[0][0]
            result["single_static_nose_target_error_px"]=float(np.linalg.norm(common_xy-row["tip"]))
        overlay = np.uint8(gt[box]*255).copy()
        cv2.drawMarker(overlay,tuple(np.rint(row["tip"]-[x,y]).astype(int)),(255,255,255),cv2.MARKER_CROSS,11,1)
        for pos,i in zip(uv[in_view], np.asarray(labels)[in_view]):
            cv2.circle(overlay,tuple(np.rint(pos-[x,y]).astype(int)),3,tuple((COLORS[i]*255).astype(int).tolist()),1,cv2.LINE_AA)
        Image.fromarray(overlay).save(dest / "nose.png")
        full=np.uint8(gt*255).copy()
        cv2.drawMarker(full,tuple(np.rint(row["tip"]).astype(int)),(255,255,255),cv2.MARKER_CROSS,11,1)
        for pos,i in zip(uv[in_view],np.asarray(labels)[in_view]):
            cv2.circle(full,tuple(np.rint(pos).astype(int)),3,tuple((COLORS[i]*255).astype(int).tolist()),1,cv2.LINE_AA)
        Image.fromarray(full[context]).save(dest/"context_nose.png")
        gy,gx=np.ogrid[y:yy,x:xx]
        rad=max(3,round((hi[0]-lo[0])*.08))
        roi=(gx-row["tip"][0])**2+(gy-row["tip"][1])**2<=rad**2
        zbuf=np.asarray(buffers)
        overlap=roi&(np.isfinite(zbuf).sum(axis=0)>=2)
        result["ray_depth"]=dict(radius_px=rad,n=int(overlap.sum()),roi=int(roi.sum()))
        if overlap.sum()>=12 and overlap.sum()>=.2*roi.sum():
            vals=zbuf[:,overlap]
            spread=np.where(np.isfinite(vals),vals,-np.inf).max(axis=0)-vals.min(axis=0)
            result["ray_depth"]["median_spread_mm"]=float(np.median(spread)*1000)
        ala=ev.M21.region_masks(row["hull"],(SIZE,SIZE))["nose_ala"][box]
        valid=ala&(np.isfinite(zbuf).sum(axis=0)>=2)
        result["ala_ray_depth"]=dict(n=int(valid.sum()),roi=int(ala.sum()))
        if valid.sum()>=12 and valid.sum()>=.2*ala.sum():
            vals=zbuf[:,valid]
            spread=np.where(np.isfinite(vals),vals,-np.inf).max(axis=0)-vals.min(axis=0)
            result["ala_ray_depth"]["median_spread_mm"]=float(np.median(spread)*1000)
    cam.save(out/"trace.json",report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("protocol","out","archive"):
        ap.add_argument(arg,type=Path)
    ap.add_argument("name")
    ap.add_argument("--prepare-faces",action="store_true")
    args=ap.parse_args()
    trace(args.protocol,args.out,args.archive,args.name,args.prepare_faces)
