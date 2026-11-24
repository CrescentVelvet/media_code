"""Three-view rigid-face diagnostic and ground-truth depth substitution."""
import argparse
import importlib.util
import itertools
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location("align", Path(__file__).with_name("32_align_face.py"))
align = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align)
trace, ev, cam, S = align.trace, align.ev, align.cam, align.SIZE
FIT = (33, 263, 133, 362, 205, 425, 50, 280, 116, 345, 98, 327)
TEST = {"nose": (1, 4), "jaw": (152, 175, 199, 172, 397)}
ARMS = ("A", "R0", "Rtri")


def robust_rigid(src, dst, scale):
    """Kabsch with fixed scale and bounded influence of 3D correspondence outliers."""
    weight = np.ones(len(src))
    for _ in range(6):
        weight /= weight.sum()
        a, b = weight @ src, weight @ dst
        u, singular, vt = np.linalg.svd(((src-a)*weight[:, None]).T @ (dst-b))
        if singular[1] < singular[0]*.01 or singular[0] < 1e-10:
            return None
        sign = np.linalg.det(vt.T @ u.T)
        rot = vt.T @ np.diag([1, 1, sign]) @ u.T
        offset = b - rot @ a
        residual = np.linalg.norm(src @ rot.T + offset - dst, axis=1)/scale
        weight = 1 / np.maximum(residual/.008, 1)
    return rot, offset


def sources_landmarks(rows, pixels, xyz, pose, name):
    mapping = np.full(3*S*S, -1, int)
    mapping[pixels] = np.arange(len(pixels))
    maps = []
    for i, row in enumerate(rows):
        frame = {}
        maps.append(frame)
        if row["hull"] is None:
            continue
        pts = row["hull"]
        interior = cv2.distanceTransform(align.hull_mask(pts), cv2.DIST_L2, 5)
        # Profile landmarks on the foreshortened hemiface may be inferred behind the nose.
        # Check horizontal span from the bridge using cheeks and outer eyes, all source-only.
        halves = ((33,133,205,50,116,98),(263,362,425,280,345,327))
        spans = np.array([np.median(abs(pts[list(side[:5]),0]-pts[6,0])) for side in halves])
        hidden = set(halves[np.argmin(spans)]) if spans.min()<.45*spans.max() else set()
        for idx in FIT + TEST["nose"] + TEST["jaw"]:
            if (name, row["raw"]) in align.BAD_TIP and idx in (1,4,98,327):
                continue
            q, valid = align.pixel_index(row["hull"][idx])
            ix = mapping[i*S*S+q]
            if idx in FIT and (idx in hidden or interior.flat[q]<2):
                continue
            if valid and ix >= 0 and (xyz[ix]-pose[i,:3,3]) @ pose[i,:3,2] > 0:
                frame[idx] = xyz[ix].astype(float)
    return maps


def triangulated_references(rows, maps, pose, k):
    cameras = [K @ np.linalg.inv(p)[:3] for K,p in zip(k,pose)]
    refs = {}
    for idx in FIT:
        candidates = []
        for i,j in align.PAIRS:
            if idx not in maps[i] or idx not in maps[j]:
                continue
            a,b = (rows[n]["hull"][idx] for n in (i,j))
            h = cv2.triangulatePoints(cameras[i],cameras[j],a[:,None],b[:,None])[:,0]
            if abs(h[3]) < 1e-9:
                continue
            point = h[:3]/h[3]
            rays = point-pose[[i,j],:3,3]
            rays /= np.linalg.norm(rays,axis=1)[:,None]
            angle = np.degrees(np.arccos(np.clip(rays[0]@rays[1],-1,1)))
            projections = [trace.project(point[None],pose[n],k[n]) for n in (i,j)]
            if (4 <= angle <= 100 and all(z[0] > 0 and np.linalg.norm(uv[0]-obs)<=5
                    for (uv,z),obs in zip(projections,(a,b)))):
                candidates.append(point)
        if candidates:
            refs[idx] = np.median(candidates,axis=0)
    return refs


def rigid_fit(rows, maps, g, pixels, weights, pose, k, scale, archive, out):
    voxel_cfg = yaml.safe_load((archive/"config.yaml").read_text())["wrapper"]
    refs = triangulated_references(rows,maps,pose,k)
    results = {}
    for arm, target in [("R0",maps[0]),("Rtri",refs)]:
        xyz, quat = g["means"].copy(), g["quats"].copy()
        details = []
        for i, row in enumerate(rows):
            ids = sorted(set(maps[i]) & set(target) & set(FIT))
            rt = None
            status = "insufficient"
            if i == 0 and arm=="R0":
                status = "fixed_reference"
            elif len(ids)>=5:
                rt = robust_rigid(np.array([maps[i][p] for p in ids]),np.array([target[p] for p in ids]),scale)
                status = "fit" if rt is not None else "degenerate"
            rot, off = rt if rt is not None else (np.eye(3),np.zeros(3))
            mask = align.hull_mask(row["hull"])
            use = (pixels//S**2==i)&(mask.ravel()[pixels%S**2]>0)
            if rt is not None:
                xyz[use] = xyz[use] @ rot.T + off
                # gsplat stores wxyz; scipy composes rotations in xyzw convention.
                q = Rotation.from_quat(quat[use][:,[1,2,3,0]])
                quat[use] = (Rotation.from_matrix(rot)*q).as_quat()[:,[3,0,1,2]]
            move = np.linalg.norm(xyz[use]-g["means"][use],axis=1)
            details.append(dict(raw=row["raw"],status=status,ids=ids,rotation=rot.tolist(),offset=off.tolist(),
                                angle_deg=float(np.degrees(Rotation.from_matrix(rot).magnitude())),
                                median_move_mm=align.median(move*1000/scale)))
        merged = align.gaussian_merge(dict(g,means=xyz,quats=quat),weights,voxel_cfg["model"]["gs_voxel_size"])
        keep = merged["opacities"]>=voxel_cfg["infer_opacity_thresh"]
        align.save_ply(out/(arm+".ply"),{key:v[keep] for key,v in merged.items()})
        results[arm] = details
    return results


def geometry(frame,row):
    z,ok = ev.M21.load_depth(frame["depth_path"])
    ok &= np.asarray(Image.open(frame["path"]).convert("RGBA"))[:,:,3]==255
    return tuple(ev.M21.remap_nearest(v,row["layout"]) for v in (z,ok))


def raster(points, colors, pose, k):
    uv,z = trace.project(points,pose,k)
    q,ok = align.pixel_index(uv)
    ok &= z>0
    q,z,colors = q[ok],z[ok],colors[ok]
    order = np.argsort(z)
    flat, first = np.unique(q[order],return_index=True)
    depth = np.full(S*S,np.inf)
    rgb = np.zeros((S*S,3),np.float32)
    depth[flat], rgb[flat] = z[order[first]],colors[order[first]]
    return depth.reshape(S,S), rgb.reshape(S,S,3)


def depth_clouds(pack,src,pose,k,scale,g,pixels,colors,dp_dir,out):
    views = {name:[] for name in ("GT","SA","DP2","DP1")}
    indices = {"GT":[], "DP1":[]}
    for i,row in enumerate(src):
        fr = next(f for f in pack["frames"] if f["raw_id"]==row["raw"])
        truth, valid = geometry(fr,row)
        dp = ev.M21.remap_nearest(np.load(dp_dir/f"{i:02}.npz")["depth"],row["layout"])
        single = out/"dp_input/single_frame/depth"/f"{i:02}.npz"
        one = ev.M21.remap_nearest(np.load(single)["depth"],row["layout"])
        take = np.flatnonzero((pixels//S**2==i)&valid.ravel()[pixels%S**2])
        pix = pixels[take]%S**2
        local = (g["means"][take]-pose[i,:3,3]) @ pose[i,:3,:3]
        ray = np.c_[pix%S,pix//S,np.ones(len(pix))] @ np.linalg.inv(k[i]).T
        assert np.quantile(np.linalg.norm(ray*local[:,2:]-local,axis=1),.99) < 1e-4
        for arm,z in [("GT",truth.ravel()[pix]*scale),("SA",local[:,2]),("DP2",dp.ravel()[pix]*scale),
                      ("DP1",one.ravel()[pix]*scale)]:
            ok = np.isfinite(z)&(z>0)
            xyz = (ray[ok]*z[ok,None]) @ pose[i,:3,:3].T + pose[i,:3,3]
            views[arm].append((xyz,colors[take[ok]]))
            if arm in indices:
                indices[arm].append(pixels[take[ok]])
    control = {}
    for arm, ix in indices.items():
        maps = sources_landmarks(src,np.concatenate(ix),np.concatenate([a for a,_ in views[arm]]),pose,pack["name"])
        adjusted = []; control[arm]=[]
        for i,(xyz,rgb) in enumerate(views[arm]):
            ids=sorted(set(FIT)&set(maps[i])&set(maps[0]))
            rt=None if i==0 or len(ids)<5 else robust_rigid(np.array([maps[i][v] for v in ids]),
                                                            np.array([maps[0][v] for v in ids]),scale)
            points=xyz.copy()
            use=align.hull_mask(src[i]["hull"]).ravel()[ix[i]%S**2]>0
            if rt is not None:
                rot,off=rt
                points[use]=xyz[use]@rot.T+off
            control[arm].append(dict(fit=rt is not None,n=len(ids),
                median_move_mm=align.median(np.linalg.norm(points[use]-xyz[use],axis=1)*1000/scale)))
            adjusted.append((points,rgb))
        views[arm+"_R0"]=adjusted
    cam.save(out/"point_rigid_control.json",control)
    for arm in ("GT","DP1","DP1_R0"):
        # One inspectable foreground cloud per arm, native RGB plus an integer source slot.
        layers=views[arm]
        vertex=np.empty(sum(len(x) for x,_ in layers),dtype=[
            (n,"f4") for n in ("x","y","z")]+[(n,"u1") for n in ("red","green","blue","source_slot")])
        start=0
        for i,(xyz,rgb) in enumerate(layers):
            block=vertex[start:start+len(xyz)]
            for key,column in zip(("x","y","z"),(xyz/scale).T):
                block[key]=column
            for key,column in zip(("red","green","blue"),np.rint(255*np.clip(rgb,0,1)).T):
                block[key]=column
            block["source_slot"]=i+1
            start+=len(xyz)
        PlyData([PlyElement.describe(vertex,"vertex")]).write(str(out/(arm+"_points_m.ply")))
    return views


def masks_and_gt(pack,row):
    fr = next(f for f in pack["frames"] if f["raw_id"]==row["raw"])
    truth,valid = geometry(fr,row)
    pts = next(v["points"] for v in cam.read(Path(pack["landmarks"]))["views"].values() if v["id"]==row["raw"])
    masks = ev.M21.region_masks(np.asarray(pts),tuple(reversed(fr["processed_wh"])))
    return truth,{key:valid & ev.M21.remap_nearest(masks[key],row["layout"])
                  for key in ("nose_tip","nose_ala","face_tight_landmark")}


def compare_clouds(clouds,pose,k,scale,truth,masks,box,sub):
    depths, rows = {}, {}
    for arm, sources in clouds.items():
        layers = [raster(x,c,pose,k) for x,c in sources]
        d = np.stack([v[0] for v in layers])/scale
        winner = np.argmin(d,axis=0)
        rgb = np.take_along_axis(np.stack([v[1] for v in layers]),winner[None,:,:,None],axis=0)[0]
        trace.write_image(sub/("points_"+arm+".png"),rgb[box])
        if arm in ("GT","DP1","DP1_R0"):
            # Equal-view blend retains farther layers hidden by a nearest-point renderer.
            count=np.maximum(np.isfinite(d).sum(axis=0),1)
            overlay=sum(layer[1] for layer in layers)/count[:,:,None]
            trace.write_image(sub/("overlay_"+arm+".png"),overlay[box])
            if arm=="DP1":
                for i,(_,layer) in enumerate(layers):
                    trace.write_image(sub/f"single_source_{i+1}.png",layer[box])
        depths[arm]=d
    # Primary GT/SA support is independent of DP2 coverage. Each comparison uses paired rays.
    for arm in ("SA","DP2","DP1","DP1_R0","GT_R0"):
        rows[arm]={}
        for region,mask in masks.items():
            a,b=depths[arm],depths["GT"]
            pairs=np.isfinite(a)&np.isfinite(b)
            if arm in ("DP1","DP1_R0"):
                pairs &= np.isfinite(depths["DP1"]) & np.isfinite(depths["DP1_R0"])
            supported=mask & pairs.any(axis=0)
            multi=mask & (pairs.sum(axis=0)>=2)
            stats={"roi":int(mask.sum()),"common":int(supported.sum()),"multi":int(multi.sum())}
            for label,d in [(arm,a),("GT",b)]:
                z=np.min(np.where(pairs,d,np.inf),axis=0)
                p90=float(np.percentile(np.abs(z[supported]-truth[supported])*1000,90)) if supported.any() else None
                span=(np.max(np.where(pairs,d,-np.inf),axis=0)[multi]-z[multi])*1000
                stats[label]=dict(p90_mm=p90,median_span_mm=align.median(span))
            rows[arm][region]=stats
    return rows


def check_landmarks(maps, fits, pose, k, target, scale):
    result={}
    for group,indices in TEST.items():
        groups={arm:{"distance_3d_mm":[],"separation_px":[],"target_error_px":[]} for arm in ARMS}
        for idx in indices:
            observations=[]
            for slot,m in enumerate(maps):
                if idx not in m:
                    continue
                transformed={"A":m[idx]}
                for arm in fits:
                    v=fits[arm][slot]
                    transformed[arm]=np.asarray(v["rotation"])@m[idx]+v["offset"]
                proj={arm:trace.project(x[None],pose,k) for arm,x in transformed.items()}
                if any(z[0]<=0 or (uv[0]<0).any() or (uv[0]>=S).any() for uv,z in proj.values()):
                    continue
                observations.append((transformed,proj))
                for arm in ARMS:
                    groups[arm]["target_error_px"].append(np.linalg.norm(proj[arm][0][0]-target[idx]))
            for (x,p),(y,q) in itertools.combinations(observations,2):
                for arm in ARMS:
                    groups[arm]["distance_3d_mm"].append(np.linalg.norm(x[arm]-y[arm])*1000/scale)
                    groups[arm]["separation_px"].append(np.linalg.norm(p[arm][0][0]-q[arm][0][0]))
        result[group]={arm:{key:align.median(v) for key,v in data.items()} for arm,data in groups.items()}
        result[group]["pair_n"]=len(groups["A"]["separation_px"])
        result[group]["target_n"]=len(groups["A"]["target_error_px"])
    return result


def run(args):
    old=args.trace/args.name
    replay,=(old/"replay").iterdir()
    packs=cam.read(args.protocol/"manifest.json")["packages"]
    pack=None if args.name=="jiajia" else next(p for p in packs if p["name"]==args.name)
    archive=Path(os.environ["JIA_ARCHIVE"]) if not pack else args.protocol/"models"/"sa_pose"/replay.name
    scene=Path(cam.read(archive/"run.json")["input_path"])
    src=align.rows(pack,archive,scene,"source",old)
    poses,ks,ids=ev.M03.load_cameras(replay/"cameras.json")
    assert ids==[r["stem"] for r in src]
    fit=ev.M15.estimate_sim3(replay,scene)[:3]
    with np.load(old/"pre.npz") as data:
        g=trace.gaussians(data)
        pixel,weights,colors=data["pixel"],data["weights"],data["base_rgb"].astype(np.float32)
    out=args.out/args.name
    out.mkdir(parents=True,exist_ok=True)
    maps=sources_landmarks(src,pixel,g["means"],poses,args.name)
    fits=rigid_fit(src,maps,g,pixel,weights,poses,ks,fit[0],archive,out)
    cam.save(out/"source_transforms.json",fits)
    # Freeze all transformations and PLYs before accessing any target landmarks or depth truth.
    synthetic=pack and pack["kind"]=="static_synthetic"
    clouds=depth_clouds(pack,src,poses,ks,fit[0],g,pixel,colors,args.depth/args.name/"dp2/depth",out) if synthetic else None
    tar=Path(pack["target_scene"]) if pack else scene.with_name(scene.name+"_t2")
    targets=align.rows(pack,archive,tar,"target",old)
    previous={r["target"]:r for r in cam.read(old/"trace.json")["frames"]}
    gs={arm:ev.M03.load_ply(replay/"gaussians.ply" if arm=="A" else out/(arm+".ply")) for arm in ARMS}
    records=[]
    for row in targets:
        raw=row["raw"]
        assert raw not in [s["raw"] for s in src]
        record=dict(raw=raw);records.append(record)
        pose=ev.M15.apply_sim3(row["pose"],*fit)
        sub=out/str(raw);sub.mkdir(exist_ok=True)
        gt=ev.M21.official_preprocess_rgb(row["path"],row["layout"])
        if "box_xyxy" not in previous[raw]:
            record["skip"]="no accepted target face"
            continue
        x,y,xx,yy=previous[raw]["box_xyxy"]
        box=np.s_[y:yy,x:xx]
        pad=max(xx-x,yy-y)//2
        ctx=np.s_[max(0,y-pad):min(S,yy+pad),max(0,x-pad):min(S,xx+pad)]
        trace.write_image(sub/"gt.png",gt[ctx])
        trace.write_image(sub/"gt_tight.png",gt[box])
        record["landmarks"]=check_landmarks(maps,fits,pose,row["k"],row["hull"],fit[0])
        if synthetic:
            truth,masks=masks_and_gt(pack,row)
            record["points"]=compare_clouds(clouds,pose,row["k"],fit[0],truth,masks,box,sub)
        record["gaussians"]={}
        ed={}
        for arm,g_arm in gs.items():
            rgb,a=ev.M03.render(g_arm,pose,row["k"],S,S,background=0,device="cuda",eps2d=.01)
            trace.write_image(sub/(arm+".png"),rgb[0][ctx])
            trace.write_image(sub/(arm+"_tight.png"),rgb[0][box])
            record["gaussians"][arm]=dict(mae=float(np.abs(rgb[0][box]-gt[box]).mean()*255),
                                         coverage=float((a[0,:,:,0][box]>.5).mean()))
            if synthetic:
                z,aa=ev.M21.M13.render_expected_depth(g_arm,pose,row["k"],S,S,device="cuda",eps2d=.01)
                ed[arm]=np.where(aa[0,:,:,0]>.01,z[0]/fit[0],np.nan)
        assert abs(record["gaussians"]["A"]["mae"]-previous[raw]["metrics"]["export"]["rgb_mae_255"])<.01
        if synthetic:
            support=np.logical_and.reduce([np.isfinite(v) for v in ed.values()])
            for key,m in masks.items():
                use=m&support
                for arm,z in ed.items():
                    e=np.abs(z[use]-truth[use])*1000
                    record["gaussians"][arm][key]=dict(p90=float(np.percentile(e,90)),mae=float(e.mean()),
                                                        coverage=float(use.sum()/m.sum()))
    cam.save(out/"eval.json",records)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","trace","depth","out"):
        ap.add_argument(key,type=Path)
    ap.add_argument("name")
    run(ap.parse_args())
