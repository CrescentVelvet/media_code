"""Training-free source visibility and local ray-depth alignment on frozen three-view captures."""
import argparse
import importlib.util
import itertools
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

spec = importlib.util.spec_from_file_location("trace", Path(__file__).with_name("31_trace_ghost.py"))
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)
cam, ev, SIZE = trace.cam, trace.ev, trace.SIZE
# Disjoint semantic IDs for alignment and source validation. Nose tip 1 is held out.
FIT = [33,263,133,362,98,327,168,197,205,425,50,280,10,152,61,291,234,454]
CHECK = [1,4,97,326,159,386,105,334,116,345,172,397]
BAD_TIP = {("real_gray_0221a3dd",43),("real_black_3a8b3cc7",196)}
PAIRS = list(itertools.combinations(range(3),2))


def rows(pack, archive, scene, split, root):
    items = trace.observations(pack,archive,scene,split)
    if pack is None or pack["kind"] != "static_synthetic":
        landmarks = cam.read(root/"landmarks_canvas.json")
        for row in items:
            pts = landmarks.get(f'{split}_{row["raw"]}')
            row["hull"] = np.array(pts) if pts is not None and row["bounds"] is not None else None
    return items


def hull_mask(points):
    m=np.zeros((SIZE,SIZE),np.uint8)
    if points is not None:
        cv2.fillConvexPoly(m,cv2.convexHull(np.rint(points).astype(np.int32)),1)
    return m


def pixel_index(uv):
    xy=np.rint(uv).astype(int)
    valid=(xy.min(axis=-1)>=0)&(xy.max(axis=-1)<SIZE)
    return np.clip(xy[...,1],0,SIZE-1)*SIZE+np.clip(xy[...,0],0,SIZE-1),valid


def median(a):
    return float(np.median(a)) if len(a) else None


def epipolar(a,b,pose,k,i,j):
    rel=np.linalg.inv(pose[j])@pose[i]
    x,y,z=rel[:3,3]
    skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    F=np.linalg.inv(k[j]).T@skew@rel[:3,:3]@np.linalg.inv(k[i])
    u=np.c_[a,np.ones(len(a))]; v=np.c_[b,np.ones(len(b))]
    l=u@F.T; m=v@F
    return np.abs((v*l).sum(axis=1))/np.sqrt((l[:,:2]**2+m[:,:2]**2).sum(axis=1).clip(1e-16))


def diagnostics(images,rows_,pose,k,synthetic,pack):
    sift=cv2.SIFT_create(nfeatures=3500)
    features=[]
    for i,(im,r) in enumerate(zip(images,rows_)):
        mask=np.zeros((SIZE,SIZE),np.uint8)
        if synthetic:
            fr=next(f for f in pack["frames"] if f["raw_id"]==r["raw"])
            alpha=np.asarray(Image.open(fr["path"]).convert("RGBA"))[:,:,3]
            mask[ev.M21.remap_nearest(alpha,r["layout"])==0]=255
        else:
            mask[:,:SIZE//5]=255
            mask[:,-SIZE//5:]=255
            mask[hull_mask(r["hull"])>0]=0
        pts,desc=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),mask)
        features.append((np.array([p.pt for p in pts]),desc))
    stats=[]
    for i,j in PAIRS:
        x,d=features[i]; y,e=features[j]
        errors=[]
        if d is not None and e is not None and min(len(d),len(e))>=2:
            def matches(d1,d2):
                return {(m.queryIdx,m.trainIdx) for m,n in cv2.BFMatcher().knnMatch(d1,d2,k=2)
                        if m.distance<.72*n.distance}
            good=sorted(matches(d,e)&{(b,a) for a,b in matches(e,d)})
            if good:
                a,b=np.array(good).T
                errors=epipolar(x[a],y[b],pose,k,i,j)
        lm1,lm2=rows_[i]["hull"],rows_[j]["hull"]
        face=[] if lm1 is None or lm2 is None else epipolar(lm1[CHECK],lm2[CHECK],pose,k,i,j)
        stats.append(dict(slots=[i+1,j+1],background_n=len(errors),background_sampson_px=median(errors),
                          face_n=len(face),face_sampson_px=median(face)))
    return stats


def reliable(i,idx,rows_,dist,name):
    pts=rows_[i]["hull"]
    if pts is None or ((name,rows_[i]["raw"]) in BAD_TIP and idx in (1,4,98,97,326,327)):
        return False
    q,ok=pixel_index(pts[idx])
    return bool(ok and dist[i].flat[q]>=2)


def fit_anchors(name,rows_,dist,pose,k,depth,scale):
    camera=[K@np.linalg.inv(c)[:3] for K,c in zip(k,pose)]
    result=[[] for _ in rows_]
    for idx in FIT:
        candidates=[[] for _ in rows_]
        for i,j in PAIRS:
            if not all(reliable(v,idx,rows_,dist,name) for v in (i,j)):
                continue
            u,v=rows_[i]["hull"][idx],rows_[j]["hull"][idx]
            h=cv2.triangulatePoints(camera[i],camera[j],u[:,None],v[:,None])[:,0]
            if abs(h[3])<1e-9:
                continue
            point=h[:3]/h[3]
            rays=point-pose[[i,j],:3,3]
            ray=rays/np.linalg.norm(rays,axis=1)[:,None]
            angle=np.degrees(np.arccos(np.clip(ray[0]@ray[1],-1,1)))
            proj=[trace.project(point[None],pose[n],k[n]) for n in (i,j)]
            if not 4<=angle<=100 or max(np.linalg.norm(p[0][0]-q) for p,q in zip(proj,(u,v)))>5:
                continue
            if min(p[1][0]/scale for p in proj)<.1:
                continue
            for n,(xy,(_,z)) in zip((i,j),zip((u,v),proj)):
                q,_=pixel_index(xy)
                own=depth[n].flat[q]
                if own>0 and abs(z[0]-own)<.2*own:
                    candidates[n].append(float((z[0]-own)/scale))
        for i,c in enumerate(candidates):
            if c:
                result[i].append((idx,median(c)))
    return result


def gaussian_merge(g,weights,voxel):
    from src.models.models.rasterization import GaussianSplatRenderer
    values={key:torch.from_numpy(a).cuda()[None] for key,a in dict(g,weights=weights).items()}
    keep=torch.ones_like(values["weights"],dtype=torch.bool)
    out=GaussianSplatRenderer.merge_voxels(SimpleNamespace(voxel_size=voxel),values,keep)
    return {key:a[0].cpu().numpy() for key,a in out.items()}


def save_ply(path,g):
    from src.utils.save_utils import save_gs_ply
    args=[torch.from_numpy(g[k]) for k in ("means","scales","quats","sh","opacities")]
    save_gs_ply(path,*args,quantile_threshold=None)


def fit(args,pack,old,archive,scene,replay,dest):
    src=rows(pack,archive,scene,"source",old)
    poses,k,stems=ev.M03.load_cameras(replay/"cameras.json")
    assert stems==[r["stem"] for r in src]
    scale=ev.M15.estimate_sim3(replay,scene)[0]
    with np.load(old/"pre.npz") as d:
        g=trace.gaussians(d)
        weights,pixel=d["weights"],d["pixel"]
    slots=pixel//SIZE**2
    uv=np.stack((pixel%SIZE,pixel//SIZE%SIZE),axis=1)
    depth=np.zeros((3,SIZE,SIZE),np.float32)
    masks=[hull_mask(r["hull"]) for r in src]
    dist=[cv2.distanceTransform(m,cv2.DIST_L2,5) for m in masks]
    images=[np.asarray(Image.open(replay/"rgb"/(s+".png")).convert("RGB")) for s in stems]
    for i in range(3):
        sel=slots==i
        depth[i].flat[pixel[sel]%SIZE**2]=(g["means"][sel]-poses[i,:3,3])@poses[i,:3,2]
    anchors=fit_anchors(args.name,src,dist,poses,k,depth,scale)
    new_xyz=g["means"].copy()
    multiplier=np.ones(len(pixel),np.float32)
    info=[]
    for i in range(3):
        ix=np.flatnonzero((slots==i)&(masks[i].ravel()[pixel%SIZE**2]>0))
        fr=dict(raw=src[i]["raw"],face_n=len(ix),anchors=anchors[i],B_changed=0,C_changed=0)
        info.append(fr)
        if not len(ix):
            continue
        pt=g["means"][ix]; pix=uv[ix]; q=pixel[ix]%SIZE**2
        width=max(1,np.ptp(src[i]["hull"][:,0]))
        taper=np.clip((dist[i].ravel()[q]-1)/max(3,.07*width),0,1)
        conflict=np.zeros(len(ix)); agreement=np.zeros(len(ix),bool)
        for j in range(3):
            if j==i:
                continue
            proj,z=trace.project(pt,poses[j],k[j])
            mapped,ok=pixel_index(proj)
            other=depth[j].ravel()[mapped]
            visible=ok&(z>0)&(other>0)&(dist[j].ravel()[mapped]>=2)
            # A point in front of another observed face contradicts its measured free space.
            delta=(other-z)/np.maximum(other,1e-6)
            agreement|=visible&(np.abs(delta)<.015)
            conflict=np.maximum(conflict,np.where(visible,np.clip((delta-.02)/.05,0,1),0))
        multiplier[ix]=1-.75*taper*conflict*(~agreement)
        fr["B_changed"]=int((multiplier[ix]<.9999).sum())
        if len(anchors[i])>=3:
            ids,dz=zip(*anchors[i])
            xy=src[i]["hull"][list(ids)]
            sigma=max(5,.16*width)
            kernel=np.exp(-((pix[:,None]-xy)**2).sum(axis=2)/(2*sigma**2))
            d=(kernel@np.clip(dz,-.025,.025))/np.maximum(kernel.sum(axis=1),1e-12)
            z=depth[i].ravel()[q]
            shift=np.clip(d*scale,-.04*z,.04*z)*taper*np.minimum(1,kernel.max(axis=1)*2)
            new_xyz[ix]=pt+(pt-poses[i,:3,3])*(shift/np.maximum(z,1e-8))[:,None]
            fr["C_changed"]=int((np.abs(shift)/scale>.0001).sum())
            fr["C_p90_abs_mm"]=float(np.percentile(np.abs(shift/scale)*1000,90))
    config=yaml.safe_load((archive/"config.yaml").read_text())["wrapper"]
    voxel=config["model"]["gs_voxel_size"]; cut=config["infer_opacity_thresh"]
    with np.load(old/"merged.npz") as d:
        frozen=trace.gaussians(d)
    exported=ev.M03.load_ply(replay/"gaussians.ply")
    keep=frozen["opacities"]>=cut
    assert np.array_equal(frozen["means"][keep],exported["means"])
    # Use the official grouping to map each input's multiplier onto the fixed P0 Gaussian.
    # Passing it through SH computes sum(w*m)/sum(w) without moving P0 centres in B.
    tmp=gaussian_merge(dict(g,sh=np.repeat(multiplier[:,None],3,axis=1)),weights,voxel)
    assert np.allclose(tmp["means"],frozen["means"],atol=1e-5)
    factor=np.clip(tmp["sh"][keep,0],.25,1)
    factor[np.abs(factor-1)<1e-6]=1
    # Keep P0's loaded attributes for unchanged B points, and use the same opacity cut.
    gb=dict(exported,opacities=exported["opacities"]*factor)
    keep_b=(factor==1)|(gb["opacities"]>=cut)
    save_ply(dest/"B.ply",{key:v[keep_b] for key,v in gb.items()})
    new=gaussian_merge(dict(g,means=new_xyz),weights,voxel)
    keep_c=new["opacities"]>=cut
    save_ply(dest/"C.ply",{key:v[keep_c] for key,v in new.items()})
    idxmap=np.full(3*SIZE**2,-1,int); idxmap[pixel]=np.arange(len(pixel))
    probe=[]
    for idx in CHECK:
        for i in range(3):
            if not reliable(i,idx,src,dist,args.name):
                continue
            p,_=pixel_index(src[i]["hull"][idx]); ix=idxmap[i*SIZE**2+p]
            if ix<0:
                continue
            probe.append(dict(id=idx,slot=i,A=g["means"][ix].tolist(),C=new_xyz[ix].tolist()))
    source_errors={}
    for arm in ("A","C"):
        errors=[]
        for v in probe:
            for j in range(3):
                if j==v["slot"] or not reliable(j,v["id"],src,dist,args.name):
                    continue
                xy,z=trace.project(np.array([v[arm]]),poses[j],k[j])
                if z[0]>0:
                    errors.append(float(np.linalg.norm(xy[0]-src[j]["hull"][v["id"]])))
        source_errors[arm]=dict(n=len(errors),median_px=median(errors))
    cam.save(dest/"fit.json",dict(sources=info,probes=probe,source_check=source_errors,
        camera_pairs=diagnostics(images,src,poses,k,pack is not None and pack["kind"]=="static_synthetic",pack),
        B_modified_exported=int((factor<1).sum()),B_removed=int((~keep_b).sum()),
        A_ply=str(replay/"gaussians.ply"),source_ids=[r["raw"] for r in src]))


def audit_sources(args,pack,old,archive,scene,replay,dest):
    """Post-fit audit. Synthetic RGB-D truth is used for scoring after outputs are frozen."""
    src=rows(pack,archive,scene,"source",old)
    pose,k,stems=ev.M03.load_cameras(replay/"cameras.json")
    scale=ev.M15.estimate_sim3(replay,scene)[0]
    if pack is not None and pack["kind"]=="static_synthetic":
        arms={arm:ev.M03.load_ply(replay/"gaussians.ply" if arm=="A" else dest/(arm+".ply")) for arm in "ABC"}
        scores=[]
        for i,row in enumerate(src):
            if row["hull"] is None:
                continue
            fr=next(f for f in pack["frames"] if f["raw_id"]==row["raw"])
            truth,valid=ev.M21.load_depth(fr["depth_path"])
            valid&=np.asarray(Image.open(fr["path"]).convert("RGBA"))[:,:,3]==255
            points=next(v["points"] for v in cam.read(Path(pack["landmarks"]))["views"].values() if v["id"]==row["raw"])
            mask=ev.M21.region_masks(np.asarray(points),truth.shape)["nose_tip"]
            support=ev.M21.remap_nearest(valid&mask,row["layout"])
            truth=ev.M21.remap_nearest(truth,row["layout"])
            z={}
            for arm,g in arms.items():
                ed,alpha=ev.M21.M13.render_expected_depth(g,pose[i],k[i],SIZE,SIZE,device="cuda",eps2d=.01)
                z[arm]=ed[0]/scale
                support&=(alpha[0,:,:,0]>.01)&np.isfinite(z[arm])
            scores.append(dict(raw=row["raw"],n=int(support.sum()),nose_tip_p90_mm={
                arm:float(np.percentile(np.abs(zz[support]-truth[support])*1000,90)) for arm,zz in z.items()}))
        cam.save(dest/"source_audit.json",dict(depth=scores))
        return
    # Wider original field of view and independent 2D matching; no use of a target pose.
    sift=cv2.SIFT_create(nfeatures=12000)
    feats=[]
    native_k=ev.M03.load_cameras(scene/"prior_camera.json")[1]
    for row in src:
        im=cv2.imread(str(row["path"]),cv2.IMREAD_GRAYSCALE)
        h,w=im.shape
        m=np.zeros_like(im)
        file=archive/"face_opt/masks"/(row["stem"]+"_person.png")
        matte=cv2.imread(str(file),cv2.IMREAD_GRAYSCALE) if file.exists() else None
        if matte is not None and matte.shape==im.shape:
            m[cv2.dilate(matte,np.ones((21,21),np.uint8))<128]=255
        else:
            m[:,:w//4]=255
            m[:,-w//4:]=255
        pts,d=sift.detectAndCompute(im,m)
        uv=np.array([p.pt for p in pts]).reshape(-1,2)
        feats.append((trace.map_native(uv,native_k[len(feats)],row["layout"]),d))
    pairs=[]
    for i,j in PAIRS:
        x,d=feats[i];y,e=feats[j]
        def pairs_one(a,b):
            return {(p.queryIdx,p.trainIdx) for p,q in cv2.BFMatcher().knnMatch(a,b,k=2)
                    if p.distance<.75*q.distance}
        matches=[] if d is None or e is None or min(len(d),len(e))<2 else sorted(
            pairs_one(d,e)&{(b,a) for a,b in pairs_one(e,d)})
        r=dict(slots=[i+1,j+1],mutual_n=len(matches),inlier_n=0,median_px=None)
        if len(matches)>=12:
            a,b=np.asarray(matches).T
            # RANSAC uses the image matches alone. Measure residual against the supplied cameras afterwards.
            cv2.setRNGSeed(32)
            _,keep=cv2.findFundamentalMat(x[a],y[b],cv2.FM_RANSAC,1.5,.999)
            if keep is not None:
                use=keep.ravel().astype(bool)
                r["inlier_n"]=int(use.sum())
                if use.sum()>=12:
                    r["median_px"]=median(epipolar(x[a][use],y[b][use],pose,k,i,j))
        pairs.append(r)
    cam.save(dest/"source_audit.json",dict(native_background=pairs))


def evaluate(args,pack,old,archive,scene,replay,dest):
    fixed=cam.read(dest/"fit.json")
    gs={arm:ev.M03.load_ply(Path(fixed["A_ply"]) if arm=="A" else dest/(arm+".ply")) for arm in "ABC"}
    tar=Path(pack["target_scene"]) if pack else scene.with_name(scene.name+"_t2")
    target=rows(pack,archive,tar,"target",old)
    scale,rot,offset,_=ev.M15.estimate_sim3(replay,scene)
    previous={r["target"]:r for r in cam.read(old/"trace.json")["frames"]}
    frames=[]
    for row in target:
        raw=row["raw"]; assert raw not in fixed["source_ids"]
        sub=dest/str(raw);sub.mkdir(exist_ok=True)
        pose=ev.M15.apply_sim3(row["pose"],scale,rot,offset)
        gt=ev.M21.official_preprocess_rgb(row["path"],row["layout"])
        ref=previous[raw]
        entry=dict(raw=raw,arms={});frames.append(entry)
        rgb={}; alpha={}; ed={}
        for arm,g in gs.items():
            image,a=ev.M03.render(g,pose,row["k"],SIZE,SIZE,background=0,device="cuda",eps2d=.01)
            rgb[arm]=image[0];alpha[arm]=a[0,:,:,0]
            trace.write_image(sub/(arm+"_full.jpg"),rgb[arm])
        if "box_xyxy" not in ref:
            entry["face"]="N/A"
            continue
        x,y,xx,yy=ref["box_xyxy"]; box=np.s_[y:yy,x:xx]
        pad=max(xx-x,yy-y)//2
        context=np.s_[max(0,y-pad):min(SIZE,yy+pad),max(0,x-pad):min(SIZE,xx+pad)]
        outside=np.ones((SIZE,SIZE),bool);outside[box]=False
        trace.write_image(sub/"gt.png",gt[context])
        trace.write_image(sub/"gt_tight.png",gt[box])
        for arm in "ABC":
            img=rgb[arm]
            trace.write_image(sub/(arm+".png"),img[context])
            trace.write_image(sub/(arm+"_tight.png"),img[box])
            entry["arms"][arm]=dict(mae_255=float(np.abs(img[box]-gt[box]).mean()*255),
               coverage=float((alpha[arm][box]>.5).mean()),outside_delta_255=float(np.abs(img[outside]-rgb["A"][outside]).mean()*255))
            if arm!="A":
                trace.write_image(sub/(arm+"_diff_x4.png"),4*np.abs(img[context]-rgb["A"][context]))
        assert abs(entry["arms"]["A"]["mae_255"]-ref["metrics"]["export"]["rgb_mae_255"])<.01
        tips=[p for p in fixed["probes"] if p["id"]==1]
        for arm in "AC":
            xy,z=trace.project(np.array([p[arm] for p in tips]).reshape(-1,3),pose,row["k"])
            xy=xy[(z>0)&(xy.min(axis=1)>=0)&(xy.max(axis=1)<SIZE)]
            entry["arms"][arm]["tip_gap_px"]=max((float(np.linalg.norm(a-b)) for a,b in itertools.combinations(xy,2)),default=None)
        entry["arms"]["B"]["tip_gap_px"]=entry["arms"]["A"]["tip_gap_px"]  # Source point centres are fixed.
        if pack is not None and pack["kind"]=="static_synthetic":
            fr=next(f for f in pack["frames"] if f["raw_id"]==raw)
            z,valid=ev.M21.load_depth(fr["depth_path"])
            valid&=np.asarray(Image.open(fr["path"]).convert("RGBA"))[:,:,3]==255
            native=next(v["points"] for v in cam.read(Path(pack["landmarks"]))["views"].values() if v["id"]==raw)
            masks=ev.M21.region_masks(np.array(native),z.shape)
            remap=lambda a:ev.M21.remap_nearest(a,row["layout"])
            gt_valid=remap(valid);valid=gt_valid.copy();truth=remap(z)
            for arm,g in gs.items():
                zz,aa=ev.M21.M13.render_expected_depth(g,pose,row["k"],SIZE,SIZE,device="cuda",eps2d=.01)
                ed[arm]=zz[0]/scale;valid&=(aa[0,:,:,0]>.01)&np.isfinite(ed[arm])
            for key in ("nose_tip","nose_ala","face_tight_landmark"):
                roi=remap(masks[key])&gt_valid
                region=roi&valid
                for arm in "ABC":
                    delta=1000*np.abs(ed[arm][region]-truth[region])
                    entry["arms"][arm][key]=dict(n=len(delta),common_fraction=float(region.sum()/roi.sum()),
                        mae=float(delta.mean()),p90=float(np.percentile(delta,90)))
    cam.save(dest/"eval.json",frames)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["fit","evaluate","audit"])
    for key in ("protocol","trace","output"):
        parser.add_argument(key,type=Path)
    parser.add_argument("name")
    args=parser.parse_args()
    old=args.trace/args.name
    packs=cam.read(args.protocol/"manifest.json")["packages"]
    pack=None if args.name=="jiajia" else next(p for p in packs if p["name"]==args.name)
    scene=next((old/"replay").iterdir()).name
    archive=(Path(os.environ["JIA_ARCHIVE"]) if pack is None
             else args.protocol/"models"/"sa_pose"/scene)
    replay=old/"replay"/scene
    scene=Path(cam.read(archive/"run.json")["input_path"])
    dest=args.output/args.name
    dest.mkdir(parents=True,exist_ok=True)
    {"fit":fit,"evaluate":evaluate,"audit":audit_sources}[args.stage](args,pack,old,archive,scene,replay,dest)
