"""Compare three depth sources on the same physical cameras and frozen synthetic heads.

prepare writes the input manifest consumed by depth_pro_2/run_inference.py.
score projects the three observed depth maps into disjoint held-out cameras.
All depths are axial Z. No held-out scale or per-frame registration is fitted.
"""
import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData

spec = importlib.util.spec_from_file_location("camera_probe", Path(__file__).with_name("29_camera_probe.py"))
cam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cam)
ev = cam.ev


def prepare(pack, out):
    scene = Path(pack["source3_scene"])
    poses, K, ids = ev.M03.load_cameras(scene/"prior_camera.json")
    root = out/"dp_input"
    (root/"rgb").mkdir(parents=True, exist_ok=True)
    frames = []
    for i, stem in enumerate(ids):
        path = scene/"images"/(stem+".png")
        dest = root/"rgb"/f"{i:02}.png"
        if not dest.exists():
            dest.symlink_to(path)
        w,h = Image.open(path).size
        frames.append(dict(width=w,height=h,K=K[i].tolist(),c2w=poses[i].tolist(),
                           w2c=np.linalg.inv(poses[i]).tolist(),source=str(path)))
    cam.save(root/"manifest.json",dict(frames=frames))


def project(z, mask, K, source, target, target_K):
    y,x = np.nonzero(mask & np.isfinite(z) & (z>0))
    pts = (np.stack((x,y,np.ones_like(x)),axis=1) @ np.linalg.inv(K).T)*z[y,x,None]
    rel = np.linalg.inv(target) @ source
    pts = pts @ rel[:3,:3].T + rel[:3,3]
    pix = pts @ target_K.T
    u,v = np.rint(pix[:,:2]/pix[:,2:]).astype(int).T
    ok = (pts[:,2]>0)&(u>=0)&(u<756)&(v>=0)&(v<756)
    buf = np.full((756,756),np.inf,np.float32)
    np.minimum.at(buf,(v[ok],u[ok]),pts[ok,2])
    return buf


def teacher_depth(xyz, pose, K, s, raw):
    local=(xyz-pose[:3,3]) @ pose[:3,:3]
    local=local[local[:,2]>.01]
    pix=local @ K.T
    uv=pix[:,:2]/pix[:,2:]
    rounded=np.rint(uv).astype(int)
    valid_ray=(np.abs(uv-rounded).max(axis=1)<.001)&(rounded.min(axis=1)>=0)&(rounded.max(axis=1)<756)
    x,y=rounded[valid_ray].T
    ratio=local[valid_ray,2]/(s*raw[y,x])
    assert len(ratio)>5000, "Insufficient PLY-to-native depth correspondences"
    factor=np.median(ratio)
    assert np.quantile(np.abs(ratio/factor-1),.99)<1e-4, "Inconsistent teacher scale"
    return raw*factor,float(factor),len(ratio)


def score(pack, root, out):
    scene=Path(pack["source3_scene"])
    model=root/"models"/"sa_pose"/scene.name
    s,R,T,_=ev.M15.estimate_sim3(model,scene)
    source,K,ids=ev.M03.load_cameras(scene/"prior_camera.json")
    exp,ek,ei=ev.M03.load_cameras(model/"cameras.json")
    assert ids==ei
    ply=PlyData.read(str(model/"points.ply"))["vertex"]
    # The exporter removes invalid pixels. Recover its shared depth multiplier
    # from points landing on the native integer pixel lattice in each camera.
    xyz=np.stack([ply[k][::13] for k in ("x","y","z")],axis=-1)
    gs=ev.M03.load_ply(model/"gaussians.ply")
    pi=np.load(out/"pi3x"/"predictions.npz")
    scale=cam.read(out/"camera.json")["arms"]["Pi3X"]["source_scale"]
    lms={int(v["id"]):np.array(v["points"]) for v in cam.read(Path(pack["landmarks"]))["views"].values()}
    target,tk,tids=ev.M03.load_cameras(Path(pack["target_scene"])/"prior_camera.json")
    rows=[]
    calibration={name:[] for name in ("SA","DP2","Pi3X")}
    srcmaps=[]
    overlays=[]
    def data(fr,k):
        wh=tuple(fr["processed_wh"])
        p=ev.evaluation_geometry(k,wh,k)
        gt,valid=ev.M21.load_depth(fr["depth_path"])
        alpha=np.asarray(Image.open(fr["path"]).convert("RGBA"))[:,:,3]
        valid &= alpha==255
        rgb=ev.M21.official_preprocess_rgb(fr["path"],p)
        roi={}
        if fr["raw_id"] in lms:
            pts=lms[fr["raw_id"]]
            masks=ev.M21.region_masks(pts,gt.shape)
            roi={key:masks[key] for key in ("nose_tip","nose_ala","face_tight_landmark")}
            core=np.zeros(gt.shape,np.uint8)
            cv2.circle(core,tuple(np.rint(pts[1]).astype(int)),6,1,-1)
            roi["tip_core"]=core.astype(bool)
            # Wider projected hemiface supplies the more exposed cheek. Fix this
            # choice from GT landmarks before inspecting any method's depths.
            tip=pts[4]
            options=[pts[205],pts[425]]
            center=max(options,key=lambda point:abs(point[0]-tip[0]))
            cheek=np.zeros(gt.shape,np.uint8)
            cv2.circle(cheek, tuple(np.rint(center).astype(int)),6,1,-1)
            roi["cheek"]=cheek.astype(bool)&masks["face_tight_landmark"]
            if pack["name"]=="rosemary" and fr["raw_id"]==38:
                roi.pop("nose_ala")
                roi.pop("cheek")
            if pack["name"]=="daven_holdout" and fr["raw_id"]==34:
                roi.pop("cheek")  # Frozen wider-side cheek overlaps the hair boundary.
            roi={key:ev.M21.remap_nearest(mask,p) for key,mask in roi.items()}
        return p,ev.M21.remap_nearest(gt,p),ev.M21.remap_nearest(valid,p),roi,rgb

    def measure(fr,gt,valid,roi,maps,rgb):
        support=np.logical_and.reduce([np.isfinite(z)&(z>0) for z in maps.values()]) & valid
        row=dict(raw_id=fr["raw_id"],split=fr["split"],methods={},roi={})
        for region,mask in roi.items():
            full=mask&valid
            den=int(full.sum())
            row["roi"][region]=dict(n=den,common=float((mask&support).sum()/den) if den else 0)
        for name,z in maps.items():
            r={}
            for region,mask in roi.items():
                use=mask&support
                if use.sum()<12:
                    continue
                e=(z[use]-gt[use])*1000
                r[region]=dict(n=int(use.sum()),bias_mm=float(np.mean(e)),mae_mm=float(np.abs(e).mean()),
                               p90_mm=float(np.percentile(np.abs(e),90)),
                               own_coverage=float((np.isfinite(z)&(z>0)&mask&valid).sum()/(mask&valid).sum()))
            if "cheek" in r and "tip_core" in r:
                a,b=roi["cheek"]&support,roi["tip_core"]&support
                dz=float((np.median(z[a])-np.median(z[b]))*1000)
                truth=float((np.median(gt[a])-np.median(gt[b]))*1000)
                r["relief"]=dict(pred_mm=dz,gt_mm=truth,error_mm=dz-truth)
            row["methods"][name]=r
        rows.append(row)
        img=np.uint8(np.clip(rgb,0,1)*255)
        for key,col in [("nose_tip",(255,160,0)),("cheek",(0,255,160)),("tip_core",(250,40,90))]:
            if key in roi:
                contours=cv2.findContours(roi[key].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(img,contours,-1,col,2)
        if "face_tight_landmark" in roi:
            y,x=np.where(roi["face_tight_landmark"])
            img=img[max(0,y.min()-10):y.max()+11,max(0,x.min()-10):x.max()+11]
        tile=Image.fromarray(img)
        tile.thumbnail((230,230))
        overlays.append(tile)

    source_frames=[f for f in pack["frames"] if f["split"]=="anchor"]
    for i,fr in enumerate(source_frames):
        p,gt,valid,roi,rgb=data(fr,K[i])
        assert np.allclose(p["K_reference"],ek[i],atol=.01)
        raw=np.load(model/"depth"/(ids[i]+".npy")).squeeze()
        teacher,factor,checks=teacher_depth(xyz,exp[i],ek[i],s,raw)
        native=np.load(out/"dp2"/"depth"/f"{i:02}.npz")["depth"]
        pc=pi["camera_poses"][i]
        pi_z=((pi["points"][i]-pc[:3,3]) @ pc[:3,:3])[:,:,2]*scale
        pi_z=cv2.resize(pi_z,tuple(fr["processed_wh"]),interpolation=cv2.INTER_LINEAR)
        maps=dict(SA=teacher,DP2=ev.M21.remap_nearest(native,p),Pi3X=ev.M21.remap_nearest(pi_z,p))
        measure(fr,gt,valid,roi,maps,rgb)
        if "cheek" in roi:
            cal=valid&roi["face_tight_landmark"]&~roi["nose_tip"]&~roi["nose_ala"]&~roi["cheek"]
            for name,z in maps.items():
                calibration[name].append(gt[cal]/z[cal])
        rows[-1]["SA_raw_to_metric"]=float(factor)
        rows[-1]["SA_factor_checks"]=checks
        srcmaps.append((maps,valid,p["K_reference"]))
    factors={name:float(np.median(np.concatenate(v))) for name,v in calibration.items()}
    for i,fr in enumerate(f for f in pack["frames"] if f["split"]=="target"):
        assert fr["scene_slots"]["target"]==tids[i]
        p,gt,valid,roi,rgb=data(fr,tk[i])
        maps={}
        for name in ("SA","DP2","Pi3X"):
            layers=[project(z[name],m,k,source[j],target[i],p["K_reference"]) for j,(z,m,k) in enumerate(srcmaps)]
            maps[name]=np.minimum.reduce(layers)
        ed,alpha=ev.M21.M13.render_expected_depth(gs,ev.M15.apply_sim3(target[i],s,R,T),
                            p["K_reference"],756,756,device="cuda",eps2d=.01)
        maps["SA_fused"]=np.where(alpha[0,...,0]>.01,ed[0]/s,np.inf)
        measure(fr,gt,valid,roi,maps,rgb)
        scaled={}
        for name,factor in factors.items():
            layers=[project(z[name]*factor,m,k,source[j],target[i],p["K_reference"]) for j,(z,m,k) in enumerate(srcmaps)]
            scaled[name]=np.minimum.reduce(layers)
        measure(dict(fr,split="target_source_scale_sensitivity"),gt,valid,roi,scaled,rgb)
        overlays.pop()
    sheet=Image.new("RGB",(230*len(overlays),230),"white")
    for i,tile in enumerate(overlays):
        sheet.paste(tile,(i*230,0))
    sheet.save(out/"nose_roi_v2.jpg",quality=95)
    if pack["name"]=="daven_holdout":
        sheet.crop((3*230,0,3*230+160,176)).save(out/"target24_nose_cheek.png")
    cam.save(out/"nose_v2.json",dict(frames=rows,pi_metric_scale=scale,source_face_scales=factors,
        target_projection="GT source matte for every point arm; nearest rounded pixel z buffer, shared target support",
        relief="median camera Z cheek minus median nose, view dependent; signed prediction minus GT"))


def real_reprojection(pack,root,out):
    """Compare visible RGB correspondences in the fixed auxiliary target ROI."""
    scene=Path(pack["source3_scene"])
    model=root/"models"/"sa_pose"/scene.name
    src,K,ids=ev.M03.load_cameras(scene/"prior_camera.json")
    tar=Path(pack["target_scene"])
    targets,tar_K,tids=ev.M03.load_cameras(tar/"prior_camera.json")
    scale=ev.M15.estimate_sim3(model,scene)[0]
    xp,xk,_=ev.M03.load_cameras(model/"cameras.json")
    verts=PlyData.read(str(model/"points.ply"))["vertex"]
    xyz=np.stack([verts[k][::13] for k in ("x","y","z")],axis=-1)
    pi=np.load(out/"pi3x"/"predictions.npz")
    ps=cam.read(out/"camera.json")["arms"]["Pi3X"]["source_scale"]
    clouds={k:[] for k in ("SA","DP2","Pi3X")}
    for i,stem in enumerate(ids):
        file=scene/"images"/(stem+".png")
        wh=Image.open(file).size
        prep=ev.evaluation_geometry(K[i],wh,K[i])
        rgb=ev.M21.official_preprocess_rgb(file,prep)
        raw=np.load(model/"depth"/(stem+".npy")).squeeze()
        sa=teacher_depth(xyz,xp[i],xk[i],scale,raw)[0]
        dp=np.load(out/"dp2"/"depth"/f"{i:02}.npz")["depth"]
        pc=pi["camera_poses"][i]
        pz=((pi["points"][i]-pc[:3,3])@pc[:3,:3])[:,:,2]*ps
        ys,xs=np.indices(sa.shape)
        rays=np.stack((xs,ys,np.ones_like(xs)),axis=-1)@np.linalg.inv(prep["K_reference"]).T
        for name,z in [("SA",sa),("DP2",ev.M21.remap_nearest(dp,prep)),
                       ("Pi3X",ev.M21.remap_nearest(cv2.resize(pz,wh),prep))]:
            ok=np.isfinite(z)&(z>.05)
            world=(rays[ok]*z[ok,None])@src[i,:3,:3].T+src[i,:3,3]
            clouds[name].append((world,rgb[ok]))
    protocol=Path(pack["n1_eval"]).parent.parent/"protocol_v1"
    old=next(v for v in cam.read(protocol/"protocol.json")["packages"] if v["scene"]==pack["name"])
    qa=cam.read(protocol/"roi_review.json")["views"][pack["name"]]
    tiles=[]
    report={}
    for i,stem in enumerate(tids):
        file=tar/"images"/(stem+".png")
        prep=ev.evaluation_geometry(tar_K[i],Image.open(file).size,tar_K[i])
        gt=ev.M21.official_preprocess_rgb(file,prep)
        raw=next(f["raw_id"] for f in pack["frames"] if f["scene_slots"].get("target")==stem)
        fr=next(f for f in old["frames"] if f["raw_frame_idx"]==raw)
        if qa[fr["camera_id"]]["status"]!="accepted":
            report[stem]={"face":"N/A"}
            continue
        mask=ev.M21.region_masks(np.array(fr["landmark_qa"]["points"]),gt.shape[:2])["face_tight_landmark"]
        ims={"reference":gt}
        observed={}
        for name,views in clouds.items():
            world=np.concatenate([c[0] for c in views])
            color=np.concatenate([c[1] for c in views])
            local=(world-targets[i,:3,3])@targets[i,:3,:3]
            pos=local[:,2]>.05
            local,color=local[pos],color[pos]
            pix=local@prep["K_reference"].T
            u,v=np.rint(pix[:,:2]/pix[:,2:]).astype(int).T
            valid=(u>=0)&(v>=0)&(u<756)&(v<756)
            idx=v[valid]*756+u[valid]
            depth=local[valid,2]
            order=np.lexsort((depth,idx))
            _,first=np.unique(idx[order],return_index=True)
            keep=order[first]
            pixrgb=np.zeros_like(gt).reshape(-1,3)
            pixrgb[idx[keep]]=color[valid][keep]
            hit=np.zeros(756*756,bool)
            hit[idx[keep]]=True
            ims[name]=pixrgb.reshape(gt.shape)
            observed[name]=hit.reshape(mask.shape)
        common=mask&np.logical_and.reduce(list(observed.values()))
        report[stem]={"common_coverage":float(common.sum()/mask.sum()),"methods":{
            k:{"coverage":float(v[mask].mean()),"rgb_mae_common":float(np.abs(ims[k][common]-gt[common]).mean()) if common.any() else None}
            for k,v in observed.items()}}
        yy,xx=np.where(mask)
        row=[]
        for img in ims.values():
            crop=img[max(0,yy.min()-6):yy.max()+7,max(0,xx.min()-6):xx.max()+7]
            tile=Image.fromarray(np.uint8(np.clip(crop,0,1)*255))
            tile.thumbnail((205,205))
            row.append(tile)
        tiles.append(row)
    sheet=Image.new("RGB",(820,205*len(tiles)),"white")
    for i,row in enumerate(tiles):
        for j,tile in enumerate(row):
            sheet.paste(tile,(205*j,205*i))
    sheet.save(out/"real_face_GT_SA_DP2_Pi3X.png")
    cam.save(out/"real_face.json",dict(targets=report,scope="nearest point, fixed known cameras; RGB and coverage only"))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=["prepare","score","real"])
    ap.add_argument("protocol",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    for p in cam.read(args.protocol/"manifest.json")["packages"]:
        if args.stage=="prepare":
            prepare(p,args.output/p["name"])
        elif args.stage=="real" and p["kind"]!="static_synthetic":
            real_reprojection(p,args.protocol,args.output/p["name"])
        elif args.stage=="score" and p["kind"]=="static_synthetic":
            score(p,args.protocol,args.output/p["name"])
