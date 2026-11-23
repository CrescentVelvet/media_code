"""Three-source camera audit and fixed-intrinsics SplatAnything pose ablation.

Run under the SA environment with its official repo on PYTHONPATH. Input is
the frozen protocol directory. Output contains Pi3X exports in NAME/pi3x/.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def module(filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


ev = module("25_n1_real_eval.py")
pairs = list(itertools.combinations(range(3), 2))


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


def camera_metrics(pred, ref):
    rows = []
    for i, j in pairs:
        a, b = (np.linalg.inv(stack[i]) @ stack[j] for stack in (pred, ref))
        u, v = a[:3, 3], b[:3, 3]
        cos = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
        rows.append(dict(pair=[i, j], rot_deg=float(Rotation.from_matrix(
            a[:3, :3].T @ b[:3, :3]).magnitude() * 180 / np.pi),
            direction_deg=float(np.degrees(np.arccos(np.clip(cos, -1, 1)))),
            baseline_m=float(np.linalg.norm(v)), scale=float(np.linalg.norm(v)/np.linalg.norm(u))))
    return rows


def background_matches(files, ref, Ks, real):
    """Reference-consistent SIFT matches. For real people sample side margins only."""
    sift = cv2.SIFT_create(nfeatures=5000)
    features = []
    for file in files:
        img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        mask = np.full(img.shape, 255, np.uint8)
        if real:
            mask[:, img.shape[1]//5:-img.shape[1]//5] = 0
        k, d = sift.detectAndCompute(img, mask)
        features.append((np.array([p.pt for p in k]), d))
    tracks = []
    for i, j in pairs:
        x, y = features[i], features[j]
        if x[1] is None or y[1] is None or len(y[1]) < 2:
            tracks.append((np.empty((0,2)), np.empty((0,2)), np.empty((0,3))))
            continue
        matches = cv2.BFMatcher().knnMatch(x[1], y[1], k=2)
        good = [a for a,b in matches if a.distance < .72*b.distance]
        uv = np.array([x[0][a.queryIdx] for a in good]).reshape(-1,2)
        vv = np.array([y[0][a.trainIdx] for a in good]).reshape(-1,2)
        if not len(uv):
            tracks.append((uv, vv, np.empty((0,3))))
            continue
        rel = np.linalg.inv(ref[j]) @ ref[i]
        points = cv2.triangulatePoints(Ks[i] @ np.eye(3,4), Ks[j] @ rel[:3], uv.T, vv.T)
        xyz = (points[:3]/points[3]).T
        zz = xyz @ rel[:3,:3].T + rel[:3,3]
        proj = zz @ Ks[j].T
        good = (xyz[:,2]>.05) & (zz[:,2]>.05) & (np.linalg.norm(proj[:,:2]/proj[:,2:]-vv,axis=1)<1.5)
        tracks.append((uv[good], vv[good], xyz[good]))
    return tracks


def prepare(pack, root, output):
    scene = Path(pack["source3_scene"])
    ref, K, ids = ev.M03.load_cameras(scene/"prior_camera.json")
    ref, K = ref.astype(float), K.astype(float)
    files = [scene/"images"/(s+".png") for s in ids]
    expected = [f for f in pack["frames"] if f["split"]=="anchor"]
    hashes = [hashlib.sha256(f.read_bytes()).hexdigest() for f in files]
    assert [p["scene_slots"]["source3"] for p in expected] == ids
    for h, f in zip(hashes, expected):
        if f.get("source3_input_file_sha256"):
            assert h == f["source3_input_file_sha256"]
    wh = np.array(Image.open(files[0]).size)
    track = background_matches(files, ref, K, pack["kind"]!="static_synthetic")
    pi = np.load(output/"pi3x"/"predictions.npz")
    zipdata = np.load(root/"models"/"zip_rgb"/scene.name/"predicted_cameras.npz")
    assert list(pi["frame_names"]) == list(zipdata["names"]) == [p.name for p in files]
    sa, sk, sids = ev.M03.load_cameras(root/"models"/"sa_rgb"/scene.name/"cameras.json")
    assert sids == ids
    kr = np.asarray([ev.evaluation_geometry(None, tuple(wh), k)["K_reference"] for k in K])
    sf = np.array([sk[:,0,0]/kr[:,0,0],sk[:,1,1]/kr[:,1,1]]).T
    zf = zipdata["cameras"][:,2:4] / K[:,[0,1],[0,1]]
    pf = pi["K_recovered"][:,[0,1],[0,1]] * (wh/pi["target_wh"]) / K[:,[0,1],[0,1]]
    result = {"anchor_ids":ids, "sha256":hashes, "arms":{}}
    for name, poses, focal in [("Pi3X", pi["camera_poses"],pf),("SA", sa,sf),("Zip",zipdata["c2w"],zf)]:
        poses = poses.astype(float)
        stats = camera_metrics(poses, ref)
        scale = float(np.median([r["scale"] for r in stats]))
        for (i,j), (_,uv,xyz), row in zip(pairs,track,stats):
            a = np.linalg.inv(poses[j]) @ poses[i]
            pt = xyz @ a[:3,:3].T + a[:3,3]*scale
            proj = pt @ K[j].T
            valid = pt[:,2] > 0
            err = np.linalg.norm(proj[valid,:2]/proj[valid,2:] - uv[valid],axis=1)
            row.update(points=len(xyz),positive=int(valid.sum()), median_px=float(np.median(err)) if len(err) else None)
        result["arms"][name] = dict(pairs=stats, focal_ratio_xy=focal.tolist(),source_scale=scale)
        if name=="Zip":
            continue
        arm = output/("prior_"+name)
        arm.mkdir(exist_ok=True)
        if not (arm/"images").exists():
            (arm/"images").symlink_to(scene/"images", target_is_directory=True)
        prior = read(scene/"prior_camera.json")
        rot = ref[0,:3,:3] @ poses[0,:3,:3].T
        for idx, row in enumerate(prior["extrinsics"]):
            c = np.eye(4)
            c[:3,:3] = rot @ poses[idx,:3,:3]
            c[:3,3] = ref[0,:3,3]+scale*rot @ (poses[idx,:3,3]-poses[0,:3,3])
            row["matrix"] = c.tolist()
        save(arm/"prior_camera.json",prior)
    save(output/"camera.json",result)


def heldout(pack, root, output):
    scene = Path(pack["source3_scene"])
    tar = Path(pack["target_scene"])
    cams, K, ids = ev.M03.load_cameras(tar/"prior_camera.json")
    grid = []
    result = {}
    face_masks = {}
    if pack["kind"] == "static_synthetic":
        landmarks = {int(v["id"]):v["points"] for v in read(Path(pack["landmarks"]))["views"].values()}
    else:
        real_protocol = Path(pack["n1_eval"]).parent.parent/"protocol_v1"
        real_pack = next(p for p in read(real_protocol/"protocol.json")["packages"] if p["scene"]==pack["name"])
        review = read(real_protocol/"roi_review.json")["views"][pack["name"]]
    for stem,k in zip(ids,K):
        raw = next(f["raw_id"] for f in pack["frames"] if f["scene_slots"].get("target")==stem)
        wh = Image.open(tar/"images"/(stem+".png")).size
        if pack["kind"] == "static_synthetic":
            native = ev.M21.region_masks(np.asarray(landmarks[raw]),wh[::-1])["face_tight_landmark"]
            face_masks[stem] = ev.M21.remap_nearest(native,ev.evaluation_geometry(k,wh,k))
        else:
            frame = next(f for f in real_pack["frames"] if f["raw_frame_idx"]==raw)
            if review[frame["camera_id"]]["status"]=="accepted":
                face_masks[stem] = ev.M21.region_masks(np.asarray(frame["landmark_qa"]["points"]),
                                                       (756,756))["face_tight_landmark"]
    for arm, p, model in [
        ("reference",scene,root/"models"/"sa_pose"/scene.name),
        ("SA",output/"prior_SA",output/"models"/"prior_SA"),
        ("Pi3X",output/"prior_Pi3X",output/"models"/"prior_Pi3X")]:
        fit = ev.M15.estimate_sim3(model,p)
        gs = ev.M03.load_ply(model/"gaussians.ply")
        result[arm] = []
        strips = []
        for stem,cam,k in zip(ids,cams,K):
            path = tar/"images"/(stem+".png")
            prep = ev.evaluation_geometry(k,Image.open(path).size,k)
            gt = ev.M21.official_preprocess_rgb(path,prep)
            rd,al = ev.M03.render(gs, ev.M15.apply_sim3(cam,*fit[:3]),prep["K_reference"],756,756,
                                  background=0,device="cuda",eps2d=.01)
            rd = rd[0]
            result[arm].append(dict(id=stem,psnr=ev.M03.psnr(gt,rd),coverage=float((al>.5).mean())))
            mask = face_masks.get(stem)
            if mask is not None:
                result[arm][-1].update(face_psnr=ev.M03.psnr(gt[mask],rd[mask]),
                                       face_coverage=float((al[0,...,0][mask]>.5).mean()))
            strips.append([Image.fromarray(np.uint8(np.clip(im,0,1)*255)).resize((220,220)) for im in [gt,rd]])
        if not grid:
            grid.append([row[0] for row in strips])
        grid.append([row[1] for row in strips])
    sheet = Image.new("RGB",(880,440))
    for j,col in enumerate(grid):
        for i,img in enumerate(col):
            sheet.paste(img,(j*220,i*220))
    sheet.save(output/"target_GT_reference_SA_Pi3X.jpg",quality=93)
    save(output/"heldout.json",result)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=["prepare","heldout"])
    ap.add_argument("protocol",type=Path)
    ap.add_argument("output",type=Path)
    ap.add_argument("name")
    args=ap.parse_args()
    pack=next(p for p in read(args.protocol/"manifest.json")["packages"] if p["name"]==args.name)
    (prepare if args.stage=="prepare" else heldout)(pack,args.protocol,args.output/args.name)
