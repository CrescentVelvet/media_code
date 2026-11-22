#!/usr/bin/env python3
"""24_zip_real_audit.py -- 在实际真实 source 上核验 Zip 原生高斯、相机与 PLY 回放。

zipsplat env 执行。重跑官方 ba=False 仅为核验，不替换原模型输出，不评 target。
输入 hash 必须等于冻结协议；输出目录不存在才创建。
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
C = type("Common", (), {
    "sha256": staticmethod(lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()),
    "write_json": staticmethod(lambda path, data: Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")),
})
M22 = importlib.util.spec_from_file_location("n1_render22", ROOT / "22_n1_head_compare.py")
R = importlib.util.module_from_spec(M22)
M22.loader.exec_module(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--scene", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()
    for scene in args.scene:
        audit_scene(args.protocol, scene, Path(args.out) / (scene + "_zip_native_" + args.version))


def audit_scene(protocol_path, scene, out):
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    pack = next((p for p in protocol["packages"] if p["scene"] == scene), None)
    if pack is None:
        raise ValueError("Unknown frozen scene: " + scene)
    expected = [f for f in pack["frames"] if f["split"] == "source"]
    expected = sorted(expected, key=lambda f: f["camera_id"])
    source = Path(pack["source_scene"])
    result = Path(protocol_path).resolve().parent / "models" / "zip_rgb" / (scene + "_source")
    if not result.is_dir():
        raise FileNotFoundError("Frozen Zip result not found: " + str(result))
    run = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
    paths = [source / "images" / (f["camera_id"] + ".png") for f in expected]
    if run["token_ba"] or run["image_files"] != [p.name for p in paths]:
        raise RuntimeError("Not the exact frozen RGB-only sources / ba=False")
    if run["official_commit"] != protocol["software"]["zip_repo"]:
        raise RuntimeError("Zip software version differs from preregistration")
    hashes = [C.sha256(p) for p in paths]
    if hashes != [f["processed_sha256"] for f in expected]:
        raise RuntimeError("Source PNG hash mismatch")
    out.mkdir(parents=True, exist_ok=False)
    import gc
    from zipsplat import ZipSplat, load_image
    model = ZipSplat(weights=run["weight_path"]).cuda().eval()
    device = next(model.parameters()).device
    print("AUDIT_DEVICE", scene, device, torch.cuda.current_device(), flush=True)
    saved = np.load(result / "predicted_cameras.npz", allow_pickle=False)
    g = R.load_ply(result / "gaussians_feedforward.ply", "wxyz")
    records = {"status": "pending", "source_images": [], "tensor_deltas": {},
               "camera_deltas": {}, "per_source": {}, "native_render": {
                   "height": None, "width": None, "eps2d": .3, "background": "explicit zero",
                   "max_abs_gate": 3 / 255}}
    records["saved_c2w"] = saved["c2w"].tolist()
    records["saved_cameras"] = saved["cameras"].tolist()
    records["saved_model_cameras"] = None
    with torch.no_grad():
        pred = model([load_image(p) for p in paths], ba=False)
        records["saved_model_cameras"] = [c.K.cpu().numpy().tolist() for c in pred["model_cameras"]]
        native = pred["gaussians"]
        for key, value in {"means": native.means, "scales": native.scales,
                           "quats": native.quats, "opacities": native.opacities,
                           "sh": native.sh_coeffs}.items():
            delta = float(torch.max(torch.abs(value.cpu() - torch.from_numpy(g[key]))).item())
            records["tensor_deltas"][key] = delta
        records["source_images"] = [{"path": str(p), "sha256": C.sha256(p)} for p in paths]
        records["ply_path"] = str(result / "gaussians_feedforward.ply")
        records["ply_sha256"] = C.sha256(records["ply_path"])
        for key, new in (("c2w", pred["poses"].cpu().Rt.numpy()),
                         ("cameras", pred["cameras"].cpu().data_.numpy())):
            records["camera_deltas"][key] = float(np.abs(new - saved[key]).max())
        if list(saved["names"]) != [p.name for p in paths]:
            raise RuntimeError("Native saved source order mismatch")
        gpu = R.to_gpu(g, device)
        for i, path in enumerate(paths):
            ref = np.asarray(Image.open(result / "renders" / f"{i:03d}.png"), np.float32) / 255
            rd, _ = native.render(pred["model_cameras"][i], pred["poses"][i])
            rd = rd[0].clamp(0, 1).moveaxis(0, -1).cpu().numpy()
            low, _, _ = R.render(gpu, g["degree"], pred["poses"][i].cpu().Rt.numpy(),
                                 pred["model_cameras"][i].K.cpu().numpy(),
                                 ref.shape[1], str(device), .3, height=ref.shape[0])
            records["native_render"]["height"], records["native_render"]["width"] = ref.shape[:2]
            rec = {"native_official_api_vs_saved": {"max": float(np.abs(rd - ref).max()),
                   "mean": float(np.abs(rd - ref).mean())},
                   "ply_lowlevel_vs_saved": {"max": float(np.abs(low - ref).max()),
                   "mean": float(np.abs(low - ref).mean())}}
            if rec["native_official_api_vs_saved"]["max"] > 3 / 255:
                raise RuntimeError("Official Zip rerun differs from saved native render")
            if rec["ply_lowlevel_vs_saved"]["max"] > 3 / 255:
                raise RuntimeError("Zip PLY low-level replay differs from native render")
            records["per_source"][path.stem] = rec
            Image.fromarray(np.uint8(np.clip(low, 0, 1) * 255)).save(out / f"{path.stem}_ply.png")
    for key, lim in {"means": 1e-6, "scales": 1e-5, "quats": 1e-6,
                     "opacities": 1e-6, "sh": 1e-6}.items():
        if records["tensor_deltas"][key] > lim:
            raise RuntimeError("Official tensor to PLY delta exceeds gate: " + key)
    if records["camera_deltas"]["c2w"] > 1e-6 or records["camera_deltas"]["cameras"] > 1e-4:
        raise RuntimeError("Zip native cameras are not deterministic for the frozen sources")
    records["status"] = "passed_native_replay_not_target_model_acceptance"
    C.write_json(out / "zip_native_audit.json", records)
    print(json.dumps(records, indent=2), flush=True)
    print("DONE", out, flush=True)
    del pred, native, model, gpu
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
