#!/usr/bin/env python3
"""Infer each of three frames independently with the official Depth Pro 2 H+.

Env vars: DEPTH_PRO_2_DIR, CKPT_PATH, FOV_CKPT_PATH, INPUT_DIR, OUT_DIR, GPU.
Every infer call has T=1. Joint context is never shared between calls. Known
manifest focal length and native normalized inverse depth are preserved.
"""
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import sys
    repo = Path(os.environ["DEPTH_PRO_2_DIR"])
    sys.path.insert(0, str(repo / "src"))
    from depth_pro_2 import PRESETS, create_model_and_transforms

    input_dir = Path(os.environ["INPUT_DIR"])
    out = Path(os.environ["OUT_DIR"])
    assert out.name == "single_frame" and out.parent == input_dir
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "normalized_inverse_depth").mkdir(exist_ok=True)
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    frames = manifest["frames"]
    assert len(frames) == 3
    rgbs = [np.asarray(Image.open(input_dir / "rgb" / f"{f['index']:02d}.png").convert("RGB")) for f in frames]
    fx = [float(fr["K"][0][0]) for fr in frames]
    assert np.allclose(fx, fx[0]), "This experiment keeps the joint run's shared focal length"
    config = dataclasses.replace(
        PRESETS["depthpro2-h"], checkpoint_uri=os.environ["CKPT_PATH"],
        fov_checkpoint_uri=os.environ["FOV_CKPT_PATH"])
    t0 = time.time()
    model, transform = create_model_and_transforms(config, device="cuda")
    print(f"✅ H+ loaded in {time.time() - t0:.1f}s", flush=True)
    metadata = {
        "model": "depthpro2-h", "official_revision": os.environ.get("OFFICIAL_REVISION", "unrecorded"),
        "checkpoint": os.environ["CKPT_PATH"], "checkpoint_sha256": sha256(Path(os.environ["CKPT_PATH"])),
        "joint_frames": False, "independent_single_frame_calls": 3, "window": 1,
        "focal_source": "same explicit manifest K fx as joint inference",
        "depth_formula": "Z = fx / (width * normalized_inverse_depth); axial Z",
        "input_manifest_sha256": sha256(input_dir / "manifest.json"), "frames": [],
    }
    for index, (fr, rgb) in enumerate(zip(frames, rgbs)):
        with torch.inference_mode():
            batch = transform(rgb).unsqueeze(0)
            assert batch.shape[0] == 1
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            prediction = model.infer(batch, f_px=fx[index])
            torch.cuda.synchronize()
        inv = prediction["inverse_depth"].float().cpu().numpy()
        model_depth = prediction["depth"].float().cpu().numpy()
        assert inv.shape == model_depth.shape == (1, *rgb.shape[:2]), inv.shape
        inv = inv[0]
        model_depth = model_depth[0]
        assert np.all(np.isfinite(inv)) and np.all(inv > 0)
        depth = fx[index] / (fr["width"] * inv)
        error = np.max(np.abs(model_depth - depth) / depth)
        assert error <= 1e-5, error
        for folder in ("depth", "normalized_inverse_depth"):
            assert not (out / folder).is_symlink()
        np.savez_compressed(out / "depth" / f"{index:02d}.npz", depth=depth.astype(np.float32),
                            K=np.asarray(fr["K"]), w2c=np.asarray(fr["w2c"]), c2w=np.asarray(fr["c2w"]))
        np.save(out / "normalized_inverse_depth" / f"{index:02d}.npy", inv.astype(np.float32))
        record = {
            "index": index, "input_rgb_sha256": sha256(input_dir / "rgb" / f"{index:02d}.png"),
            "seconds": time.time() - t0, "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
            "depth_quantiles_m": {str(q): float(np.quantile(depth, q)) for q in (0.01, 0.5, 0.99)},
            "depth_npz_sha256": sha256(out / "depth" / f"{index:02d}.npz"),
        }
        metadata["frames"].append(record)
        write_json(out / "inference.json", metadata)
        print(f"✅ independent frame {index}: {record['seconds']:.2f}s; "
              f"VRAM={record['peak_vram_gib']:.2f}GiB", flush=True)
    print(f"🎉 Three independent single-frame predictions: {out}", flush=True)


if __name__ == "__main__":
    main()
