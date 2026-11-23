#!/usr/bin/env python3
"""Run official Depth Pro 2 jointly on one three-frame clip.

Env vars: DEPTH_PRO_2_DIR, CKPT_PATH, FOV_CKPT_PATH, INPUT_DIR, OUT_DIR, GPU.
Known manifest focal length is passed explicitly; FOV checkpoint is only a fallback.
The official normalized inverse depth is retained; depth = inverse_depth^-1 * fx / W.
"""
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def main():
    import sys
    repo = Path(os.environ["DEPTH_PRO_2_DIR"])
    sys.path.insert(0, str(repo / "src"))
    from depth_pro_2 import PRESETS, create_model_and_transforms

    input_dir = Path(os.environ["INPUT_DIR"])
    out = Path(os.environ["OUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    rgbs = [np.asarray(Image.open(input_dir / "rgb" / f"{i:02d}.png").convert("RGB"))
            for i in range(len(manifest["frames"]))]
    if len({a.shape for a in rgbs}) != 1:
        raise ValueError("Joint inference requires three identically sized frames")
    config = dataclasses.replace(
        PRESETS["depthpro2-h"], checkpoint_uri=os.environ["CKPT_PATH"],
        fov_checkpoint_uri=os.environ["FOV_CKPT_PATH"])
    fx = [float(fr["K"][0][0]) for fr in manifest["frames"]]
    if not np.allclose(fx, fx[0]):
        raise ValueError("Official joint clips share one focal length; input frames differ")
    t0 = time.time()
    model, transform = create_model_and_transforms(config, device="cuda")
    print(f"✅ H+ loaded in {time.time() - t0:.1f}s; fov={os.environ['FOV_CKPT_PATH']}")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        batch = torch.stack([transform(rgb) for rgb in rgbs])
        t0 = time.time()
        prediction = model.infer(batch, f_px=fx[0])
        torch.cuda.synchronize()
    inverse_depth = prediction["inverse_depth"].float().cpu().numpy()
    model_depth = prediction["depth"].float().cpu().numpy()
    if inverse_depth.shape != (3, *rgbs[0].shape[:2]):
        raise ValueError(f"Unexpected output shape: {inverse_depth.shape}")
    print(f"⏱️ Joint 3-frame inference: {time.time() - t0:.2f}s; "
          f"peak VRAM={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB")
    metadata = {
        "model": "depthpro2-h", "official_repo": str(repo),
        "official_revision": os.environ.get("OFFICIAL_REVISION", "unrecorded"),
        "checkpoint": os.environ["CKPT_PATH"], "joint_frames": True,
        "focal_source": "explicit manifest K fx passed to official infer",
        "window": 3, "precision": "bfloat16 autocast",
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "inference_seconds": time.time() - t0, "frames": [],
    }
    (out / "depth").mkdir(exist_ok=True)
    (out / "normalized_inverse_depth").mkdir(exist_ok=True)
    for i, (fr, inv) in enumerate(zip(manifest["frames"], inverse_depth)):
        if not np.all(np.isfinite(inv)) or np.any(inv <= 0):
            raise ValueError(f"Invalid normalized inverse depth at frame {i}")
        w = fr["width"]
        frame_fx = float(fr["K"][0][0])
        expected_depth = inv ** -1 * frame_fx / w
        depth_error = np.max(np.abs(model_depth[i] - expected_depth) / expected_depth)
        if depth_error > 1e-5:
            raise ValueError(f"Unexpected official depth normalization: relative error {depth_error}")
        fx = frame_fx
        depth = fx / (w * inv)
        np.savez_compressed(out / "depth" / f"{i:02d}.npz", depth=depth.astype(np.float32),
                            K=np.asarray(fr["K"]), w2c=np.asarray(fr["w2c"]),
                            c2w=np.asarray(fr["c2w"]))
        np.save(out / "normalized_inverse_depth" / f"{i:02d}.npy", inv.astype(np.float32))
        metadata["frames"].append({
            "index": i, "fx_for_metric_depth": fx,
            "depth_quantiles_m": {str(q): float(np.quantile(depth, q))
                                  for q in (0.01, 0.5, 0.99)},
        })
        print(f"✅ frame {i}: depth p01/p50/p99="
              f"{np.quantile(depth, .01):.3f}/{np.quantile(depth, .5):.3f}/{np.quantile(depth, .99):.3f}")
    write_json(out / "inference.json", metadata)
    print(f"🎉 Native-resolution depth: {out / 'depth'}")


if __name__ == "__main__":
    main()
