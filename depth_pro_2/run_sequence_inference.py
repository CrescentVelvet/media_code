#!/usr/bin/env python3
"""Run official buffered DP2 on an ordered calibrated image sequence.

Env vars: INPUT_DIR (manifest+frames), OUT_DIR, DEPTH_PRO_2_DIR, CKPT_PATH,
FOV_CKPT_PATH, WINDOW (16), OVERLAP (4), PLAYBACK_FPS (12).
Known fx is passed explicitly. Poses are saved only for subsequent PCD lifting;
they are not accepted by or supplied to DP2's image/focal inference interface.
"""
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from io_utils import sha256, write_json


def main():
    import sys
    sys.path.insert(0, str(Path(os.environ["DEPTH_PRO_2_DIR"]) / "src"))
    from depth_pro_2 import PRESETS, create_model_and_transforms

    source = Path(os.environ["INPUT_DIR"])
    out = Path(os.environ["OUT_DIR"])
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    window, overlap = int(os.environ.get("WINDOW", "16")), int(os.environ.get("OVERLAP", "4"))
    assert 0 <= overlap < window and manifest["frame_count"] >= window
    if out.resolve() == source.resolve() or out.is_symlink():
        raise ValueError("Inference output must be a separate non-symlink directory")
    for folder in (out, out / "depth", out / "normalized_inverse_depth"):
        assert not folder.is_symlink()
        folder.mkdir(parents=True, exist_ok=True)
    config = dataclasses.replace(PRESETS["depthpro2-h"], checkpoint_uri=os.environ["CKPT_PATH"],
                                fov_checkpoint_uri=os.environ["FOV_CKPT_PATH"])
    fx = [f["K"][0][0] for f in manifest["frames"]]
    if not np.allclose(fx, fx[0], rtol=1e-5, atol=1e-5):
        raise ValueError(f"Buffered clips require one shared focal: min={min(fx)}, max={max(fx)}")
    model, transform = create_model_and_transforms(config, device="cuda")

    def inputs():
        for i in range(manifest["frame_count"]):
            rgb = np.asarray(Image.open(source / "frames" / f"{i:06d}.jpg").convert("RGB"))
            yield transform(rgb)

    metadata = {
        "model": "depthpro2-h", "official_repo": os.environ["DEPTH_PRO_2_DIR"],
        "official_revision": os.environ.get("OFFICIAL_REVISION", "unrecorded"),
        "checkpoint_sha256": sha256(Path(os.environ["CKPT_PATH"])),
        "fov_checkpoint_sha256": sha256(Path(os.environ["FOV_CKPT_PATH"])),
        "input_manifest_sha256": sha256(source / "manifest.json"),
        "mode": "buffered", "window": window, "overlap": overlap,
        "output_width": manifest["frames"][0]["width"], "output_height": manifest["frames"][0]["height"],
        "focal_source": "Explicit calibrated post-undistort K fx passed to official infer_video",
        "camera_poses": "Available in manifest for world-space PCD lifting after inference; not model conditioning",
        "output_fps": float(os.environ.get("PLAYBACK_FPS", manifest["playback_fps"])),
        "processed_frames": 0, "frames": [],
    }
    write_json(out / "inference.json", metadata)
    start = time.time()
    for i, prediction in enumerate(model.infer_video(inputs(), window=window, overlap=overlap,
                                                    f_px=fx[0], mode="buffered")):
        inv = prediction["inverse_depth"].float().cpu().numpy()
        depth = prediction["depth"].float().cpu().numpy()
        focal = float(prediction["focallength_px"].cpu())
        h, w = manifest["frames"][i]["height"], manifest["frames"][i]["width"]
        assert inv.shape == depth.shape == (h, w) and np.all(inv >= 0)
        assert np.all(depth > 0)  # Positive or +inf; zero inverse depth is legal.
        expected = np.divide(focal, w * inv, out=np.full_like(inv, np.inf, dtype=np.float32), where=inv > 0)
        finite = np.isfinite(depth) & (depth > 0)
        assert np.allclose(depth[finite], expected[finite], rtol=1e-5, atol=0)
        assert np.all(np.isinf(depth[~finite]))
        frame = manifest["frames"][i]
        np.savez_compressed(out / "depth" / f"{i:06d}.npz", depth=depth.astype(np.float32),
                            K=np.asarray(frame["K"]), c2w=np.asarray(frame["c2w"]),
                            w2c=np.asarray(frame["w2c"]), source_index=i)
        np.save(out / "normalized_inverse_depth" / f"{i:06d}.npy", inv.astype(np.float32))
        values = depth[finite]
        assert len(values)
        metadata["frames"].append({
            "index": i, "source_index": i, "name": frame["name"], "focal_px": focal,
            "nonfinite_depth_fraction": float((~np.isfinite(depth)).mean()),
            "depth_quantiles_m": {str(q): float(np.quantile(values, q)) for q in (.01, .5, .99)},
        })
        metadata["processed_frames"] = i + 1
        if i % 5 == 0 or i == manifest["frame_count"] - 1:
            metadata["inference_seconds_so_far"] = time.time() - start
            metadata["peak_vram_gib_so_far"] = torch.cuda.max_memory_allocated() / 2**30
            write_json(out / "inference.json", metadata)
            print(f"✅ {i + 1}/{manifest['frame_count']} | {time.time() - start:.1f}s | "
                  f"VRAM={metadata['peak_vram_gib_so_far']:.2f}GiB", flush=True)
    assert metadata["processed_frames"] == manifest["frame_count"]
    metadata.update(inference_seconds=time.time() - start,
                    peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30)
    write_json(out / "inference.json", metadata)
    print(f"🎉 Sequence depth: {metadata['processed_frames']} frames -> {out}", flush=True)


if __name__ == "__main__":
    main()
