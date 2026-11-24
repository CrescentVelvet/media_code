"""Replay the archived three-view P0 and tap the official voxel merge input."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "archive", "output"):
        ap.add_argument(name, type=Path)
    args = ap.parse_args()
    run = json.loads((args.archive / "run.json").read_text())
    assert run["mode"] == "pose" and not any(run.get(k) for k in (
        "external_depth_dir", "sky_seg", "no_cam_align")), "Expected the reference-pose P0 arm"
    sys.path.insert(0, str(args.repo.resolve()))
    import inference
    from src.models.models.rasterization import GaussianSplatRenderer

    args.output.mkdir(parents=True, exist_ok=True)
    width, height = run["resolution"]
    original = GaussianSplatRenderer.merge_voxels

    def capture(renderer, splats, keep):
        assert keep.shape == (1, 3 * width * height)
        pixel = torch.where(keep[0])[0]
        data = {k: v[0, pixel].cpu().numpy() for k, v in splats.items()}
        np.savez(args.output / "pre.npz", pixel=pixel.cpu().numpy(), **data)
        merged = original(renderer, splats, keep)
        np.savez(args.output / "merged.npz", **{k: val[0].cpu().numpy() for k, val in merged.items()})
        return merged

    with patch.object(GaussianSplatRenderer, "merge_voxels", capture):
        inference.main(["--config_path", str(args.archive / "config.yaml"), "--mode", "pose",
                        "--input_path", run["input_path"], "--output_path", str(args.output / "replay"),
                        "--resolution", str(width), str(height)])


if __name__ == "__main__":
    main()
