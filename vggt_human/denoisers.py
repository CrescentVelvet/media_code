#!/usr/bin/env python3
"""denoisers.py — Pluggable denoising model registry.

Each denoiser is a function: (image: np.ndarray [0,1], device: str) -> np.ndarray [0,1].
Models are lazy-loaded on first call and cached. Switch via env DENOISER=diffbir|swinir|nafnet|none.

Add a new denoiser:
  1. Write a function denoise_my_model(image, device, **kw) -> np.ndarray
  2. Register it: DENOISERS["my_model"] = denoise_my_model
  3. Use: DENOISER=my_model bash 05_denoise_novel.sh
"""
import os
import sys
import numpy as np

_cache = {}


# ---------------------------------------------------------------------------
# DiffBIR — blind image restoration with generative diffusion prior.
# Repo: https://github.com/csxliang/DiffBIR
# ---------------------------------------------------------------------------
def denoise_diffbir(image, device="cuda", **kw):
    """DiffBIR: two-stage (restoration + diffusion refinement)."""
    if "diffbir" not in _cache:
        import torch
        diffbir_dir = os.environ.get("DIFFBIR_DIR", "../DiffBIR")
        if not os.path.isdir(diffbir_dir):
            sys.exit(f"❌ DiffBIR repo not found at {diffbir_dir}. "
                     f"Run: INSTALL_DENOISER=1 bash vggt_human/00_setup_env.sh")
        sys.path.insert(0, diffbir_dir)
        import yaml
        from ldm.util import instantiate_from_config  # noqa: E402

        ckpt = os.environ.get("DIFFBIR_CKPT", "")
        if not os.path.isfile(ckpt):
            sys.exit(f"❌ DiffBIR checkpoint not found: {ckpt}. "
                     f"Run: INSTALL_DENOISER=1 bash vggt_human/00_setup_env.sh")

        config_path = os.environ.get(
            "DIFFBIR_CONFIG",
            os.path.join(diffbir_dir, "configs/inference/cldm.yaml"))
        print(f"  🏋️ loading DiffBIR: {ckpt}")
        config = yaml.safe_load(open(config_path, "r"))
        model = instantiate_from_config(config["model"])
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state), strict=False)
        model = model.to(device).eval()
        _cache["diffbir"] = model
        print("  ✅ DiffBIR loaded")

    model = _cache["diffbir"]
    import torch
    with torch.no_grad():
        t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        t = (t - 0.5) / 0.5  # [0,1] -> [-1,1]
        t = t.to(device)
        # DiffBIR's test() returns restored image in [-1, 1]
        out = model.test(lq=t)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        out = (out + 1) / 2  # [-1,1] -> [0,1]
        return out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


# ---------------------------------------------------------------------------
# SwinIR — image restoration using Swin Transformer.
# Repo: https://github.com/JingyunLiang/SwinIR
# Uses the real-image-denoising variant by default.
# ---------------------------------------------------------------------------
def denoise_swinir(image, device="cuda", **kw):
    """SwinIR: single-forward denoising (faster than diffusion-based DiffBIR)."""
    if "swinir" not in _cache:
        import torch
        swinir_dir = os.environ.get("SWINIR_DIR", "../SwinIR")
        if not os.path.isdir(swinir_dir):
            sys.exit(f"❌ SwinIR repo not found at {swinir_dir}. "
                     f"Run: INSTALL_DENOISER=1 bash vggt_human/00_setup_env.sh")
        sys.path.insert(0, os.path.join(swinir_dir, "models"))
        from network_swinir import SwinIR  # noqa: E402

        ckpt = os.environ.get("SWINIR_CKPT", "")
        if not os.path.isfile(ckpt):
            sys.exit(f"❌ SwinIR checkpoint not found: {ckpt}. "
                     f"Run: INSTALL_DENOISER=1 bash vggt_human/00_setup_env.sh")

        # Real-image denoising architecture (from SwinIR repo, ClassSRImageDemo)
        model = SwinIR(
            upscale=1, in_chans=3, img_size=64, window_size=8,
            img_range=1.0, depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2, resi_connection="1conv",
            upsampler="",  # no upsampling (denoising, not SR)
        )
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("params", state), strict=True)
        model = model.to(device).eval()
        _cache["swinir"] = model
        print("  ✅ SwinIR loaded")

    model = _cache["swinir"]
    import torch
    with torch.no_grad():
        t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        # Tile-based inference for large images (SwinIR repo's approach)
        out = _tile_inference(model, t, tile_size=512, overlap=32)
        return out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def _tile_inference(model, x, tile_size=512, overlap=32):
    """Process large images in overlapping tiles to avoid OOM."""
    import torch
    _, _, H, W = x.shape
    if H <= tile_size and W <= tile_size:
        return model(x)
    stride = tile_size - overlap
    out = torch.zeros_like(x)
    weight = torch.zeros_like(x)
    for h in range(0, H, stride):
        for w in range(0, W, stride):
            h2, w2 = min(h + tile_size, H), min(w + tile_size, W)
            tile = x[:, :, h:h2, w:w2]
            out[:, :, h:h2, w:w2] += model(tile)
            weight[:, :, h:h2, w:w2] += 1
    return out / weight.clamp(min=1)


# ---------------------------------------------------------------------------
# NAFNet — stub (implement when needed).
# ---------------------------------------------------------------------------
def denoise_nafnet(image, device="cuda", **kw):
    raise NotImplementedError(
        "❌ NAFNet denoiser not implemented yet.\n"
        "   Add a denoise_nafnet function to denoisers.py and register it.\n"
        "   Reference: https://github.com/megvii-research/NAFNet")


# ---------------------------------------------------------------------------
# Identity (no denoising).
# ---------------------------------------------------------------------------
def denoise_none(image, device="cuda", **kw):
    """Identity — DENOISER=none (renders novel views but skips denoising)."""
    return image


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------
DENOISERS = {
    "diffbir": denoise_diffbir,
    "swinir":  denoise_swinir,
    "nafnet":  denoise_nafnet,
    "none":    denoise_none,
}


def get_denoiser(name):
    fn = DENOISERS.get(name)
    if fn is None:
        raise ValueError(
            f"❌ unknown DENOISER='{name}'. Available: {list(DENOISERS)}")
    return fn
