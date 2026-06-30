#!/usr/bin/env python3
"""Real dataset for HunyuanVideo-1.5 training (drop-in for train.py's dummy).

The official train.py ships a placeholder `create_dummy_dataloader()`. This
module provides a real dataset that reads video/image clips + text prompts from
a directory, and a `create_dataloader()` with the same signature the launcher
(train_run.py) injects in place of the dummy.

Dataset layout (DATA_DIR):
    DATA_DIR/
    ├── captions.jsonl          # preferred: {"file": "videos/x.mp4", "text": "..."}
    ├── videos/*.mp4            # video clips (data_type="video")
    ├── images/*.{png,jpg,jpeg,webp,bmp}   # stills (data_type="image")
    └── prompts/<stem>.txt      # per-clip caption (used when no captions.jsonl)

Caption resolution order:
  1. captions.jsonl  (file paths relative to DATA_DIR; data_type inferred from ext)
  2. prompts/<stem>.txt for each clip under videos/ or images/
  3. the file stem (underscores -> spaces) as a fallback prompt

Sample format returned by __getitem__ (matches what train.py expects):
  {"pixel_values": Tensor[3, F, H, W] in [-1,1] float32 (video)
                   or Tensor[3, H, W]   in [-1,1] float32 (image),
   "text": str,
   "data_type": "video" | "image"}
F must be 4n+1 (e.g. 1, 5, 9, ..., 41, 121) for the causal VAE.

Env vars (read by create_dataloader):
  TRAIN_VIDEO_LENGTH (default 41)   must be 4n+1
  TRAIN_RESOLUTION   (default 480p) "480p"->(480,848) "720p"->(720,1280)
  TRAIN_HEIGHT / TRAIN_WIDTH        override resolution (must be divisible by 16)
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def _sample_indices(total, num_frames):
    """Evenly-spaced frame indices in [0, total-1] (with repeats if total<n)."""
    if total <= 1:
        return [0] * num_frames
    return np.linspace(0, total - 1, num_frames).round().astype(int).tolist()


def _via_decord(path, num_frames):
    import decord
    vr = decord.VideoReader(path, num_threads=1)
    total = len(vr)
    if total == 0:
        return None
    idxs = _sample_indices(total, num_frames)
    return vr.get_batch(idxs).asnumpy()  # [n,H,W,3] uint8


def _via_imageio(path, num_frames):
    import imageio
    with imageio.get_reader(path, "ffmpeg") as reader:
        frames = np.stack([f for f in reader])  # read all [T,H,W,3] uint8
    total = len(frames)
    if total == 0:
        return None
    idxs = _sample_indices(total, num_frames)
    return frames[idxs]


def _via_torchvision(path, num_frames):
    from torchvision.io import read_video
    frames, _, _ = read_video(path, pts_unit="sec")  # [T,H,W,3] uint8
    total = int(frames.shape[0])
    if total == 0:
        return None
    idxs = _sample_indices(total, num_frames)
    return frames[idxs].numpy()


def _load_video_sample(path, num_frames):
    last = None
    for fn in (_via_decord, _via_imageio, _via_torchvision):
        try:
            out = fn(path, num_frames)
            if out is not None and out.shape[0] == num_frames:
                return out[..., :3]  # drop alpha if any
        except Exception as e:  # pragma: no cover - environment dependent
            last = e
    raise RuntimeError(f"failed to read video {path}: {last}")


class _EpochedDistributedSampler(DistributedSampler):
    """DistributedSampler that reshuffles each new epoch (train.py re-iterates
    the dataloader every epoch; the base sampler would otherwise keep the same
    order). No dist.init_process_group needed at construction (rank/num_replicas
    taken from torchrun env vars)."""

    def __iter__(self):
        it = super().__iter__()
        self.epoch += 1
        return it


class HunyuanVideoDataset(Dataset):
    def __init__(self, data_dir, video_length=41, height=480, width=848):
        if (video_length - 1) % 4 != 0:
            raise ValueError(
                f"TRAIN_VIDEO_LENGTH must be 4n+1 (got {video_length}); the causal VAE requires it."
            )
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"HxW must be divisible by 16 (got {height}x{width}); the VAE requires it.")
        self.data_dir = data_dir
        self.video_length = video_length
        self.height = height
        self.width = width
        self.samples = self._index()
        if not self.samples:
            raise RuntimeError(
                f"No samples found under {data_dir}. Expected captions.jsonl, or "
                f"videos/*.mp4 (+ optional prompts/<stem>.txt). See train_dataset.py docstring."
            )

    def _index(self):
        root = self.data_dir
        samples = []

        cap_jsonl = os.path.join(root, "captions.jsonl")
        if os.path.isfile(cap_jsonl):
            with open(cap_jsonl, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    file = obj.get("file") or obj.get("path") or obj.get("video") or obj.get("image")
                    if not file:
                        continue
                    text = obj.get("text") or obj.get("prompt") or obj.get("caption") or ""
                    ext = os.path.splitext(file)[1].lower()
                    dt = "image" if ext in IMG_EXTS else "video"
                    if not text:
                        text = os.path.splitext(os.path.basename(file))[0].replace("_", " ").strip()
                    samples.append((file, text, dt))
            return samples

        stem_text = {}
        for pdir in ("prompts", "captions"):
            d = os.path.join(root, pdir)
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    if fn.lower().endswith(".txt"):
                        stem = os.path.splitext(fn)[0]
                        with open(os.path.join(d, fn), encoding="utf-8") as f:
                            stem_text[stem] = f.read().strip()

        for sub, dt, exts in (("videos", "video", VIDEO_EXTS), ("images", "image", IMG_EXTS)):
            d = os.path.join(root, sub)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if os.path.splitext(fn)[1].lower() in exts:
                    stem = os.path.splitext(fn)[0]
                    text = stem_text.get(stem, stem.replace("_", " ").strip())
                    samples.append((os.path.join(sub, fn), text, dt))

        # Last resort: scan the root for clips (no videos/ subfolder).
        if not samples:
            for fn in sorted(os.listdir(root)):
                ext = os.path.splitext(fn)[1].lower()
                if ext in VIDEO_EXTS or ext in IMG_EXTS:
                    stem = os.path.splitext(fn)[0]
                    dt = "image" if ext in IMG_EXTS else "video"
                    text = stem_text.get(stem, stem.replace("_", " ").strip())
                    samples.append((fn, text, dt))
        return samples

    def __len__(self):
        return len(self.samples)

    def _to_neg1_1(self, arr_uint8):
        # arr_uint8: [H,W,3] or [n,H,W,3] uint8 -> float tensor in [-1,1]
        t = torch.from_numpy(np.asarray(arr_uint8, dtype=np.float32) / 255.0)
        if t.ndim == 3:  # [H,W,3] -> [3,H,W]
            t = t.permute(2, 0, 1)
        else:            # [n,H,W,3] -> [3,n,H,W]
            t = t.permute(3, 0, 1, 2)
        return t * 2.0 - 1.0

    def __getitem__(self, idx):
        relpath, text, data_type = self.samples[idx]
        path = os.path.join(self.data_dir, relpath)
        if data_type == "image":
            img = Image.open(path).convert("RGB").resize((self.width, self.height))
            t = self._to_neg1_1(np.asarray(img, dtype=np.uint8))
            return {"pixel_values": t.contiguous(), "text": text, "data_type": "image"}

        frames = _load_video_sample(path, self.video_length)  # [n,H,W,3] uint8
        t = self._to_neg1_1(frames)  # [3,n,H,W]
        if t.shape[-2:] != (self.height, self.width):
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(self.height, self.width),
                mode="bilinear", align_corners=False, antialias=True,
            )[0]
        return {"pixel_values": t.contiguous(), "text": text, "data_type": "video"}


def create_dataloader(config, data_dir):
    """Build a DataLoader over DATA_DIR. Signature matches what train_run.py
    injects as `create_dummy_dataloader` in the official train.py."""
    video_length = int(os.environ.get("TRAIN_VIDEO_LENGTH", "41"))
    res = os.environ.get("TRAIN_RESOLUTION", "480p")
    if res == "480p":
        h, w = 480, 848
    elif res == "720p":
        h, w = 720, 1280
    else:
        h, w = 480, 848
    if os.environ.get("TRAIN_HEIGHT"):
        h = int(os.environ["TRAIN_HEIGHT"])
    if os.environ.get("TRAIN_WIDTH"):
        w = int(os.environ["TRAIN_WIDTH"])

    dataset = HunyuanVideoDataset(data_dir, video_length=video_length, height=h, width=w)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[train_dataset] {len(dataset)} samples under {data_dir} "
              f"(video_length={video_length}, {h}x{w}, dtype inferred per item)")

    world = int(os.environ.get("WORLD_SIZE", "1"))
    num_workers = getattr(config, "num_workers", 4)
    if world > 1:
        sampler = _EpochedDistributedSampler(
            dataset, num_replicas=world, rank=int(os.environ["RANK"]), shuffle=True)
        return DataLoader(dataset, batch_size=config.batch_size, sampler=sampler,
                          num_workers=num_workers, drop_last=True, persistent_workers=num_workers > 0)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True, persistent_workers=num_workers > 0)
