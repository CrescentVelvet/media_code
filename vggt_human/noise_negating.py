#!/usr/bin/env python3
"""noise_negating.py — DINOv2 + MLP online dynamic mask learning (P0-3, core).

During 3DGS training, a lightweight MLP (384→16→1+Sigmoid) learns per-pixel
dynamic masks from DINOv2 semantic features. The MLP is trained by comparing
GT and rendered image features (cosine dissimilarity) + residual histogram
bounds. The learned mask replaces SegTrack's initial mask after epoch 15,
and a dynamic-aware loss (L1+SSIM only on static regions) replaces standard
L1+SSIM.

Components:
  - DINOFeatureExtractor: loads DINOv2 ViT-S/14 reg, extracts patch features
  - MLPModel: 384→16→1+Sigmoid, with get_regularizer() + get_residual_loss()
  - calculate_residual_mask: histogram-based upper/lower bounds (EMA)
  - nn_initial: precompute GT features + init MLP + optimizer + histogram
  - nn_loss: dynamic-aware loss (static-only L1+SSIM)
  - mlp_update: online MLP training (cosine dissim + residual bounds + reg)
  - dilate_black_region: dilate the 0 (dynamic) regions of a binary mask

Conventions:
  - MLP output: 0=static, 1=dynamic
  - mask_negate (in nn_loss): 1=static (compute loss), 0=dynamic (skip)

Env vars:
  DINO_MODEL_PATH  : DINOv2 checkpoint dir or .pth (default: $MODEL_DIR/dinov2)
  DEVICE           : cuda | cpu (default: cuda)
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DINO_MODEL_PATH = os.environ.get("DINO_MODEL_PATH", "")
DEVICE = os.environ.get("DEVICE", "cuda")


# ---------------------------------------------------------------------------
# DINOv2 feature extractor
# ---------------------------------------------------------------------------
class DINOFeatureExtractor(nn.Module):
    """Loads DINOv2 ViT-S/14 with registers, extracts patch-level features.

    forward(image, feature_size) → (384, F, F) where F = feature_size.
    The image is bilinearly resized to (F*14, F*14), then patch tokens are
    reshaped to a 2D feature map.
    """

    PATCH_SIZE = 14
    EMBED_DIM = 384  # ViT-S

    def __init__(self, model_path="", device="cuda"):
        super().__init__()
        self.device = device
        model = self._load_model(model_path or DINO_MODEL_PATH)
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model = model

    def _load_model(self, model_path):
        # 1. Local .pth checkpoint → build arch + load
        if model_path and os.path.isfile(model_path):
            try:
                model = self._build_arch()
                state = torch.load(model_path, map_location="cpu")
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                model.load_state_dict(state, strict=False)
                print(f"🏋️ DINOv2 loaded from local checkpoint: {model_path}")
                return model
            except Exception as e:
                print(f"⚠️ DINOv2 local checkpoint load failed: {e}")

        # 2. Local repo dir → torch.hub source=local
        if model_path and os.path.isdir(model_path):
            try:
                model = torch.hub.load(
                    model_path, "dinov2_vits14_reg", source="local")
                print(f"🏋️ DINOv2 loaded from local repo: {model_path}")
                return model
            except Exception as e:
                print(f"⚠️ DINOv2 local repo load failed: {e}")

        # 3. torch.hub download
        try:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
            print("🏋️ DINOv2 loaded from torch.hub (downloaded)")
            return model
        except Exception as e:
            sys.exit(
                f"❌ DINOv2 not available ({e}). Options:\n"
                f"  1. pip install torch  (torch.hub will download)\n"
                f"  2. Set DINO_MODEL_PATH to a local .pth or repo dir\n"
                f"  3. Clone: git clone https://github.com/facebookresearch/dinov2.git"
            )

    def _build_arch(self):
        """Build DINOv2 ViT-S/14 reg architecture (for loading local weights)."""
        try:
            from timm.models.vision_transformer import VisionTransformer
            model = VisionTransformer(
                img_size=518, patch_size=14, embed_dim=384, depth=12,
                num_heads=6, mlp_ratio=4, reg_tokens=4,
                block_fn=None,
            )
            return model
        except Exception:
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")

    def forward(self, image, feature_size):
        """Extract patch features as a 2D map.

        Args:
            image: (3, H, W) or (B, 3, H, W) tensor in [0, 1].
            feature_size: int — output spatial size F (36 fine, 16 coarse).
        Returns:
            (384, F, F) if B==1, else (B, 384, F, F).
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)
        B = image.shape[0]

        # Resize to (F*14, F*14) for exact F*F patches
        target = feature_size * self.PATCH_SIZE
        img_resized = F.interpolate(
            image, size=(target, target), mode="bilinear", align_corners=False)

        # Forward
        features = self.model.forward_features(img_resized)

        # Extract patch tokens (format varies: dict or tensor)
        n_patches = feature_size * feature_size
        if isinstance(features, dict):
            patch_tokens = features["x_norm_patchtokens"]
        else:
            patch_tokens = features[:, -n_patches:, :]

        # Reshape to feature map: (B, F*F, D) → (B, F, F, D) → (B, D, F, F)
        feat_map = patch_tokens.reshape(B, feature_size, feature_size, self.EMBED_DIM)
        feat_map = feat_map.permute(0, 3, 1, 2)  # (B, D, F, F)

        return feat_map.squeeze(0) if B == 1 else feat_map


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------
class MLPModel(nn.Module):
    """384→16→1+Sigmoid. Predicts per-pixel dynamic mask (0=static, 1=dynamic)."""

    def __init__(self, in_dim=384, hidden_dim=16):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """x: (B, 384, H, W) or (384, H, W) → (B, 1, H, W) or (1, H, W). sigmoid."""
        squeeze = False
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (1, 384, H, W)
            squeeze = True
        B, C, H, W = x.shape
        # (B, C, H, W) → (B, H, W, C) → (B*H*W, C)
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        x = F.relu(self.layer1(x))       # (B*H*W, 16)
        x = torch.sigmoid(self.layer2(x))  # (B*H*W, 1)
        x = x.reshape(B, H, W, 1).permute(0, 3, 1, 2)  # (B, 1, H, W)
        return x.squeeze(0) if squeeze else x

    def get_regularizer(self):
        """Product of max |W| across both layers."""
        return self.layer1.weight.abs().max() * self.layer2.weight.abs().max()

    def get_residual_loss(self, mask, lower_mask, upper_mask):
        """Penalize mask values outside [lower, upper] bounds.

        All inputs are flat tensors of the same length.
        loss = mean(ReLU(mask - upper) + ReLU(lower - mask))
        """
        return (F.relu(mask - upper_mask) + F.relu(lower_mask - mask)).mean()


# ---------------------------------------------------------------------------
# Residual histogram upper/lower bounds
# ---------------------------------------------------------------------------
def calculate_residual_mask(gt_image, render_image, cum_hist):
    """Compute lower/upper static masks from pixel residuals + histogram EMA.

    Args:
        gt_image: (3, H, W) tensor.
        render_image: (3, H, W) tensor.
        cum_hist: (10000,) tensor — EMA cumulative histogram.
    Returns:
        lower_mask: (H, W) float — 1 where residual < 60th percentile (likely static).
        upper_mask: (H, W) float — 1 where residual < 80th percentile (likely static/low-dyn).
        cum_hist: updated (10000,) tensor.
    """
    # 1. Per-pixel residual (mean abs across channels)
    residual = (gt_image - render_image).abs().mean(dim=0)  # (H, W)

    # 2. Histogram of current residuals
    error_hist = torch.histc(
        residual.float().cpu(), bins=10000, min=0, max=1)

    # 3. EMA update
    cum_hist = 0.95 * cum_hist + error_hist

    # 4. CDF
    cdf = torch.cumsum(cum_hist, dim=0)
    cdf = cdf / cdf[-1].clamp(min=1)

    # 5. Percentile thresholds
    lower_idx = (cdf >= 0.60).int().argmax().item()
    upper_idx = (cdf >= 0.80).int().argmax().item()
    lower_threshold = lower_idx / 10000.0
    upper_threshold = upper_idx / 10000.0

    # 6. Masks (1 = below threshold = likely static)
    lower_mask = (residual < lower_threshold).float()
    upper_mask = (residual < upper_threshold).float()

    # 7. 3x3 neighborhood voting (smooth via avg-pool + threshold)
    lower_mask = _neighborhood_vote(lower_mask, kernel=3)
    upper_mask = _neighborhood_vote(upper_mask, kernel=3)

    return lower_mask, upper_mask, cum_hist


def _neighborhood_vote(mask, kernel=3):
    """3x3 (or kxk) majority vote smoothing for a binary mask (H, W)."""
    m = mask.unsqueeze(0).unsqueeze(0)
    pad = kernel // 2
    m = F.avg_pool2d(m, kernel_size=kernel, stride=1, padding=pad)
    return (m.squeeze(0).squeeze(0) > 0.5).float()


# ---------------------------------------------------------------------------
# Static region dilation
# ---------------------------------------------------------------------------
def dilate_black_region(mask, radius=15):
    """Dilate the 0 (dynamic/black) regions of a mask (H, W) by `radius` pixels.

    Expands dynamic regions so the static/dynamic border has a wider no-loss
    zone, preventing border artifacts.
    """
    inverted = (1.0 - mask).unsqueeze(0).unsqueeze(0)  # 1=dynamic
    kernel = 2 * radius + 1
    inverted = F.max_pool2d(inverted, kernel_size=kernel, stride=1, padding=radius)
    return (1.0 - inverted).squeeze(0).squeeze(0)


# ---------------------------------------------------------------------------
# Initialization (precompute GT features + init MLP + optimizer + histogram)
# ---------------------------------------------------------------------------
def nn_initial(train_cameras):
    """Precompute GT DINOv2 features for all cameras + init MLP + histogram.

    Args:
        train_cameras: list of Camera objects (with .image_name, .original_image).
    Returns:
        mlp_model, mlp_optimizer, feature_extractor,
        features_fine (dict {name: (384, 36, 36)} on CPU),
        features_coarse (dict {name: (384, 16, 16)} on CPU),
        historical_hist (tensor (10000,)).
    """
    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load DINOv2
    print("  📐 loading DINOv2 ViT-S/14 reg...")
    extractor = DINOFeatureExtractor(device=device)

    # Precompute GT features (fine 36 + coarse 16), cache on CPU
    features_fine = {}
    features_coarse = {}
    print(f"  🔍 precomputing GT features for {len(train_cameras)} cameras...")
    for cam in train_cameras:
        name = cam.image_name
        gt = cam.original_image  # (3, H, W) on device
        with torch.no_grad():
            feat_fine = extractor(gt, 36).cpu()    # (384, 36, 36)
            feat_coarse = extractor(gt, 16).cpu()  # (384, 16, 16)
        features_fine[name] = feat_fine
        features_coarse[name] = feat_coarse
    print(f"  ✅ GT features cached ({len(features_fine)} fine + {len(features_coarse)} coarse)")

    # Init MLP
    mlp_model = MLPModel(in_dim=384, hidden_dim=16).to(device)
    mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)

    # Init histogram
    historical_hist = torch.zeros(10000)

    return mlp_model, mlp_optimizer, extractor, features_fine, features_coarse, historical_hist


# ---------------------------------------------------------------------------
# Dynamic-aware loss (replaces standard L1+SSIM)
# ---------------------------------------------------------------------------
def nn_loss(camera, mlp_model, features, pred_image, gt_image, epoch, dynamic_masks):
    """Compute dynamic-aware loss: L1+SSIM only on static regions.

    Args:
        camera: Camera object (with .image_name).
        mlp_model: MLPModel.
        features: dict {name: (384, 36, 36)} — fine GT features (on CPU).
        pred_image: rendered image (3, H, W) — differentiable (connected to 3DGS).
        gt_image: GT image (3, H, W).
        epoch: int — current epoch (iteration // num_cameras).
        dynamic_masks: dict {name: mask(H,W) uint8} — SegTrack initial masks.
    Returns:
        loss_negate: scalar tensor — dynamic-aware loss.
        mask_mlp: (1, H, W) tensor — MLP prediction (0=static, 1=dynamic).
    """
    cam_name = camera.image_name
    H, W = gt_image.shape[1], gt_image.shape[2]
    dev = pred_image.device

    # 1. MLP prediction (differentiable — gradient flows to MLP params)
    gt_feature = features[cam_name].to(dev)  # (384, 36, 36)
    upsample = F.interpolate(
        gt_feature.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
    mask_mlp = mlp_model(upsample.squeeze(0))  # (1, H, W) ∈ [0, 1]

    # 2. Determine static mask (1=static/compute loss, 0=dynamic/skip)
    if epoch < 15:
        # Use SegTrack initial mask
        if dynamic_masks and cam_name in dynamic_masks:
            init_mask = dynamic_masks[cam_name]
            mask_t = torch.tensor(
                init_mask, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
            mask_t = F.interpolate(mask_t, size=(H, W), mode="nearest")
            mask_negate = 1.0 - mask_t.squeeze(0).squeeze(0)  # 1=static
        else:
            mask_negate = torch.ones(H, W, device=dev)
    else:
        # Use MLP prediction (threshold 0.25 → 1 where static)
        mask_negate = (mask_mlp.squeeze(0) <= 0.25).float()  # 1=static
        # Dilate dynamic regions (expand 0s) via min-pool = -maxpool(-x)
        mn = mask_negate.unsqueeze(0).unsqueeze(0)
        mask_negate = (-F.max_pool2d(
            -mn, kernel_size=7, stride=1, padding=3)).squeeze(0).squeeze(0)
        # Further dilate dynamic regions (wider no-loss zone at borders)
        mask_negate = dilate_black_region(mask_negate, radius=15)

    # 3. Dynamic-aware loss (only on static regions)
    l1_nn = (mask_negate * (pred_image - gt_image).abs().mean(dim=0)).mean()
    # Masked SSIM: multiply both images by mask before SSIM
    mask3 = mask_negate.unsqueeze(0)  # (1, H, W)
    ssim_val = _ssim_masked(mask3 * pred_image, mask3 * gt_image)
    ssim_nn = 1.0 - ssim_val
    loss_negate = 0.8 * l1_nn + 0.2 * ssim_nn

    return loss_negate, mask_mlp


def _ssim_masked(x, y):
    """SSIM for (3, H, W) masked images. Falls back to L1 if ssim unavailable."""
    try:
        from utils.loss_utils import ssim
        return ssim(x, y).mean()
    except Exception:
        return 1.0 - (x - y).abs().mean()


# ---------------------------------------------------------------------------
# MLP online update (after 3DGS optimizer step)
# ---------------------------------------------------------------------------
def mlp_update(epoch, camera, mlp_model, mask_mlp, mlp_optimizer,
               features_fine, features_coarse, feature_extractor,
               pred_image, gt_image, historical_hist):
    """Online MLP update: train MLP to match cosine dissimilarity + residual bounds.

    Args:
        epoch: int.
        camera: Camera object.
        mlp_model: MLPModel.
        mask_mlp: (1, H, W) — previous MLP prediction (from nn_loss, graph may be freed).
        mlp_optimizer: torch.optim.Adam.
        features_fine: dict {name: (384, 36, 36)} on CPU.
        features_coarse: dict {name: (384, 16, 16)} on CPU.
        feature_extractor: DINOFeatureExtractor.
        pred_image: rendered image (3, H, W) — detached.
        gt_image: GT image (3, H, W).
        historical_hist: tensor (10000,).
    """
    # Periodic skip after epoch 30 (prevent overfitting)
    if epoch >= 30:
        reset_start = (epoch // 5) * 5
        reset_end = reset_start + 1
        if reset_start <= epoch < reset_end:
            return

    cam_name = camera.image_name
    dev = pred_image.device
    H, W = gt_image.shape[1], gt_image.shape[2]

    # 1. Select feature resolution
    if epoch >= 20:
        gt_feature = features_fine[cam_name].to(dev)   # (384, 36, 36)
        F_size = 36
    else:
        gt_feature = features_coarse[cam_name].to(dev)  # (384, 16, 16)
        F_size = 16

    # Recompute mask_mlp (differentiable — fresh graph for backward)
    upsample = F.interpolate(
        gt_feature.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
    mask_mlp = mlp_model(upsample.squeeze(0))  # (1, H, W) ∈ [0, 1]

    # 2. Render features (detached — no grad to 3DGS)
    with torch.no_grad():
        render_feature = feature_extractor(pred_image, F_size)  # (384, F, F)

    # 3. Residual upper/lower bounds
    lower_mask, upper_mask, historical_hist = calculate_residual_mask(
        gt_image, pred_image, historical_hist)
    lower_flat = lower_mask.flatten().to(dev)
    upper_flat = upper_mask.flatten().to(dev)

    # 4. Cosine dissimilarity
    cosine_sim = F.cosine_similarity(gt_feature, render_feature, dim=0, eps=1e-6)
    if cosine_sim.ndim == 2:
        cosine_sim = cosine_sim.unsqueeze(0)  # (1, F, F)
    # Map: cosine_sim ∈ [0,1] → dissim ∈ [0,1], only nonzero when sim < 0.5
    cosine_dissim = ((1.0 - cosine_sim) - 0.5) / 0.5  # = 1 - 2*sim
    cosine_dissim = cosine_dissim.clamp(0, 1)         # (1, F, F)
    # Upsample to image resolution
    cosine_dissim = F.interpolate(
        cosine_dissim, size=(H, W), mode="bilinear", align_corners=False)  # (1, H, W)

    # 5. MLP loss
    reg_loss = 0.5 * mlp_model.get_regularizer()
    decay = torch.exp(torch.tensor(-epoch / 4.0, device=dev))
    reg_loss = reg_loss + 2.0 * ((1 - mask_mlp) * decay).mean()
    residual_loss = mlp_model.get_residual_loss(
        mask_mlp.flatten(), lower_flat, upper_flat)
    mask_loss = (
        0.5 * (mask_mlp - cosine_dissim).abs().mean()
        + 0.5 * residual_loss
        + reg_loss
    )

    # 6. Backward + step
    mask_loss.backward()
    mlp_optimizer.step()
    mlp_optimizer.zero_grad()
