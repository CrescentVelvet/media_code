#!/usr/bin/env python3
"""noise_negating.py — DINOv2 + MLP 在线动态区域抑制（P0-3）。

在 3DGS 训练过程中，用一个轻量 MLP（384→16→1+Sigmoid）从 DINOv2 语义特征
在线学习每帧的"动态区域"概率图，训练损失只在**静态区域**上计算，从而让高斯
不去拟合跨视角不一致的内容（行人、飘动衣物、曝光跳变等）。

接入官方 gaussian-splatting 的方式（见 train_noise_negate.py）：

    nn = NoiseNegatingModule(scene.getTrainCameras())
    ...
    mask_static = nn.compute_static_mask(cam, H, W, epoch)   # (1,H,W), 无梯度
    Ll1   = masked_l1(image, gt_image, mask_static)
    ssimv = masked_ssim(image, gt_image, mask_static)
    loss  = (1.0 - λ) * Ll1 + λ * (1.0 - ssimv)
    ...
    loss.backward(); optimizer.step()
    nn.update_mlp(cam, image, gt_image, epoch)

约定：mask_static 中 1 = 参与 loss（静态），0 = 屏蔽（动态）。
MLP 输出 mask_mlp 中 1 = dynamic，0 = static（与原实现一致）。

相对 archive 原版实现的修正（每一条都会导致训练崩坏或约束静默失效）：

 1. **MLP 只在 DINO 特征图 (F,F) 上推理**，再把 mask 上采样到图像分辨率。
    原实现先把 384 通道特征插值到 (384,H,W) 再跑 MLP —— 对 1440×1920 图像
    会产生 4GB 级中间激活，实际不可行。同时监督信号（cosine 不相似度）本来
    就在 (F,F) 分辨率，改到 (F,F) 后监督对齐也更正确。
 2. **mask 必须 detach**。否则 3DGS 重建 loss 会顺着 mask 反传到 MLP，MLP
    为了最小化重建 loss 会学会"屏蔽所有高误差区域"，形成对抗性塌缩。
 3. **masked L1 除以静态像素数**而非全图像素数。否则 loss 被系统性缩小，与
    官方 lambda_dssim 配比、densify 梯度阈值的语义全部脱节。
 4. **动态比例分位数兜底**（≤ NN_MAX_DYNAMIC_RATIO）。MLP 随机初始化时输出
    饱和在 0.5，固定阈值 0.25 会让几乎全图判为动态、loss 归零、训练崩塌。
 5. **残差边界项方向原实现写反**：原 `relu(mask-upper)+relu(lower-mask)` 在
    "确定静态"区（lower=upper=1）反而把 mask 推向 1（dynamic）。已按
    mask 语义（1=dynamic）转换为 `relu(mask-(1-lower)) + relu((1-upper)-mask)`。
 6. **masked SSIM 用 ssim map × mask 加权平均**，而不是先把两图乘 mask 再算
    ssim —— 后者在屏蔽区因两图同为 0 而给出 SSIM≈1 的高估，使 loss 虚低。
 7. 正则项方向改为**保守起步**（早期偏向判静态）：原 `2*(1-mask)*decay` 在
    epoch 0 把 MLP 推向全 dynamic；改为 `2*mask*decay`，配合 (4) 的分位数
    兜底，保证场景确实静态时能自动退化成"不屏蔽任何区域"（等价于基线）。
 8. 残差直方图（每 iter CPU histc 2.7M 像素 + 10000-bin cumsum）改为
    **GPU 采样分位数 + 阈值 EMA**，避免每 iter 的 CPU 同步与传输。

Env:
  DINO_MODEL_PATH        DINOv2 权重（.pth 或 repo dir）
  NN_FEATURE_SIZE        DINO 特征图边长 F（默认 36）
  NN_WARMUP_EPOCHS       前 N 个 epoch 用全 1 mask（默认 15）
  NN_MAX_DYNAMIC_RATIO   动态像素比例上限（默认 0.2）
  NN_FIXED_THR           MLP 概率硬阈值下限（默认 0.25）
  NN_DILATE_RADIUS       动态区膨胀半径（默认 10，0=不膨胀）
  NN_MLP_LR              MLP 学习率（默认 1e-3）
  NN_SAMPLE_PIXELS       残差分位数采样像素数（默认 50000）
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DINO_MODEL_PATH = os.environ.get("DINO_MODEL_PATH", "")
DEVICE = os.environ.get("DEVICE", "cuda")


def _fenv(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _ienv(name, default):
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


NN_FEATURE_SIZE = _ienv("NN_FEATURE_SIZE", 36)
NN_WARMUP_EPOCHS = _ienv("NN_WARMUP_EPOCHS", 15)
NN_MAX_DYNAMIC_RATIO = _fenv("NN_MAX_DYNAMIC_RATIO", 0.2)
NN_FIXED_THR = _fenv("NN_FIXED_THR", 0.25)
NN_DILATE_RADIUS = _ienv("NN_DILATE_RADIUS", 10)
NN_MLP_LR = _fenv("NN_MLP_LR", 1e-3)
NN_SAMPLE_PIXELS = _ienv("NN_SAMPLE_PIXELS", 50000)


# ---------------------------------------------------------------------------
# DINOv2 feature extractor
# ---------------------------------------------------------------------------
class DINOFeatureExtractor(nn.Module):
    """Loads DINOv2 ViT-S/14 with registers, extracts patch-level features.

    forward(image, feature_size) -> (384, F, F) where F = feature_size.
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
        # 1. Local .pth checkpoint -> build arch + load
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

        # 2. Local repo dir -> torch.hub source=local
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
        # 1. torch.hub with pretrained=False (uses cached repo, no download)
        try:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg",
                                   pretrained=False, trust_repo=True)
            return model
        except Exception as e:
            print(f"⚠️ torch.hub build failed ({e})")
        # 2. Fallback: timm
        try:
            from timm.models.vision_transformer import VisionTransformer
            return VisionTransformer(
                img_size=518, patch_size=14, embed_dim=384, depth=12,
                num_heads=6, mlp_ratio=4, reg_tokens=4,
            )
        except Exception as e:
            print(f"⚠️ timm build failed ({e})")
        # 3. Last resort: torch.hub with pretrained=True (downloads checkpoint)
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg",
                              trust_repo=True)

    def forward(self, image, feature_size):
        """Extract patch features as a 2D map.

        Args:
            image: (3, H, W) or (B, 3, H, W) tensor in [0, 1].
            feature_size: int — output spatial size F.
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

        # Reshape to feature map: (B, F*F, D) -> (B, F, F, D) -> (B, D, F, F)
        feat_map = patch_tokens.reshape(B, feature_size, feature_size, self.EMBED_DIM)
        feat_map = feat_map.permute(0, 3, 1, 2)  # (B, D, F, F)

        return feat_map.squeeze(0) if B == 1 else feat_map


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------
class MLPModel(nn.Module):
    """384->16->1+Sigmoid. Predicts per-pixel dynamic mask (0=static, 1=dynamic)."""

    def __init__(self, in_dim=384, hidden_dim=16):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """x: (B, 384, H, W) or (384, H, W) -> (B, 1, H, W) or (1, H, W). sigmoid."""
        squeeze = False
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (1, 384, H, W)
            squeeze = True
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        x = F.relu(self.layer1(x))       # (B*H*W, 16)
        x = torch.sigmoid(self.layer2(x))  # (B*H*W, 1)
        x = x.reshape(B, H, W, 1).permute(0, 3, 1, 2)  # (B, 1, H, W)
        return x.squeeze(0) if squeeze else x

    def get_regularizer(self):
        """Product of max |W| across both layers."""
        return self.layer1.weight.abs().max() * self.layer2.weight.abs().max()

    def get_residual_loss(self, mask, lower_mask, upper_mask):
        """Penalize mask values outside the static-consistent band.

        Args:
            mask: flat (N,) — MLP dynamic probability (1 = dynamic).
            lower_mask: flat (N,) — 1 where residual < 60th pct (confident static).
            upper_mask: flat (N,) — 1 where residual < 80th pct (likely static).

        Semantics: confident-static pixels should have mask -> 0, and
        high-residual (dynamic) pixels should have mask -> 1. Converting the
        static-bound indicators into a band for the *dynamic* probability:

            upper bound = 1 - lower_mask   (confident static -> bound 0)
            lower bound = 1 - upper_mask   (high residual   -> bound 1)

        loss = mean(ReLU(mask - upper_bound) + ReLU(lower_bound - mask))

        NOTE: the archived implementation used
        `ReLU(mask - upper_mask) + ReLU(lower_mask - mask)`, which in a
        confident-static region (lower=upper=1) pushes mask -> 1, i.e. it
        *labels static pixels as dynamic*. That is inverted; fixed here.
        """
        upper_bound = 1.0 - lower_mask
        lower_bound = 1.0 - upper_mask
        return (F.relu(mask - upper_bound) + F.relu(lower_bound - mask)).mean()


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------
def _neighborhood_vote(mask, kernel=3):
    """kxk majority vote smoothing for a binary mask (H, W)."""
    m = mask.unsqueeze(0).unsqueeze(0)
    pad = kernel // 2
    m = F.avg_pool2d(m, kernel_size=kernel, stride=1, padding=pad)
    return (m.squeeze(0).squeeze(0) > 0.5).float()


def dilate_black_region(mask, radius=15):
    """Dilate the 0 (dynamic) regions of a mask (H, W) by `radius` pixels.

    Expands dynamic regions so the static/dynamic border has a wider no-loss
    zone, preventing border artifacts.
    """
    inverted = (1.0 - mask).unsqueeze(0).unsqueeze(0)  # 1=dynamic
    kernel = 2 * radius + 1
    inverted = F.max_pool2d(inverted, kernel_size=kernel, stride=1, padding=radius)
    return (1.0 - inverted).squeeze(0).squeeze(0)


_WINDOW_CACHE = {}


def _gaussian_window(window_size, channel, device, dtype):
    key = (window_size, channel, device)
    w = _WINDOW_CACHE.get(key)
    if w is None:
        x = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
        g = torch.exp(-(x ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        w2 = g[:, None] * g[None, :]
        w = w2.expand(channel, 1, window_size, window_size).contiguous()
        _WINDOW_CACHE[key] = w
    return w.to(dtype)


def ssim_map(img1, img2, window_size=11):
    """Per-pixel SSIM map for two (3, H, W) images in [0, 1] -> (H, W).

    Same constants as utils.loss_utils._ssim (C1=0.01^2, C2=0.03^2).
    """
    channel = img1.shape[0]
    window = _gaussian_window(window_size, channel, img1.device, img1.dtype)
    x1 = img1.unsqueeze(0)
    x2 = img2.unsqueeze(0)
    pad = window_size // 2
    mu1 = F.conv2d(x1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(x2, window, padding=pad, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(x1 * x1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(x2 * x2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(x1 * x2, window, padding=pad, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    s = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return s.mean(1).squeeze(0)  # (H, W)


def masked_l1(pred_image, gt_image, mask_static):
    """L1 restricted to static pixels, normalised by the static pixel count.

    Normalising by `mask.sum()` (not H*W) keeps the loss on the same scale as
    the official unmasked L1, so lambda_dssim and the densify gradient
    threshold keep their usual meaning.
    """
    m = mask_static.detach()
    if m.dim() == 2:
        m = m.unsqueeze(0)
    diff = (pred_image - gt_image).abs().mean(dim=0, keepdim=True)  # (1,H,W)
    return (diff * m).sum() / m.sum().clamp(min=1.0)


def masked_ssim(pred_image, gt_image, mask_static):
    """SSIM averaged over static pixels only.

    Computed on the *unmasked* images so the local statistics inside the masked
    region stay correct, then averaged with the static mask as weights.
    (Masking the images before SSIM would make both zero inside the masked
    region and report SSIM ~ 1 there, understating the loss.)
    """
    m = mask_static.detach()
    if m.dim() == 2:
        m = m.unsqueeze(0)
    s = ssim_map(pred_image, gt_image)  # (H, W)
    return (s * m.squeeze(0)).sum() / m.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Noise-negating module (stateful; one instance per training run)
# ---------------------------------------------------------------------------
class NoiseNegatingModule:
    """Online DINOv2 + MLP dynamic-mask learner.

    Usage inside the official training loop:
        nn = NoiseNegatingModule(scene.getTrainCameras())
        mask_static = nn.compute_static_mask(cam, H, W, epoch)   # (1,H,W)
        ... compute masked loss, backward, optimizer.step() ...
        nn.update_mlp(cam, rendered_image, gt_image, epoch)
    """

    def __init__(self, train_cameras, device=None, feature_size=None):
        self.device = device or (DEVICE if torch.cuda.is_available() else "cpu")
        self.feature_size = int(feature_size or NN_FEATURE_SIZE)
        self.warmup_epochs = NN_WARMUP_EPOCHS
        self.max_dynamic_ratio = NN_MAX_DYNAMIC_RATIO
        self.fixed_thr = NN_FIXED_THR
        self.dilate_radius = NN_DILATE_RADIUS
        self.sample_pixels = NN_SAMPLE_PIXELS

        print("  📐 loading DINOv2 ViT-S/14 reg (from D drive, ~2 min)...")
        self.extractor = DINOFeatureExtractor(device=self.device)

        # Precompute GT features for every training camera (cached on CPU).
        self.features = {}
        print(f"  🔍 precomputing GT features for {len(train_cameras)} cameras "
              f"(F={self.feature_size})...")
        for cam in train_cameras:
            gt = cam.original_image
            if torch.is_tensor(gt):
                gt = gt.to(self.device)
            else:
                gt = torch.tensor(np.asarray(gt), device=self.device).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                feat = self.extractor(gt, self.feature_size).cpu()
            self.features[cam.image_name] = feat
        print(f"  ✅ GT features cached ({len(self.features)} cameras)")

        self.mlp = MLPModel(in_dim=384, hidden_dim=16).to(self.device)
        self.opt = torch.optim.Adam(self.mlp.parameters(), lr=NN_MLP_LR)

        # Residual percentile thresholds (EMA across iterations).
        self.ema_lower = None
        self.ema_upper = None

        # Monitoring.
        self.last_static_ratio = torch.ones(1, device=self.device)
        self.last_mlp_loss = 0.0
        self.n_cameras = max(1, len(train_cameras))

    # -- static mask used by the 3DGS loss ---------------------------------
    def compute_static_mask(self, camera, H, W, epoch):
        """Return (1, H, W) static mask (1 = contributes to loss). Detached.

        During warmup (epoch < NN_WARMUP_EPOCHS) returns all-ones, i.e. the
        plain official loss, so the gaussians get a clean start.
        """
        if epoch < self.warmup_epochs:
            return torch.ones(1, H, W, device=self.device)

        feat = self.features[camera.image_name].to(self.device)  # (384, F, F)
        with torch.no_grad():
            prob = self.mlp(feat)                                # (1, F, F)

            # Adaptive threshold: cap the dynamic fraction so the MLP can never
            # blank out the whole image (which would zero the loss).
            q = torch.quantile(prob.reshape(-1), 1.0 - self.max_dynamic_ratio)
            thr = torch.clamp(q, min=self.fixed_thr)
            dyn = (prob > thr).float()                           # (1, F, F)

            dyn_up = F.interpolate(
                dyn.unsqueeze(0), size=(H, W), mode="bilinear",
                align_corners=False).squeeze(0)                  # (1, H, W)
            static = 1.0 - dyn_up

            if self.dilate_radius > 0:
                k = 2 * self.dilate_radius + 1
                static = -F.max_pool2d(
                    -static, kernel_size=k, stride=1, padding=self.dilate_radius)

        self.last_static_ratio = static.mean().detach()
        return static.detach()

    # -- residual bounds ----------------------------------------------------
    def _residual_bounds(self, pred, gt):
        """Static soft bounds from the render-vs-GT residual.

        Returns (lower_mask, upper_mask), each (H, W), 1 = low residual.
        Thresholds are quantiles of a random pixel sample, smoothed with an
        EMA across iterations (replaces the per-iteration CPU histogram).
        """
        residual = (pred - gt).abs().mean(dim=0)  # (H, W)
        flat = residual.reshape(-1)
        n = flat.numel()
        k = min(n, self.sample_pixels)
        idx = torch.randint(0, n, (k,), device=flat.device)
        samp = flat[idx]
        lo = torch.quantile(samp, 0.60).detach()
        up = torch.quantile(samp, 0.80).detach()
        if self.ema_lower is None:
            self.ema_lower = lo
            self.ema_upper = up
        else:
            self.ema_lower = 0.95 * self.ema_lower + 0.05 * lo
            self.ema_upper = 0.95 * self.ema_upper + 0.05 * up

        lower_mask = _neighborhood_vote((residual < self.ema_lower).float(), 3)
        upper_mask = _neighborhood_vote((residual < self.ema_upper).float(), 3)
        return lower_mask, upper_mask

    # -- online MLP update (called after the 3DGS optimizer step) -----------
    def update_mlp(self, camera, pred_image, gt_image, epoch):
        """Train the MLP to predict dynamic regions.

        Only MLP parameters receive gradients here: `pred_image` is detached
        (no gradient flows back into the gaussians) and the render features are
        computed under no_grad.
        """
        # Periodic skip after epoch 30 to limit over-fitting to recent frames.
        if epoch >= 30:
            reset_start = (epoch // 5) * 5
            if reset_start <= epoch < reset_start + 1:
                return

        pred = pred_image.detach()
        gt = gt_image.detach()
        Fsz = self.feature_size
        H, W = gt.shape[1], gt.shape[2]

        gt_feat = self.features[camera.image_name].to(self.device)  # (384, F, F)

        # MLP forward on the (F, F) feature map (cheap: F^2 pixels).
        mask_mlp = self.mlp(gt_feat)                                # (1, F, F)

        with torch.no_grad():
            render_feat = self.extractor(pred, Fsz)                 # (384, F, F)

        # Residual bounds: (H, W) -> (F, F) to align with the MLP output.
        lower_mask, upper_mask = self._residual_bounds(pred, gt)
        lower_f = F.adaptive_avg_pool2d(
            lower_mask[None, None], (Fsz, Fsz)).squeeze()
        upper_f = F.adaptive_avg_pool2d(
            upper_mask[None, None], (Fsz, Fsz)).squeeze()

        # Cosine dissimilarity between GT and rendered DINO features.
        cos = F.cosine_similarity(gt_feat, render_feat, dim=0, eps=1e-6)  # (F,F)
        dissim = (((1.0 - cos) - 0.5) / 0.5).clamp(0, 1).unsqueeze(0)     # (1,F,F)

        decay = math.exp(-epoch / 4.0)
        # Conservative prior: early on, push the mask towards "static".
        reg = 0.5 * self.mlp.get_regularizer() + 2.0 * (mask_mlp * decay).mean()
        res_loss = self.mlp.get_residual_loss(
            mask_mlp.reshape(-1), lower_f.reshape(-1), upper_f.reshape(-1))
        loss = (0.5 * (mask_mlp - dissim).abs().mean()
                + 0.5 * res_loss
                + reg)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        self.last_mlp_loss = 0.1 * loss.item() + 0.9 * self.last_mlp_loss


def nn_initial(train_cameras, **kwargs):
    """Convenience wrapper kept from the original API -> NoiseNegatingModule."""
    return NoiseNegatingModule(train_cameras, **kwargs)
