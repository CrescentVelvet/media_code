#!/usr/bin/env python3
"""Probe RetouchFormer's attention_list + test attention-mask de-color (方案 C),
then restore the same LQ with each trained HYPIR LoRA and stitch everything into
one compare PNG per image.

Three phases (keeps only ONE heavy model on the GPU at a time → low VRAM):
  Phase 1  RetouchFormer + decolor: per image, run beauty + attention_list +
           wavelet/high_freq_dc/C_soft/C_hard, also blur src -> lq. Store all
           tensors in RAM (N images × small tensors).
  Phase 2  HYPIR LoRA: for each LoRA in LORA_PATHS, load SD2Enhancer once, restore
           every image's lq (UPSCALE=1, pure restoration), store results, then free.
  Phase 3  Stitch: per image, cat [src | lq | beauty | mask | C_soft | C_hard |
           high_freq_dc | LoRA1 | LoRA2 | ...] -> probe_<stem>.png.

Env (RetouchFormer): RETOUCH_DIR, WEIGHT_PATH, MODEL_NAME, INPUT_DIR, OUTPUT_DIR, DEVICE
Env (HYPIR LoRA):     LORA_PATHS (comma-sep; each = .pth | checkpoint-N dir | experiment
                      root), HYPIR_DIR, BASE_MODEL_PATH, LORA_RANK, LORA_MODULES,
                      MODEL_T, COEFF_T, UPSCALE(=1), PATCH_SIZE, STRIDE, SEED
Run:
  GPU=0 INPUT_DIR=../HYPIR/input/test_faces_hq OUTPUT_DIR=../../output/hypir_test_results/probe \
    python hypir/decolor_probe.py
  # 限定对比的 LoRA(默认下方三个):
  GPU=0 LORA_PATHS=../HYPIR/experiments/beauty_decolor_20260828 \
    INPUT_DIR=... OUTPUT_DIR=... python hypir/decolor_probe.py
"""
import os
import sys
import time
import importlib
from pathlib import Path

# 直接 python 运行(不走 .sh/_env.sh)时 GPU->CUDA_VISIBLE_DEVICES 映射不存在；
# torch 只认 CUDA_VISIBLE_DEVICES，必须在 import torch / 任何 cuda 调用前设，
# 否则 CUDA context 已初始化、改了无效(进程仍跑默认卡 → OOM)
_gpu = os.environ.get("GPU")
if _gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor

# Reuse 03e's pure helpers (wavelet_*, build_transform, make_blur_fn, decolor_image, IMG_EXTS).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decolor_beauty_dataset as D  # noqa: E402

# ── RetouchFormer ──
MODEL_NAME = os.environ.get("MODEL_NAME", "RetouchFormer")
MODEL_DIR = os.environ.get("MODEL_DIR", "../../model/RetouchFormer")
WEIGHT_PATH = os.environ.get("WEIGHT_PATH") or f"{MODEL_DIR}/release_model/gen_best.pth"
INPUT_DIR = os.environ.get("INPUT_DIR")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./probe_out")
DEVICE = os.environ.get("DEVICE", "cuda")

# ── HYPIR LoRA (Phase 2) ──
# 默认对比 B/C/D 三组美颜实验(各取最新 checkpoint 的 ema 权重)
LORA_PATHS = os.environ.get(
    "LORA_PATHS",
    "../HYPIR/experiments/beauty_ppr50k_20260721,"
    "../HYPIR/experiments/beauty_strong_20260814,"
    "../HYPIR/experiments/beauty_decolor_20260828",
)
HYPIR_DIR = os.environ.get("HYPIR_DIR", "../HYPIR")
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "../../model/HYPIR/sd2_base")
LORA_RANK = int(os.environ.get("LORA_RANK", "256"))
LORA_MODULES = os.environ.get(
    "LORA_MODULES",
    "to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj",
).split(",")
MODEL_T = int(os.environ.get("MODEL_T", "200"))
COEFF_T = int(os.environ.get("COEFF_T", "200"))
UPSCALE = int(os.environ.get("UPSCALE", "1"))   # 1=纯复原不超分(512->512)
PATCH_SIZE_H = int(os.environ.get("PATCH_SIZE", "512"))
STRIDE_H = int(os.environ.get("STRIDE", "256"))
SEED = int(os.environ.get("SEED", "231"))

IMG_EXTS = D.IMG_EXTS


def resolve_lora_path(p):
    """接受 .pth 文件 / checkpoint-N 目录 / 实验根目录 -> 解析到权重文件。
    实验根目录下自动取 step 最大的 checkpoint-N，优先 ema_state_dict.pth。"""
    p = Path(p)
    if p.is_file() and p.suffix == ".pth":
        return p
    if p.is_dir():
        for name in ("ema_state_dict.pth", "state_dict.pth"):
            if (p / name).is_file():
                return p / name
        ckpts = [d for d in p.glob("checkpoint-*") if d.is_dir()]
        ckpts.sort(key=lambda d: int(d.name.split("-")[-1]))
        if ckpts:
            d = ckpts[-1]
            for name in ("ema_state_dict.pth", "state_dict.pth"):
                if (d / name).is_file():
                    return d / name
    sys.exit(f"❌ cannot resolve LoRA path: {p}  (传 .pth / checkpoint-N 目录 / 实验根目录)")


def load_hypir(weight_path):
    """新建一个 SD2Enhancer 并 init_models(加载 SD2 base + LoRA)。"""
    from HYPIR.enhancer.sd2 import SD2Enhancer
    enh = SD2Enhancer(
        base_model_path=BASE_MODEL_PATH,
        weight_path=str(weight_path),
        lora_modules=LORA_MODULES,
        lora_rank=LORA_RANK,
        model_t=MODEL_T,
        coeff_t=COEFF_T,
        device=DEVICE,
    )
    enh.init_models()
    return enh


def hypir_restore(enh, lq_m11):
    """lq_m11: [1,3,H,W] in [-1,1]; SD2Enhancer.enhance 要 [0,1]; 返回 [-1,1] 对齐 lq 尺寸。"""
    lq_01 = (lq_m11 + 1) / 2
    pil = enh.enhance(
        lq=lq_01, prompt="", scale_by="factor", upscale=UPSCALE,
        target_longest_side=None, patch_size=PATCH_SIZE_H, stride=STRIDE_H,
        return_type="pil",
    )[0]
    t = to_tensor(pil).unsqueeze(0).to(lq_m11.device)   # [0,1]
    # UPSCALE=1 应 1:1，但兜底对齐尺寸
    if t.shape[-1] != lq_m11.shape[-1] or t.shape[-2] != lq_m11.shape[-2]:
        t = F.interpolate(t, size=(lq_m11.shape[-2], lq_m11.shape[-1]),
                          mode="bilinear", align_corners=False)
    return t * 2 - 1   # -> [-1,1] 和其他 panel 同范围


def collect_images(input_dir):
    imgs = []
    for r, _, fs in os.walk(input_dir):
        for f in fs:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                imgs.append(Path(r) / f)
    imgs.sort()
    return imgs


def main():
    if not INPUT_DIR:
        sys.exit("❌ set INPUT_DIR (folder of face images; loops all)")
    if not os.path.isfile(WEIGHT_PATH):
        sys.exit(f"❌ RetouchFormer WEIGHT_PATH not found: {WEIGHT_PATH} (set WEIGHT_PATH or MODEL_DIR)")

    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(0)
    device = torch.device(DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[*] device={device}")

    lora_paths_raw = [x.strip() for x in LORA_PATHS.split(",") if x.strip()]
    lora_labels = [Path(x).name for x in lora_paths_raw]
    lora_weights = [resolve_lora_path(x) for x in lora_paths_raw] if lora_paths_raw else []

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    imgs = collect_images(Path(INPUT_DIR))
    if not imgs:
        sys.exit(f"❌ no images in {INPUT_DIR}")

    # ── 加载 RetouchFormer ──
    print(f"\n🏋️ loading RetouchFormer '{MODEL_NAME}'...")
    net = importlib.import_module("model." + MODEL_NAME)
    rt_model = net.InpaintGenerator().to(device)
    print(f"[*] checkpoint: {WEIGHT_PATH}")
    state = torch.load(WEIGHT_PATH, map_location=device, weights_only=False)
    rt_model.load_state_dict(state)
    rt_model.eval()

    tfm = D.build_transform()
    blur_fn = D.make_blur_fn()   # 03e 同款高斯模糊(BLUR_SEED=231)，复现训练 LQ

    # ── 加载所有 HYPIR LoRA enhancer(显存够就一次性加载，per-image 不用切换) ──
    enhancers = []
    if lora_weights:
        print(f"\n🏋️ loading {len(lora_weights)} HYPIR LoRA(s)  (base={BASE_MODEL_PATH})")
        sys.path.insert(0, HYPIR_DIR)   # so `from HYPIR.enhancer.sd2 import ...` resolves
        torch.manual_seed(SEED)
        for lw, lbl in zip(lora_weights, lora_labels):
            print(f"  [{lbl}] {lw}")
            enhancers.append(load_hypir(lw))

    print(f"\n🚀 probe {len(imgs)} images -> {out}\n")
    for idx, fp in enumerate(imgs):
        print(f"{'#' * 64}")
        print(f"### [{idx + 1}/{len(imgs)}] {fp.name}")
        print(f"{'#' * 64}")
        src = tfm(Image.open(fp).convert("RGB")).unsqueeze(0).to(device)  # [1,3,512,512] in [-1,1]
        print(f"[*] src shape={tuple(src.shape)} range=[{src.min():.3f},{src.max():.3f}]")

        with torch.no_grad():
            beauty, attention_list = rt_model(src)   # 接住 attention_list (之前 `beauty, _` 丢了)

        # ── 1. attention_list 格式 ──
        print()
        print("=" * 64)
        print("=== 1. attention_list 格式 ===")
        print("=" * 64)
        print(f"len(attention_list) = {len(attention_list)}")
        # 双引号串里撇号免转义；f-string 表达式部分({}内)不能含反斜杠(Python<3.12)，故提前定义而非内联三元字符串
        TAG_OK = "  (sigmoid'd [0,1])"
        TAG_BAD = "  (NOT [0,1])"
        for i, a in enumerate(attention_list):
            is01 = (a.min() >= -1e-3 and a.max() <= 1 + 1e-3)
            print(f"  [{i}] shape={tuple(a.shape)} dtype={a.dtype} "
                  f"min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f}"
                  f"{TAG_OK if is01 else TAG_BAD}")

        # ── 2. 红润根源: src/lq/beauty per-channel mean ──
        print()
        print("=" * 64)
        print("=== 2. 红润根源: per-channel mean ([-1,1] -> [0,1] 读) ===")
        print("=" * 64)
        print("  ΔR-B > 0 = 偏红暖; 越大越红")
        lq = blur_fn(src)   # [1,3,512,512] in [-1,1] — 训练时的 LQ
        for name, t in [("src(orig)  ", src), ("lq(blur)   ", lq), ("beauty     ", beauty)]:
            m = (t.mean(dim=[0, 2, 3]) + 1) / 2   # [-1,1] -> [0,1]
            print(f"  {name} R={m[0]:.4f} G={m[1]:.4f} B={m[2]:.4f}  ΔR-B={m[0]-m[2]:+.4f}")

        # ── 3. beauty_high per-channel DC (红在高频全局 DC?) ──
        print()
        print("=" * 64)
        print("=== 3. beauty_high per-channel DC (红在高频全局 DC?) ===")
        print("=" * 64)
        print("  理论 ~0; R>0 且 B<0 = 高频带红 DC (high_freq_dc 能去); ~0 = 红是局部, 需 mask")
        beauty_high, _ = D.wavelet_decomposition(beauty)
        src_high, _ = D.wavelet_decomposition(src)
        for name, t in [("beauty_high", beauty_high), ("src_high   ", src_high)]:
            m = t.mean(dim=[0, 2, 3])
            print(f"  {name} R={m[0]:+.5f} G={m[1]:+.5f} B={m[2]:+.5f}")

        # ── 4. 方案 C: attention mask 像素融合 ──
        print()
        print("=" * 64)
        print("=== 4. 方案 C: attention mask 像素融合 ===")
        print("=" * 64)
        # attention_list: 6 个 [B,C,H,W]; 取空间分辨率最大的那个(最精细), C 可能很大
        # (如 512) → channel-mean 降到 [B,1,H,W] 当瑕疵概率图
        best = max(attention_list, key=lambda a: a.shape[-1])
        print(f"  finest scale: shape={tuple(best.shape)}")
        # C 可能很大(如 512): 降维到 [B,1,H,W]. max = 每位置取最像瑕疵的通道(更对,
        # 对应 forward 里 per-channel >0.5 的判断); mean 会稀释(打印作对比). 用 max 融合.
        mask_mean = best.mean(dim=1, keepdim=True)
        mask = best.max(dim=1, keepdim=True)[0]
        print(f"  channel-mean: range=[{mask_mean.min():.4f},{mask_mean.max():.4f}] mean={mask_mean.mean():.4f}")
        print(f"  channel-max : range=[{mask.min():.4f},{mask.max():.4f}] mean={mask.mean():.4f}  <- 用这个")
        blemish = (mask > 0.5).float().mean()
        print(f"  >0.5 blemish area (max): {blemish:.4f}  (小=瑕疵少/正常皮肤多; 大≈1=mask 方向反了, 用 1-mask)")
        # resize 到 512 (attention 在 64x64 潜空间, 上采样做像素 mask)
        if mask.shape[-1] != src.shape[-1]:
            mask = F.interpolate(
                mask, size=(src.shape[-1], src.shape[-1]), mode="bilinear", align_corners=False)
            print(f"  resized mask -> {tuple(mask.shape)} (上采样到像素空间)")
        # soft / hard fusion: blemish region (>0.5) 用 beauty(修复), 正常区 用 src(原图零红润)
        decolor_C_soft = mask * beauty + (1 - mask) * src
        mask_hard = (mask > 0.5).float()
        decolor_C_hard = mask_hard * beauty + (1 - mask_hard) * src
        # high_freq_dc 对比 (已实现)
        decolor_dc = D.decolor_image(beauty, src, "high_freq_dc")

        # ── 5. 各去红方案 per-channel mean ──
        print()
        print("=" * 64)
        print("=== 5. 各去红方案 per-channel mean (ΔR-B 越接近 src 越不红) ===")
        print("=" * 64)
        for name, t in [("src(orig)     ", src),
                        ("lq(blur)      ", lq),
                        ("beauty(红)    ", beauty),
                        ("C_soft        ", decolor_C_soft),
                        ("C_hard        ", decolor_C_hard),
                        ("high_freq_dc  ", decolor_dc)]:
            m = (t.mean(dim=[0, 2, 3]) + 1) / 2
            print(f"  {name} R={m[0]:.4f} G={m[1]:.4f} B={m[2]:.4f}  ΔR-B={m[0]-m[2]:+.4f}")

        # ── 6. HYPIR LoRA 复原(lq -> 各 LoRA 复原图, 看训练后是否带红) ──
        lora_res = []
        if enhancers:
            print()
            print("=" * 64)
            print("=== 6. HYPIR LoRA 复原结果 (ΔR-B 越接近 src 越不红; 接近 beauty=训出红润) ===")
            print("=" * 64)
            with torch.no_grad():
                for enh, lbl in zip(enhancers, lora_labels):
                    t0 = time.time()
                    res = hypir_restore(enh, lq)
                    dt = time.time() - t0
                    m = (res.mean(dim=[0, 2, 3]) + 1) / 2
                    print(f"  {lbl:<28} R={m[0]:.4f} G={m[1]:.4f} B={m[2]:.4f}  "
                          f"ΔR-B={m[0]-m[2]:+.4f}  ({dt:.2f}s)")
                    lora_res.append(res)

        # ── 拼 compare (多行 grid, 避免 N 张拼一行太长) ──
        # 全 512x512; cat(dim=3)=横向, cat(dim=2)=纵向. 末行不足 N_COLS 补黑边(-1)对齐
        mask_vis = (mask * 2 - 1).repeat(1, 3, 1, 1)   # [0,1]->[-1,1]; [1,1,H,W]->[1,3,H,W] 灰度三联, 通道维对齐才能 cat(dim=3)
        panels = [src, lq, beauty, mask_vis, decolor_C_soft, decolor_C_hard, decolor_dc] + lora_res
        labels = ["src(orig)", "lq(blur)", "beauty(red)", "mask(C-max)",
                  "C_soft", "C_hard", "high_freq_dc"] + lora_labels
        n_cols = int(os.environ.get("N_COLS", "5"))   # 默认 5: 10 panel -> 2 行 x 5
        n_cols = max(1, min(n_cols, len(panels)))
        rows, label_rows = [], []
        for r in range(0, len(panels), n_cols):
            chunk = panels[r:r + n_cols]
            rows.append(torch.cat(chunk, dim=3))
            label_rows.append(labels[r:r + n_cols])
        full_w = rows[0].shape[-1]
        for i, row in enumerate(rows):
            if row.shape[-1] < full_w:   # 末行不足补黑边
                pad = torch.full((1, 3, row.shape[-2], full_w - row.shape[-1]),
                                 -1.0, device=row.device)
                rows[i] = torch.cat([row, pad], dim=3)
        cmp = torch.cat(rows, dim=2)   # 行间纵向拼接
        cmp_path = out / f"probe_{fp.stem}.png"
        save_image(cmp, str(cmp_path), normalize=True, value_range=(-1, 1))
        print(f"\n🖼️ saved: {cmp_path}")
        print(f"   grid ({len(rows)} rows x {n_cols} cols):")
        for ri, rl in enumerate(label_rows):
            print(f"   row {ri + 1}: {' | '.join(rl)}")
        print()

    print(f"🎉 Done: {len(imgs)} compare images -> {out}")


if __name__ == "__main__":
    main()
