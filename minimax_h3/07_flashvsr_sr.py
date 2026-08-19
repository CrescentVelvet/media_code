#!/usr/bin/env python3
"""FlashVSR 视频超分：把 int8 量化 MiniMax 生成的低分辨率视频 4× 超分到高清。

用 OpenImagingLab/FlashVSR（CVPR 2026，one-step diffusion streaming VSR）。
默认 v1.1 + full pipeline（最高质量）；PIPELINE=tiny_long 切到 tiny + 长视频流式
管线（低显存 / 长视频用，需 TCDecoder.ckpt）。

输入是视频文件（.mp4/.mov/...）或图片帧目录；输出 SR 视频到 OUTPUT_DIR。
MiniMax 视频带原生立体声——MUX_AUDIO=1（默认）时把原视频音频 mux 回 SR 输出。

Env vars:
  INPUT (必填)         : 输入视频路径或帧目录
  FLASHVSR_MODEL_DIR   : 权重目录（含 diffusion_pytorch_model_streaming_dmd.safetensors 等）
  FLASHVSR_DIR         : 官方仓路径（import utils.* 用，由 .sh 设）
  PIPELINE             : full (默认) | tiny_long
  SCALE                : 4 (4× 超分)
  SEED                 : 0
  SPARSE_RATIO         : 2.0 (1.5 更快 / 2.0 更稳；LCSA 稀疏度)
  LOCAL_RANGE          : 11 (9 更锐 / 11 更稳)
  TILED                : 0 (full 专用；1=低显存分块，更慢)
  COLOR_FIX            : 1
  MUX_AUDIO            : 1 (把原视频音频 mux 回 SR 输出)
  DEVICE               : cuda (GPU 选卡用 GPU=N，由 _env.sh 映射 CUDA_VISIBLE_DEVICES)
  OUTPUT_DIR / OUTPUT_NAME
"""
import os, sys, re, time, subprocess, shutil


def _add_wanvsr_path():
    """把 examples/WanVSR 加到 sys.path 最前，让 `from utils.utils import ...` 命中官方 utils/。"""
    flash_dir = os.environ.get("FLASHVSR_DIR", "")
    p = os.path.join(flash_dir, "examples", "WanVSR") if flash_dir else ""
    if p and os.path.isdir(p):
        sys.path.insert(0, p)


_add_wanvsr_path()

try:
    from diffsynth import ModelManager
    import torch
    import numpy as np
    from PIL import Image
    import imageio
    from tqdm import tqdm
    from einops import rearrange
except ImportError as e:
    sys.exit(f"❌ {e}. Run: INSTALL_DEPS=1 CONDA_ENV=flashvsr bash minimax_h3/07_flashvsr_sr.sh")


# 权重文件名（v1 / v1.1 通用）
DIT = "diffusion_pytorch_model_streaming_dmd.safetensors"
LQ = "LQ_proj_in.ckpt"
VAE = "Wan2.1_VAE.pth"
TCD = "TCDecoder.ckpt"


# ─────────────────────── 视频 / 图像工具（取自官方 infer_flashvsr_v1.1_*.py） ──
def tensor2video(frames: torch.Tensor):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    return [Image.fromarray(frame) for frame in frames]


def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'([0-9]+)', os.path.basename(name))]


def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs


def largest_8n1_leq(n):  # 8n+1（官方约定：F = 8n+1，含 4 帧 padding）
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def is_video(path):
    return os.path.isfile(path) and path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))


def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)  # HWC
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0                                               # CHW in [-1,1]
    return t.to(dtype)


def save_video(frames, save_path, fps=30, quality=5):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
        w.append_data(np.array(f))
    w.close()


def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128):
    if w0 <= 0 or h0 <= 0 or scale <= 0:
        raise ValueError(f"invalid size {w0}x{h0} scale={scale}")
    sW, sH = int(round(w0 * scale)), int(round(h0 * scale))
    tW, tH = (sW // multiple) * multiple, (sH // multiple) * multiple
    if tW == 0 or tH == 0:
        raise ValueError(f"scaled {sW}x{sH} too small for multiple={multiple}; increase SCALE")
    return sW, sH, tW, tH


def upscale_then_center_crop(img: Image.Image, scale: float, tW: int, tH: int) -> Image.Image:
    w0, h0 = img.size
    sW, sH = int(round(w0 * scale)), int(round(h0 * scale))
    if tW > sW or tH > sH:
        raise ValueError(f"target crop {tW}x{tH} exceeds scaled {sW}x{sH}; increase SCALE")
    up = img.resize((sW, sH), Image.BICUBIC)
    l, t = (sW - tW) // 2, (sH - tH) // 2
    return up.crop((l, t, l + tW, t + tH))


def prepare_input_tensor(path: str, scale: float = 4, dtype=torch.bfloat16,
                         device='cuda', frame_device=None):
    """读视频 / 帧目录 -> BICUBIC 放大 scale× + 中心裁剪到 128 倍数 -> 1,C,F,H,W 张量。
    frame_device=None 时用 device；tiny_long 传 'cpu' 省显存。"""
    fdev = frame_device if frame_device is not None else device

    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"no images in {path}")
        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        N0 = len(paths0)
        print(f"  🖼️ {os.path.basename(path)}: {w0}x{h0} | {N0} frames")
        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale)
        print(f"  📐 scaled x{scale}: {sW}x{sH} -> target (128-mul): {tW}x{tH}")
        paths = paths0 + [paths0[-1]] * 4
        F = largest_8n1_leq(len(paths))
        if F == 0:
            raise RuntimeError(f"not enough frames after padding: {len(paths)}")
        paths = paths[:F]
        print(f"  🎬 target frames (8n-3): {F - 4}")
        frames = []
        for p in paths:
            with Image.open(p).convert('RGB') as img:
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, fdev))
        vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
        return vid, tH, tW, F, 30

    if is_video(path):
        rdr = imageio.get_reader(path)
        first = Image.fromarray(rdr.get_data(0)).convert('RGB')
        w0, h0 = first.size
        meta = {}
        try:
            meta = rdr.get_meta_data()
        except Exception:
            pass
        fps_val = meta.get('fps', 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames(r):
            try:
                nf = meta.get('nframes', None)
                if isinstance(nf, int) and nf > 0:
                    return nf
            except Exception:
                pass
            try:
                return r.count_frames()
            except Exception:
                n = 0
                try:
                    while True:
                        r.get_data(n); n += 1
                except Exception:
                    return n

        total = count_frames(rdr)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"cannot read frames from {path}")
        print(f"  🖼️ {os.path.basename(path)}: {w0}x{h0} | {total} frames | {fps}fps")
        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale)
        print(f"  📐 scaled x{scale}: {sW}x{sH} -> target (128-mul): {tW}x{tH}")
        idx = list(range(total)) + [total - 1] * 4
        F = largest_8n1_leq(len(idx))
        if F == 0:
            rdr.close()
            raise RuntimeError(f"not enough frames after padding: {len(idx)}")
        idx = idx[:F]
        print(f"  🎬 target frames (8n-3): {F - 4}")
        frames = []
        try:
            for i in idx:
                img = Image.fromarray(rdr.get_data(i)).convert('RGB')
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
                frames.append(pil_to_tensor_neg1_1(img_out, dtype, fdev))
        finally:
            try:
                rdr.close()
            except Exception:
                pass
        vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
        return vid, tH, tW, F, fps

    raise ValueError(f"unsupported input: {path}")


# ─────────────────────── pipeline 初始化 ─────────────────────────────────────
def init_pipeline(pipeline: str, model_dir: str, device: str):
    print(f"📦 loading FlashVSR {pipeline} pipeline...")
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")

    if pipeline == "full":
        from diffsynth import FlashVSRFullPipeline
        from utils.utils import Causal_LQ4x_Proj
        mm.load_models([
            os.path.join(model_dir, DIT),
            os.path.join(model_dir, VAE),
        ])
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(
            in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=torch.bfloat16)
        lq_path = os.path.join(model_dir, LQ)
        if os.path.exists(lq_path):
            pipe.denoising_model().LQ_proj_in.load_state_dict(
                torch.load(lq_path, map_location="cpu"), strict=True)
        pipe.denoising_model().LQ_proj_in.to(device)
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
        pipe.to(device); pipe.enable_vram_management(num_persistent_param_in_dit=None)
        pipe.init_cross_kv(); pipe.load_models_to_device(["dit", "vae"])
        return pipe

    # tiny_long
    from diffsynth import FlashVSRTinyLongPipeline
    from utils.utils import Causal_LQ4x_Proj
    from utils.TCDecoder import build_tcdecoder
    mm.load_models([os.path.join(model_dir, DIT)])
    pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(
        in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=torch.bfloat16)
    lq_path = os.path.join(model_dir, LQ)
    if os.path.exists(lq_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(
            torch.load(lq_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768)
    mis = pipe.TCDecoder.load_state_dict(
        torch.load(os.path.join(model_dir, TCD)), strict=False)
    print(f"  TCDecoder: missing={len(mis.missing_keys)} unexpected={len(mis.unexpected_keys)}")
    pipe.to(device); pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(); pipe.load_models_to_device(["dit", "vae"])
    return pipe


# ─────────────────────── 音频 mux（保留 MiniMax 原生立体声） ──────────────────
def mux_audio(sr_path: str, src_path: str):
    if not shutil.which("ffmpeg"):
        print("⚠️ ffmpeg not found, skipping audio mux")
        return
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-i", src_path],
                           capture_output=True, text=True, timeout=60)
    except Exception as e:
        print(f"⚠️ probe failed: {e}")
        return
    if "Audio:" not in (r.stderr or ""):
        print("⏭️ no audio stream in source, skip mux")
        return
    tmp = sr_path + ".mux.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", sr_path, "-i", src_path,
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
             "-shortest", tmp],
            check=True, capture_output=True, text=True, timeout=900,
        )
        os.replace(tmp, sr_path)
        print(f"🎵 audio muxed from {os.path.basename(src_path)}")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ mux failed: {(e.stderr or str(e))[-400:]}")
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception as e:
        print(f"⚠️ mux error: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)


# ─────────────────────── main ─────────────────────────────────────────────────
def main():
    INPUT = os.environ.get("INPUT", "")
    if not INPUT:
        sys.exit("❌ INPUT not set (video path or frame directory)")
    if not os.path.exists(INPUT):
        sys.exit(f"❌ INPUT not found: {INPUT}")

    MODEL_DIR = os.environ.get("FLASHVSR_MODEL_DIR", "../../model/FlashVSR")
    PIPELINE = os.environ.get("PIPELINE", "full").lower()
    SCALE = float(os.environ.get("SCALE", "4"))
    SEED = int(os.environ.get("SEED", "0"))
    SPARSE_RATIO = float(os.environ.get("SPARSE_RATIO", "2.0"))
    LOCAL_RANGE = int(os.environ.get("LOCAL_RANGE", "11"))
    TILED = os.environ.get("TILED", "0") == "1"
    COLOR_FIX = os.environ.get("COLOR_FIX", "1") == "1"
    MUX_AUDIO = os.environ.get("MUX_AUDIO", "1") == "1"
    DEVICE = os.environ.get("DEVICE", "cuda")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "../MiniMax-H3/results_sr")
    in_name = os.path.splitext(os.path.basename(INPUT.rstrip('/')))[0] or "input"
    OUTPUT_NAME = os.environ.get("OUTPUT_NAME") or f"{in_name}_sr.mp4"

    # 权重文件检查
    need = [DIT, LQ]
    if PIPELINE == "full":
        need.append(VAE)
    elif PIPELINE == "tiny_long":
        need.append(TCD)
    else:
        sys.exit(f"❌ unknown PIPELINE={PIPELINE} (full | tiny_long)")
    missing = [f for f in need if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        sys.exit(f"❌ missing weights in {MODEL_DIR}: {missing}")

    print(f"🚀 [07] FlashVSR 4× SR ({PIPELINE} pipeline)")
    print(f"  🖼️ input: {INPUT}")
    print(f"  🏋️ model: {MODEL_DIR}")
    print(f"  📐 scale={SCALE}×  seed={SEED}  sparse_ratio={SPARSE_RATIO}  local_range={LOCAL_RANGE}")
    print(f"  🎮 device: {DEVICE}")

    # full: 帧放 GPU；tiny_long: 帧留 CPU（省显存，管线流式搬）
    frame_device = "cpu" if PIPELINE == "tiny_long" else DEVICE
    LQ, th, tw, F, fps = prepare_input_tensor(
        INPUT, scale=SCALE, dtype=torch.bfloat16, device=DEVICE, frame_device=frame_device)
    print(f"  📐 target: {tw}x{th} (128-multiple)  {F - 4} frames @ {fps}fps")

    pipe = init_pipeline(PIPELINE, MODEL_DIR, DEVICE)

    t0 = time.time()
    kwargs = dict(
        prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=SEED,
        LQ_video=LQ, num_frames=F, height=th, width=tw,
        is_full_block=False, if_buffer=True,
        topk_ratio=SPARSE_RATIO * 768 * 1280 / (th * tw),
        kv_ratio=3.0,
        local_range=LOCAL_RANGE,
        color_fix=COLOR_FIX,
    )
    if PIPELINE == "full":
        kwargs["tiled"] = TILED  # True=低显存分块（更慢）
    print("🎬 super-resolving...")
    video = pipe(**kwargs)
    print(f"  ⏱️ SR: {time.time() - t0:.0f}s")

    frames = tensor2video(video)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    save_video(frames, out_path, fps=fps, quality=6)
    sz = os.path.getsize(out_path)
    print(f"✅ SR saved: {out_path} ({sz / 1e6:.1f} MB)")

    if MUX_AUDIO and is_video(INPUT):
        mux_audio(out_path, INPUT)
    print("🎉 [07] Done.")


if __name__ == "__main__":
    main()
