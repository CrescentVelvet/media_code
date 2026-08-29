#!/usr/bin/env python3
"""06d — MiniMax-H3 int8 + Turbo LoRA 常驻 HTTP 服务（单卡 3090 最优解）。

= 06c 的 int8 + block offload 加载（权重 ~62GB 塞进 56GB+swap，3090 24GB 够）
+ 06b 的 Turbo 4 步蒸馏 LoRA（50 步→4 步，~10× 快）。

为什么单独写 06d：06c 单卡 3090 能跑但 50 步很慢（每步搬运 block）；06b 4 步快
但 bf16 全量 ~124GB RAM，3090+64GB 跑不动。06d 取两者之长——int8 压 RAM（62GB）
+ Turbo 4 步省算力，3090 上又快又能跑。

与 06c 的 API 完全一致，但去噪只用 4 步（由启动时的 LoRA checkpoint 决定，
不按请求改——shift/NFE 与 LoRA 配方绑定，改了效果会崩）：
  GET  /health     → {"status":"ready","busy":false}
  POST /generate   → {prompt, first_frame, last_frame, num_frames, seed,
                       max_pixels, fps, width, height, output_name}
  POST /shutdown

Turbo 配方（启动时固定，见 06b.sh 注释的 checkpoint 表）：
  默认 768p 4-step v1.0：NFE=4 VIDEO_SHIFT=6 LORA_ALPHA=128 MAX_PIXELS=1032192(1344x768)
  544p 4-step v0.1   ：NFE=4 VIDEO_SHIFT=12 LORA_ALPHA=8 MAX_PIXELS=522240(960x544)

Env vars:
  MODEL_PATH, PORT, DEVICE, OUTPUT_DIR, MAX_PIXELS, FPS, NUM_FRAMES,
  LORA_PATH, NUM_INFERENCE_STEPS, VIDEO_SHIFT, AUDIO_SHIFT, LORA_ALPHA, LORA_SCALE, FUSE_LORA
"""
import os, sys, time, json, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# ── 1. 解析环境变量 ──────────────────────────────────────────────────────
MODEL_PATH = os.path.abspath(os.environ.get("MODEL_PATH", "/mnt/d/wheel/minimaxh3_ms"))
PORT = int(os.environ.get("PORT", "8000"))
DEVICE = os.environ.get("DEVICE", "cuda:0")
OUTPUT_DIR = os.path.abspath(os.environ.get(
    "OUTPUT_DIR", "/mnt/d/output/minimaxh3_rotate_results/results_int8turbo"))
DEFAULT_MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(512 * 768)))
DEFAULT_FPS = int(os.environ.get("FPS", "24"))
DEFAULT_NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "124"))
# Turbo 蒸馏参数（默认 768p 4-step v1.0 配方；与 06b 同源）
LORA_PATH = os.environ.get("LORA_PATH", "")
DEFAULT_NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "4"))
VIDEO_SHIFT = float(os.environ.get("VIDEO_SHIFT", "6.0"))   # 768p 4-step 用 6；544p 用 12
AUDIO_SHIFT = float(os.environ.get("AUDIO_SHIFT", "3.0"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "128"))       # 768p 4-step v1.0 训练用 128；544p 用 8
LORA_SCALE = float(os.environ.get("LORA_SCALE", "1.0"))
FUSE_LORA = os.environ.get("FUSE_LORA", "0") == "1"

# ── 2. 全局状态 ──────────────────────────────────────────────────────────
pipe = None
gen_lock = threading.Lock()
busy = {"value": False}
stats = {"generations": 0, "last_path": "", "last_time_s": 0, "last_res": ""}


def load_model():
    """加载 int8 量化 pipeline + Turbo LoRA + block offload（~20-40min 从 D 盘）。"""
    from diffusers import ModularPipeline, MiniMaxH3Transformer3DModel, TorchAoConfig
    from diffusers.hooks import apply_group_offloading
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig
    from _turbo_lora import load_lora_adapter
    import torch

    if not LORA_PATH:
        sys.exit("❌ LORA_PATH not set (Turbo LoRA, e.g. minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors)")
    if not os.path.isdir(MODEL_PATH):
        sys.exit(f"❌ MODEL_PATH not found: {MODEL_PATH}")
    if not os.path.isfile(os.path.join(MODEL_PATH, "model_index.json")):
        sys.exit(f"❌ model_index.json not found in {MODEL_PATH} — incomplete download?")
    # int8 + FUSE_LORA 互斥：fuse 要把 bf16 LoRA delta 加进 Int8Tensor 权重，safe_fusing 会直接报错
    if FUSE_LORA:
        sys.exit("❌ FUSE_LORA=1 与 int8 不兼容（融不进量化权重）；06d 用 set_adapters 运行时合并，无需 fuse")

    print(f"🚀 [06d] Loading MiniMax-H3 int8 + Turbo from {MODEL_PATH}")
    print(f"  🏋️ lora: {LORA_PATH}")
    print(f"  ⏱️ NFE: {DEFAULT_NUM_INFERENCE_STEPS} (grid={DEFAULT_NUM_INFERENCE_STEPS+1})  shifts: video={VIDEO_SHIFT} audio={AUDIO_SHIFT}")
    print(f"  🎮 device: {DEVICE}")
    print(f"  💾 output: {OUTPUT_DIR}")
    print(f"  ⏳ loading from D drive (HDD) — this takes 20-40 min...")

    from _ensure_modular_index import ensure_modular_model_index
    print(f"  📦 {ensure_modular_model_index(MODEL_PATH)}")

    t0 = time.time()
    p = ModularPipeline.from_pretrained(MODEL_PATH)
    p.update_components(
        transformer=MiniMaxH3Transformer3DModel.from_pretrained(
            MODEL_PATH, subfolder="transformer", dtype=torch.bfloat16,
            quantization_config=TorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=[
                    "proj_in", "audio_proj_in", "context_embedder", "time_embedder",
                    "time_proj", "token_refiner", "norm_out", "proj_out", "audio_proj_out",
                ],
            ),
            low_cpu_mem_usage=True,
        ),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, subfolder="text_encoder", dtype=torch.bfloat16,
            quantization_config=TransformersTorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=[
                    "model.visual", "model.language_model.embed_tokens",
                    "model.language_model.norm", "lm_head",
                ],
            ),
            low_cpu_mem_usage=True,
        ),
    )
    p.load_components(workflow="fl2va", dtype=torch.bfloat16,
                      pretrained_model_name_or_path=MODEL_PATH)
    p.transformer.requires_grad_(False)
    p.text_encoder.requires_grad_(False)

    # Turbo shift（必须在生成前设；与 LoRA 配方绑定）
    p.scheduler.set_shift(VIDEO_SHIFT)
    p.audio_scheduler.set_shift(AUDIO_SHIFT)

    # 注入 Turbo LoRA：必须在 enable_group_offload 之前，否则新增的 lora_A/B 子模块
    # 没被 offload hook 覆盖，前向时 block 搬到 GPU 而 LoRA 留在 CPU → device 不匹配
    load_lora_adapter(p.transformer, LORA_PATH, LORA_ALPHA, LORA_SCALE, FUSE_LORA)

    # block offload（同 06c；此时 LoRA 已在 transformer 内，hook 一并覆盖）
    offload = dict(onload_device=torch.device(DEVICE), offload_device=torch.device("cpu"))
    p.transformer.enable_group_offload(
        offload_type="block_level", num_blocks_per_group=1,
        low_cpu_mem_usage=True,  # Int8Tensor 不支持 pin_memory()，跳过
        **offload,
    )
    apply_group_offloading(
        p.text_encoder.model, offload_type="leaf_level",
        low_cpu_mem_usage=True, **offload,
    )
    p.vae.to(DEVICE)
    p.audio_vae.to(DEVICE)

    dt = time.time() - t0
    print(f"  ✅ pipeline loaded in {dt/60:.1f} min (int8 + Turbo {DEFAULT_NUM_INFERENCE_STEPS}-step, block offload on {DEVICE})")
    return p


def auto_resolution(first_frame, max_pixels, width, height):
    """从首帧图算 auto 分辨率（同 06c/06a）。"""
    if width > 0 and height > 0:
        return width, height
    import math
    if first_frame:
        from diffusers.utils import load_image as _li
        _img = _li(first_frame)
        iw, ih = _img.size
        ratio = iw / ih
        print(f"  📐 auto ratio from image: {iw}x{ih} -> {ratio:.2f}")
    else:
        ratio = 16 / 9
    if ratio >= 1:
        h = int(math.sqrt(max_pixels / ratio))
        w = int(h * ratio)
    else:
        r2 = 1 / ratio
        w = int(math.sqrt(max_pixels / r2))
        h = int(w * r2)
    w = max(32, w - (w % 32))
    h = max(32, h - (h % 32))
    return w, h


def generate(req):
    """执行一次 Turbo 生成（调用时已持锁）。返回 dict 结果。"""
    import torch
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video

    prompt = req.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}
    first_frame = req.get("first_frame", "")
    last_frame = req.get("last_frame", "")
    num_frames = int(req.get("num_frames", DEFAULT_NUM_FRAMES))
    seed = int(req.get("seed", 0))
    max_pixels = int(req.get("max_pixels", DEFAULT_MAX_PIXELS))
    fps = int(req.get("fps", DEFAULT_FPS))
    width = int(req.get("width", 0))
    height = int(req.get("height", 0))
    output_name = req.get("output_name", "")

    task = "fl2va" if first_frame else "t2va"
    if not output_name:
        output_name = f"{task}_int8turbo_seed{seed}.mp4"

    width, height = auto_resolution(first_frame, max_pixels, width, height)
    print(f"  🎯 task={task}  frames={num_frames}  res={width}x{height}  seed={seed}  nfe={DEFAULT_NUM_INFERENCE_STEPS}")

    generator = torch.Generator(DEVICE).manual_seed(seed)
    outputs = ["videos", "audio", "sampling_rate"]
    kwargs = dict(
        prompt=prompt, num_frames=num_frames, generator=generator, output=outputs,
        # MiniMaxH3Scheduler 把 num_inference_steps 解释为含末尾零点的 sigma grid，
        # N 次去噪要传 N+1（照 Turbo 仓库 inference_minimax_h3.py）
        num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS + 1,
    )
    if width > 0:
        kwargs["width"] = width
    if height > 0:
        kwargs["height"] = height
    if first_frame:
        kwargs["image"] = load_image(first_frame)
    if last_frame:
        kwargs["last_image"] = load_image(last_frame)

    t0 = time.time()
    print(f"  🎬 generating (Turbo {DEFAULT_NUM_INFERENCE_STEPS}-step)...")
    with torch.inference_mode():
        results = pipe(**kwargs)
    dt = time.time() - t0

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, output_name)
    encode_video(
        results["videos"][0],
        fps=fps,
        output_path=out_path,
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    sz = os.path.getsize(out_path)
    print(f"  ✅ saved: {out_path} ({sz/1e6:.1f} MB) in {dt:.0f}s")

    stats["generations"] += 1
    stats["last_path"] = out_path
    stats["last_time_s"] = round(dt)
    stats["last_res"] = f"{width}x{height}"

    return {
        "status": "ok",
        "path": out_path,
        "size_mb": round(sz / 1e6, 1),
        "time_s": round(dt),
        "resolution": f"{width}x{height}",
        "nfe": DEFAULT_NUM_INFERENCE_STEPS,
    }


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/health", "/status", "/"):
            self._json(200, {
                "status": "ready" if pipe is not None else "loading",
                "busy": busy["value"],
                "mode": "int8+turbo",
                "nfe": DEFAULT_NUM_INFERENCE_STEPS,
                "model_path": MODEL_PATH,
                "lora_path": LORA_PATH,
                "device": DEVICE,
                "generations": stats["generations"],
                "last_path": stats["last_path"],
                "last_time_s": stats["last_time_s"],
                "last_resolution": stats["last_res"],
            })
        else:
            self._json(404, {"error": f"unknown path: {path}"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/generate":
            if pipe is None:
                self._json(503, {"error": "model still loading"})
                return
            if not gen_lock.acquire(blocking=False):
                self._json(429, {"error": "another generation in progress"})
                return
            busy["value"] = True
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                req = json.loads(raw) if raw else {}
                print(f"\n{'='*60}")
                print(f"📩 POST /generate  prompt={req.get('prompt','')[:60]}...")
                result = generate(req)
                code = 200 if "error" not in result else 400
                self._json(code, result)
            except Exception as e:
                traceback.print_exc()
                self._json(500, {"error": str(e)})
            finally:
                busy["value"] = False
                gen_lock.release()
        elif path == "/shutdown":
            print("\n🛑 shutdown requested")
            self._json(200, {"status": "shutting down"})
            threading.Thread(target=lambda: (time.sleep(1), os._exit(0)), daemon=True).start()
        else:
            self._json(404, {"error": f"unknown path: {path}"})

    def log_message(self, fmt, *args):
        pass  # 静默默认日志（我们自己的 print 够了）


def main():
    global pipe
    print(f"{'='*60}")
    print(f"🎬 MiniMax-H3 int8 + Turbo 常驻服务")
    print(f"  🤖 model: {MODEL_PATH}")
    print(f"  🏋️ lora:  {LORA_PATH}")
    print(f"  ⏱️ NFE:   {DEFAULT_NUM_INFERENCE_STEPS} (shifts video={VIDEO_SHIFT} audio={AUDIO_SHIFT})")
    print(f"  📡 port:  {PORT}")
    print(f"  🎮 device: {DEVICE}")
    print(f"  💾 output: {OUTPUT_DIR}")
    print(f"{'='*60}")

    pipe = load_model()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\n🎉 服务就绪: http://localhost:{PORT}")
    print(f"  GET  /health   — 健康检查")
    print(f"  POST /generate — 生成视频（Turbo {DEFAULT_NUM_INFERENCE_STEPS} 步）")
    print(f"  POST /shutdown — 关闭服务")
    print(f"\n示例:")
    print(f'  curl -X POST http://localhost:{PORT}/generate \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"prompt":"a drone shot over mountains","seed":42}}\'')
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Ctrl-C received, shutting down...")
    finally:
        server.shutdown()
        print("🎉 server stopped.")


if __name__ == "__main__":
    main()
