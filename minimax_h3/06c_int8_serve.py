#!/usr/bin/env python3
"""06c — MiniMax-H3 int8 常驻 HTTP 服务（从 D 盘加载，多次请求不重新加载）。

与 06a 相同的 int8 + block offload 加载逻辑，但加载后启动 HTTP 服务，
保持模型在 RAM 中，多次 POST /generate 请求不重新加载。
适合从 D 盘（HDD）加载的场景——20-40min 加载成本只付一次。

API:
  GET  /health     → {"status":"ready","busy":false}
  POST /generate   → JSON body → 生成视频 → {"path":...,"size_mb":...,"time_s":...}
  POST /shutdown   → 优雅关闭

generate JSON body（所有字段可选，除 prompt）：
  {
    "prompt":      "integrated_multimodal_description: ...",  // 必填
    "first_frame": "/mnt/d/img.jpg",   // 可选，FL2VA 首帧
    "last_frame":  "/mnt/d/last.jpg",  // 可选，末帧
    "num_frames":  124,               // 默认 env NUM_FRAMES
    "seed":        42,                // 默认 0
    "max_pixels":  133120,            // 默认 env MAX_PIXELS
    "fps":         24,                // 默认 env FPS
    "width":       0,                 // 0=auto
    "height":      0,                 // 0=auto
    "output_name": "my.mp4"           // 默认自动生成
  }

Env vars:
  MODEL_PATH: 模型路径（默认 /mnt/d/wheel/minimaxh3_ms）
  PORT: HTTP 端口（默认 8000）
  DEVICE: GPU（默认 cuda:0）
  OUTPUT_DIR: 视频输出目录
  MAX_PIXELS, FPS, NUM_FRAMES: 默认生成参数
"""
import os, sys, time, json, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# ── 1. 解析环境变量 ──────────────────────────────────────────────────────
MODEL_PATH = os.path.abspath(os.environ.get("MODEL_PATH", "/mnt/d/wheel/minimaxh3_ms"))
PORT = int(os.environ.get("PORT", "8000"))
DEVICE = os.environ.get("DEVICE", "cuda:0")
OUTPUT_DIR = os.path.abspath(os.environ.get(
    "OUTPUT_DIR", os.path.expanduser("~/output/minimaxh3_rotate_results/results_int8")))
DEFAULT_MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(512 * 768)))
DEFAULT_FPS = int(os.environ.get("FPS", "24"))
DEFAULT_NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "124"))

# ── 2. 全局状态 ──────────────────────────────────────────────────────────
pipe = None
gen_lock = threading.Lock()
busy = {"value": False}
stats = {"generations": 0, "last_path": "", "last_time_s": 0, "last_res": ""}


def load_model():
    """加载 int8 量化 pipeline + block offload（同 06a，~20-40min 从 D 盘）。"""
    from diffusers import ModularPipeline, MiniMaxH3Transformer3DModel, TorchAoConfig
    from diffusers.hooks import apply_group_offloading
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig
    import torch

    if not os.path.isdir(MODEL_PATH):
        sys.exit(f"❌ MODEL_PATH not found: {MODEL_PATH}")
    if not os.path.isfile(os.path.join(MODEL_PATH, "model_index.json")):
        sys.exit(f"❌ model_index.json not found in {MODEL_PATH} — incomplete download?")

    print(f"🚀 [06c] Loading MiniMax-H3 int8 from {MODEL_PATH}")
    print(f"  🎮 device: {DEVICE}")
    print(f"  💾 output: {OUTPUT_DIR}")
    print(f"  ⏳ loading from D drive (HDD) — this takes 20-40 min...")

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

    offload = dict(onload_device=torch.device(DEVICE), offload_device=torch.device("cpu"))
    p.transformer.enable_group_offload(
        offload_type="block_level", num_blocks_per_group=1,
        low_cpu_mem_usage=True, **offload,
    )
    apply_group_offloading(
        p.text_encoder.model, offload_type="leaf_level",
        low_cpu_mem_usage=True, **offload,
    )
    p.vae.to(DEVICE)
    p.audio_vae.to(DEVICE)

    dt = time.time() - t0
    print(f"  ✅ pipeline loaded in {dt/60:.1f} min (block offload on {DEVICE})")
    return p


def auto_resolution(first_frame, max_pixels, width, height):
    """从首帧图算 auto 分辨率（同 06a）。"""
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
    """执行一次生成（调用时已持锁）。返回 dict 结果。"""
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
        output_name = f"{task}_int8_seed{seed}.mp4"

    width, height = auto_resolution(first_frame, max_pixels, width, height)
    print(f"  🎯 task={task}  frames={num_frames}  res={width}x{height}  seed={seed}")

    generator = torch.Generator(DEVICE).manual_seed(seed)
    outputs = ["videos", "audio", "sampling_rate"]
    kwargs = dict(prompt=prompt, num_frames=num_frames, generator=generator, output=outputs)
    if width > 0:
        kwargs["width"] = width
    if height > 0:
        kwargs["height"] = height
    if first_frame:
        kwargs["image"] = load_image(first_frame)
    if last_frame:
        kwargs["last_image"] = load_image(last_frame)

    t0 = time.time()
    print("  🎬 generating...")
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
                "model_path": MODEL_PATH,
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
    print(f"🎬 MiniMax-H3 int8 常驻服务")
    print(f"  🤖 model: {MODEL_PATH}")
    print(f"  📡 port:  {PORT}")
    print(f"  🎮 device: {DEVICE}")
    print(f"  💾 output: {OUTPUT_DIR}")
    print(f"{'='*60}")

    pipe = load_model()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\n🎉 服务就绪: http://localhost:{PORT}")
    print(f"  GET  /health   — 健康检查")
    print(f"  POST /generate — 生成视频")
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
