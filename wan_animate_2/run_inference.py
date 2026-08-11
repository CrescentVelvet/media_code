#!/usr/bin/env python3
"""Wan-Animate-2 character animation inference.

Loads the official Wan-Animate-2 pipeline (via core.build_object_from_config_file),
patches the YAML parallel config (sp_size/sharding_size) to match the visible GPU
count so it runs on any node, runs the pipeline on a reference image + driving
video, and saves the result as $OUTPUT_DIR/$OUTPUT_NAME.mp4.

The official YAML defaults to sp_size=8, sharding_size=8 (8×A800). On a different
GPU count we override those two fields in a patched copy of the YAML (co-located
with the original in infer/ so ../ckpts/... relative paths still resolve).

Env vars (set by 02_run_inference.sh):
  OFFICIAL_DIR, YAML_PATH, MODEL_VARIANT,
  PROMPT, PROMPT_REF, REFER_IMAGE, REFER_VIDEO,
  OUTPUT_DIR, OUTPUT_NAME, WIDTH, HEIGHT, FPS, CLIP_LEN, SEED,
  STEP, SAMPLE_GUIDE_SCALE.
"""
import os, re, sys, time, shutil, glob

OFFICIAL_DIR = os.environ["OFFICIAL_DIR"]
YAML_PATH = os.environ["YAML_PATH"]
PROMPT = os.environ["PROMPT"]
PROMPT_REF = os.environ.get("PROMPT_REF", "人物动作的参考视频")
REFER_IMAGE = os.environ["REFER_IMAGE"]
REFER_VIDEO = os.environ["REFER_VIDEO"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
OUTPUT_NAME = os.environ["OUTPUT_NAME"]
WIDTH = int(os.environ["WIDTH"])
HEIGHT = int(os.environ["HEIGHT"])
FPS = int(os.environ["FPS"])
CLIP_LEN = int(os.environ["CLIP_LEN"])
SEED = int(os.environ["SEED"])
STEP = int(os.environ["STEP"])
SAMPLE_GUIDE_SCALE = float(os.environ["SAMPLE_GUIDE_SCALE"])

# --- detect visible GPU count (sp_size = sharding_size = n_gpu) ---
cud = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if cud:
    n_gpu = len([x for x in cud.split(",") if x.strip() != ""])
else:
    try:
        import torch
        n_gpu = torch.cuda.device_count()
    except Exception:
        n_gpu = 1
n_gpu = max(1, n_gpu)
print(f"🎮 detected {n_gpu} visible GPU(s) -> sp_size=sharding_size={n_gpu}")

# --- patch the YAML: override sp_size / sharding_size to n_gpu ---
# Text replacement (not yaml.load) because the YAML uses `torch.bfloat16` literals
# which a plain yaml.load would choke on; the official core loader resolves them.
infer_dir = os.path.join(OFFICIAL_DIR, "infer")
patched_yaml = os.path.join(infer_dir, f"_patched_{os.path.basename(YAML_PATH)}")
with open(YAML_PATH, "r", encoding="utf-8") as f:
    text = f.read()


def _replace(key, val, txt):
    return re.sub(
        rf"^(\s*{key}:\s*)\S+", rf"\g<1>{val}", txt, count=1, flags=re.MULTILINE
    )


text = _replace("sp_size", n_gpu, text)
text = _replace("sharding_size", n_gpu, text)
with open(patched_yaml, "w", encoding="utf-8") as f:
    f.write(text)
print(f"✅ patched YAML -> {patched_yaml}")

# --- import the official pipeline factory (pip install -e . makes `core` importable) ---
sys.path.insert(0, OFFICIAL_DIR)
try:
    from core import build_object_from_config_file
except Exception as e:
    sys.exit(f"❌ cannot import core.build_object_from_config_file: {e}\n"
             f"   run: INSTALL_DEPS=1 bash wan_animate_2/00_setup_env.sh")

# --- build pipeline ---
t0 = time.time()
print("🤖 loading Wan-Animate-2 pipeline ...")
pipeline = build_object_from_config_file(patched_yaml)
print(f"✅ pipeline loaded in {time.time() - t0:.1f}s")

# --- run inference ---
# A private scratch dir for this run; the pipeline writes the output video here.
scratch = os.path.join(OUTPUT_DIR, f".session_{OUTPUT_NAME}")
os.makedirs(scratch, exist_ok=True)
mp4s_before = set(glob.glob(os.path.join(scratch, "**", "*.mp4"), recursive=True))

print("🖼️  refer image: " + REFER_IMAGE)
print("🎬 refer video: " + REFER_VIDEO)
print(f"📐 {WIDTH}x{HEIGHT}  clip_len={CLIP_LEN}  fps={FPS}")
print(f"   step={STEP}  guide_scale={SAMPLE_GUIDE_SCALE}  seed={SEED}")

t1 = time.time()
result = pipeline(
    refer_img_path=REFER_IMAGE,
    tpl_video_path=REFER_VIDEO,
    output_path=scratch,
    width=WIDTH,
    height=HEIGHT,
    fps=FPS,
    seed=SEED,
    clip_len=CLIP_LEN,
    sample_guide_scale=SAMPLE_GUIDE_SCALE,
    step=STEP,
    prompt=PROMPT,
    prompt_ref=PROMPT_REF,
)
print(f"🎬 generation done in {time.time() - t1:.1f}s")

if result is not None:
    print(f"   pipeline returned: {type(result).__name__}")

# --- locate the generated mp4 (the pipeline writes into output_path=scratch) ---
mp4s_after = set(glob.glob(os.path.join(scratch, "**", "*.mp4"), recursive=True))
new_mp4s = sorted(mp4s_after - mp4s_before)
if not new_mp4s:
    # fall back to any mp4 in scratch (covers the case the file pre-existed)
    new_mp4s = sorted(mp4s_after)
if not new_mp4s:
    # fall back to anything the pipeline returned (string path)
    if isinstance(result, str) and os.path.isfile(result):
        new_mp4s = [result]
    elif isinstance(result, (list, tuple)) and result and isinstance(result[0], str):
        new_mp4s = [p for p in result if os.path.isfile(p)]

if not new_mp4s:
    print(f"❌ no output mp4 found under {scratch}", file=sys.stderr)
    print("   the pipeline may write to a different location — inspect:", file=sys.stderr)
    print(f"     ls -R {scratch}", file=sys.stderr)
    sys.exit(1)

# --- move/rename the first found mp4 to the predictable output path ---
src = new_mp4s[0]
dst = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.mp4")
if os.path.abspath(src) == os.path.abspath(dst):
    print(f"✅ saved: {dst}")
else:
    shutil.move(src, dst)
    print(f"✅ saved: {dst}")
# cleanup the scratch session dir
try:
    shutil.rmtree(scratch, ignore_errors=True)
except Exception:
    pass
