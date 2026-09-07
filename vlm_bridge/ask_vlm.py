#!/usr/bin/env python3
"""Bridge image understanding to a vision LLM API (Qwen3.8-Max by default).

The coding model (GLM-5.3) has no vision. When it needs to *see* an image
(a rendered frame, a screenshot, a comparison), it runs this script via
Bash; the image is base64-embedded into an OpenAI-compatible request and
the answer comes back as plain text on stdout.

Reads config from vlm.env next to this script (see vlm.env.example).
Only uses the Python standard library — no pip install needed.

Usage:
  python ask_vlm.py IMAGE_PATH "question in natural language"
  python ask_vlm.py img1.jpg img2.png "compare these two images"
  python ask_vlm.py --raw IMAGE_PATH "question"     # do not append guidance
  echo "question" | python ask_vlm.py IMAGE_PATH    # question via stdin

Env vars (from vlm.env or environment):
  VLM_API_URL, VLM_API_KEY, VLM_MODEL, VLM_FALLBACK_MODEL,
  VLM_MAX_TOKENS, VLM_HTTP_PROXY
"""
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
MIME_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".bmp": "image/bmp", ".webp": "image/webp", ".gif": "image/gif",
}

# Appended when --raw is NOT given: makes the VLM answer like a careful
# human observer, giving concrete, verifiable facts the caller can act on.
GUIDANCE = (
    "Answer in the same language as the question. Be concrete and factual: "
    "report what is actually visible (objects, layout, colors, text, "
    "artifacts, anomalies), not generic descriptions. If something is "
    "unclear or ambiguous, say so explicitly instead of guessing."
)


def load_env():
    """Read vlm.env (KEY=VALUE lines, `export ` prefix optional), env wins."""
    env_path = SCRIPT_DIR / "vlm.env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)  # real env vars take precedence

    url = os.environ.get("VLM_API_URL")
    key = os.environ.get("VLM_API_KEY")
    model = os.environ.get("VLM_MODEL", "qwen3.8-max")
    if not url or not key:
        sys.exit(
            "ERROR: VLM_API_URL / VLM_API_KEY not set. "
            "Copy vlm_bridge/vlm.env.example to vlm_bridge/vlm.env and fill them "
            "(values available in ~/.workbuddy/models.json)."
        )
    return url, key, model


def encode_image(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext not in IMG_EXTS:
        sys.exit(f"ERROR: unsupported image type '{ext}': {path}")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{MIME_MAP[ext]};base64,{b64}", MIME_MAP[ext]


def call_vlm(url, key, model, images, question, max_tokens):
    content = [{"type": "image_url", "image_url": {"url": u}} for u in images]
    content.append({"type": "text", "text": question})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.2,  # vision QA: factual, low-hallucination
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    proxy = os.environ.get("VLM_HTTP_PROXY", "").strip()
    if proxy:
        handler = urllib.request.ProxyHandler(
            {"http": proxy, "https": proxy}
        )
        opener = urllib.request.build_opener(handler)
    else:
        opener = urllib.request.build_opener()
    try:
        with opener.open(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
        sys.exit(f"ERROR: API HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        sys.exit(f"ERROR: cannot reach VLM API ({e.reason}). "
                 "Check network / VLM_HTTP_PROXY in vlm.env.")

    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        sys.exit(f"ERROR: unexpected API response: {json.dumps(body)[:500]}")


def main():
    args = [a for a in sys.argv[1:] if a != "--raw"]
    raw_mode = "--raw" in sys.argv[1:]
    if not args:
        sys.exit(__doc__)

    # Split args into image paths (existing files / valid extensions) and
    # the question (everything else, joined). Question may also come from stdin.
    images, others = [], []
    for a in args:
        p = Path(a)
        if p.suffix.lower() in IMG_EXTS and (p.is_file() or not p.exists()):
            images.append(p)
        else:
            others.append(a)
    question = " ".join(others)
    if not question and not sys.stdin.isatty():
        question = sys.stdin.read().strip()
    if not images:
        sys.exit("ERROR: no image path given.")
    if not question:
        sys.exit("ERROR: no question given.")
    for p in images:
        if not p.is_file():
            sys.exit(f"ERROR: image not found: {p}")
    if len(images) > 4:
        sys.exit("ERROR: at most 4 images per call.")

    url, key, model = load_env()
    max_tokens = int(os.environ.get("VLM_MAX_TOKENS", "1024"))

    if not raw_mode:
        question = f"{question}\n\n{GUIDANCE}"
    encoded = [encode_image(p)[0] for p in images]

    print(f"🤖 VLM: {model} | 🖼️ {len(images)} image(s) | Q: {question[:60]}…",
          file=sys.stderr)
    try:
        answer = call_vlm(url, key, model, encoded, question, max_tokens)
    except SystemExit as e:
        # Primary model failed -> one retry with the fallback model.
        fb = os.environ.get("VLM_FALLBACK_MODEL", "").strip()
        msg = str(e)
        if fb and fb != model and ("ERROR: API" in msg):
            print(f"⚠️ primary model failed, retrying with {fb} …",
                  file=sys.stderr)
            answer = call_vlm(url, key, fb, encoded, question, max_tokens)
        else:
            raise

    print(answer)


if __name__ == "__main__":
    main()
