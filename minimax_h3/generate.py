#!/usr/bin/env python3
"""MiniMax-H3 video (+ audio) generation client for the SGLang Diffusion server.

Talks to the server started by 02_serve.sh (the H3-Base 768p deployment):
    POST   /v1/videos          -> submit a generation job, get a video id
    GET    /v1/videos/<id>      -> poll status (queued / in_progress / completed / failed)
    GET    /v1/videos/<id>/content -> download the generated mp4

All knobs are env vars (forwarded by 03_generate.sh). The request body mirrors
the official reproducible-768p scripts (task/prompt/conditions/target/seed) and
the SGLang cookbook (adds num_inference_steps/flow_shift/audio_flow_shift).

Task -> conditions mapping (matches the cookbook recipes):
  t2va   : no conditions (text-only -> video+audio)
  fl2va  : FIRST_FRAME (role=keyframe, frame_index=0) and/or LAST_FRAME (frame_index=-1)
  ref2va : REF_IMAGES / REF_AUDIOS / REF_VIDEOS  (role=reference; videos take
           start_time_seconds). Mixed order is images, then audios, then videos.
"""
import os
import sys
import time
import json
import urllib.parse

try:
    import requests
except ImportError:
    sys.exit("ERROR: 'requests' not installed. pip install requests")


def env(k, default=""):
    v = os.environ.get(k)
    return v if (v is not None and v != "") else default


def to_uri(p):
    """Local path -> file:// URI; http(s) URL -> as-is."""
    if not p:
        return ""
    if p.startswith("http://") or p.startswith("https://") or p.startswith("file://"):
        return p
    return "file://" + os.path.abspath(p)


def split_list(s):
    """Comma-separated string -> list (preserves order, drops empties)."""
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def build_conditions():
    """Build the `conditions` array from env vars according to the task.

    For exact reproduction of an official case whose condition ORDER matters
    (e.g. Ref2VA references <Video 1>=video, <Audio 1>=video soundtrack,
    <Audio 2>=audio file), set CONDITIONS_FILE=<path.json> to pass a literal
    JSON array verbatim — it overrides the env-var construction below.
    """
    cf = env("CONDITIONS_FILE")
    if cf:
        with open(cf, encoding="utf-8") as f:
            arr = json.load(f)
        print(f"[*] using conditions from {cf} ({len(arr)} item(s))")
        return arr

    task = env("TASK", "t2va")
    conds = []

    if task == "fl2va":
        first = env("FIRST_FRAME")
        last = env("LAST_FRAME")
        if first:
            conds.append({"type": "image", "uri": to_uri(first),
                          "role": "keyframe", "frame_index": 0})
        if last:
            conds.append({"type": "image", "uri": to_uri(last),
                          "role": "keyframe", "frame_index": -1})
        if not conds:
            # No endpoint frames -> pure T2VA-style on the FL2VA weights.
            pass
        return conds

    if task == "ref2va":
        # Order matches the cookbook mixed_ref recipe: images, audios, videos.
        for i, img in enumerate(split_list(env("REF_IMAGES")), 1):
            conds.append({"type": "image", "uri": to_uri(img), "role": "reference"})
        for i, aud in enumerate(split_list(env("REF_AUDIOS")), 1):
            conds.append({"type": "audio", "uri": to_uri(aud), "role": "reference"})
        vids = split_list(env("REF_VIDEOS"))
        starts = split_list(env("REF_VIDEO_STARTS"))
        video_audio = env("VIDEO_AS_AUDIO_REF", "0") == "1"  # type=video_audio carries the soundtrack
        for i, vid in enumerate(vids):
            c = {"type": "video_audio" if video_audio else "video",
                 "uri": to_uri(vid), "role": "reference"}
            if i < len(starts):
                try:
                    c["start_time_seconds"] = float(starts[i])
                except ValueError:
                    pass
            conds.append(c)
        return conds

    # t2va or unknown -> no conditions
    return conds


def build_request():
    task = env("TASK", "t2va")

    # Prompt: inline PROMPT wins, else PROMPT_FILE, else a built-in default.
    prompt = env("PROMPT")
    if not prompt:
        pf = env("PROMPT_FILE")
        if pf:
            with open(pf, encoding="utf-8") as f:
                prompt = f.read().strip()
    if not prompt:
        prompt = ("A cinematic shot of a red fox walking through a snowy forest "
                  "at dawn, soft morning light, detailed fur, gentle camera dolly, "
                  "with ambient wind and soft footsteps.")

    duration = int(env("DURATION", "5"))
    # FL2VA / Ref2VA default to adaptive aspect (auto); T2VA defaults to 16:9.
    if env("ASPECT_RATIO"):
        ar = env("ASPECT_RATIO")
    elif task in ("fl2va", "ref2va"):
        ar = "auto"
    else:
        ar = "16:9"

    body = {
        "task": task,
        "prompt": prompt,
        "conditions": build_conditions(),
        "target": {
            "short_edge": int(env("SHORT_EDGE", "768")),
            "aspect_ratio": ar,
            "duration_seconds": duration,
        },
        "seed": int(env("SEED", "0")),
        # Sampling params (SGLang cookbook defaults; match the reproducible
        # server defaults). Override via env if you want fewer/more steps.
        "num_inference_steps": int(env("NUM_INFERENCE_STEPS", "50")),
        "flow_shift": float(env("FLOW_SHIFT", "12.0")),
        "audio_flow_shift": float(env("AUDIO_FLOW_SHIFT", "3.0")),
    }
    no = env("NUM_OUTPUTS")
    if no:
        body["num_outputs_per_prompt"] = int(no)
    return body


def submit(server_url, body, timeout=120):
    url = server_url.rstrip("/") + "/v1/videos"
    print(f"[*] POST {url}")
    print(f"    task={body['task']}  duration={body['target']['duration_seconds']}s  "
          f"aspect={body['target']['aspect_ratio']}  seed={body['seed']}  "
          f"steps={body.get('num_inference_steps')}  conditions={len(body['conditions'])}")
    r = requests.post(url, json=body, timeout=timeout)
    if r.status_code >= 400:
        sys.exit(f"ERROR: submit failed HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    vid = data.get("id")
    if not vid:
        sys.exit(f"ERROR: no 'id' in response: {json.dumps(data)[:500]}")
    print(f"    -> video_id = {vid}")
    return vid


TERMINAL = {"completed", "succeeded", "failed", "error", "canceled"}


def poll(server_url, vid):
    url = server_url.rstrip("/") + f"/v1/videos/{vid}"
    interval = int(env("POLL_INTERVAL", "10"))
    timeout_s = int(env("TIMEOUT_MINS", "30")) * 60
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code < 400:
                d = r.json()
                st = d.get("status") or d.get("state") or ""
                if st != last:
                    extra = ""
                    if st in ("in_progress", "processing"):
                        prog = d.get("progress") or d.get("progress_ratio")
                        if prog is not None:
                            extra = f"  progress={prog}"
                    print(f"    [{int(time.time()-t0):4d}s] status={st or '?'}{extra}")
                    last = st
                if st in TERMINAL:
                    return st, d
            else:
                print(f"    [poll] HTTP {r.status_code}: {r.text[:120]}")
        except Exception as e:
            print(f"    [poll] transient error: {e}")
        time.sleep(interval)
    sys.exit(f"ERROR: timed out after {timeout_s}s waiting for video {vid}")


def download(server_url, vid, out_path):
    url = server_url.rstrip("/") + f"/v1/videos/{vid}/content"
    print(f"[*] GET {url}")
    with requests.get(url, stream=True, timeout=300) as r:
        if r.status_code >= 400:
            sys.exit(f"ERROR: download failed HTTP {r.status_code}: {r.text[:300]}")
        total = int(r.headers.get("Content-Length", 0))
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        done = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r    downloaded {done/1e6:.1f}/{total/1e6:.1f} MB", end="")
        print()
    sz = os.path.getsize(out_path)
    print(f"    saved: {out_path}  ({sz/1e6:.1f} MB)")
    return out_path


def main():
    server_url = env("SERVER_URL", "http://localhost:30010")
    # Allow bare "host:port" too.
    if "://" not in server_url:
        server_url = "http://" + server_url

    out_dir = env("OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             "..", "MiniMax-H3", "results", env("TASK", "t2va")))
    out_dir = os.path.abspath(out_dir)
    name = env("OUTPUT_NAME") or f"{env('TASK','t2va')}_seed{env('SEED','0')}.mp4"
    out_path = os.path.join(out_dir, name)

    t0 = time.time()
    body = build_request()
    vid = submit(server_url, body)
    st, info = poll(server_url, vid)
    if st not in ("completed", "succeeded"):
        sys.exit(f"ERROR: generation ended with status={st}: {json.dumps(info)[:400]}")
    download(server_url, vid, out_path)
    print(f"[*] done in {time.time()-t0:.0f}s  ->  {out_path}")


if __name__ == "__main__":
    main()
