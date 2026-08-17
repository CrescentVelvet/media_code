#!/usr/bin/env python3
"""调 MiniMax H3-Context-IR API 把短 prompt（+可选图片）转成 H3-Context-IR 格式长描述。

H3-Context-IR 是 MiniMax 的 hosted preprocessing API（非开源），把自由短 prompt +
可选首帧图转成结构化长描述（integrated_multimodal_description / overall_soundscape
/ non_diegetic_music），效果比短 prompt 好（官方推荐）。

成功时 stdout 只输出长描述（给 .sh 捕获传给 06），调试信息到 stderr。

Env vars:
  MINIMAX_API_KEY:   MiniMax API token（必填，platform.minimax.io / platform.minimaxi.com 申请）
  MINIMAX_API_BASE:  API 基址（默认 https://api.minimaxi.com CN；Global 用 https://api.minimax.io）
  PROMPT:            短 prompt（必填）
  FIRST_FRAME:       首帧图 URL（可选；必须是 http URL，API 不读本地路径）
  DURATION:          时长秒（默认 5）
  RATIO:             比例（默认 adaptive；可选 16:9 / 9:16 / 1:1）
"""
import os, sys, time, json

try:
    import requests
except ImportError:
    sys.exit("❌ 'requests' not installed. pip install requests")

API_KEY = os.environ.get("MINIMAX_API_KEY", "")
API_BASE = os.environ.get("MINIMAX_API_BASE", "https://api.minimaxi.com").rstrip("/")
PROMPT = os.environ.get("PROMPT", "")
FIRST_FRAME = os.environ.get("FIRST_FRAME", "")
DURATION = int(os.environ.get("DURATION", "5"))
RATIO = os.environ.get("RATIO", "adaptive")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def main():
    if not API_KEY:
        sys.exit("❌ MINIMAX_API_KEY not set (apply at platform.minimaxi.com)")
    if not PROMPT:
        sys.exit("❌ PROMPT not set")

    # 构造 content
    content = [{"type": "text", "text": PROMPT}]
    if FIRST_FRAME:
        if not FIRST_FRAME.startswith("http"):
            sys.exit(f"❌ FIRST_FRAME must be http URL (API can't read local path): {FIRST_FRAME}")
        content.append({
            "type": "image_url",
            "image_url": {"url": FIRST_FRAME},
            "role": "first_frame",
        })

    body = {
        "model": "MiniMax-H3",
        "content": content,
        "duration": DURATION,
        "ratio": RATIO,
    }

    url = f"{API_BASE}/v2/h3_context_ir"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    log(f"📤 POST {url}")
    log(f"   prompt: {PROMPT[:80]}{'...' if len(PROMPT)>80 else ''}")
    if FIRST_FRAME:
        log(f"   first_frame: {FIRST_FRAME}")

    r = requests.post(url, json=body, headers=headers, timeout=120)
    if r.status_code >= 400:
        sys.exit(f"❌ HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()

    # 拿 prompt（同步返回 or 轮询）
    task = data.get("task", data)
    task_id = task.get("id", "")
    status = task.get("status", "")
    content_obj = task.get("content", {})
    prompt_text = content_obj.get("prompt", "") if isinstance(content_obj, dict) else ""

    # 没拿到 prompt 且状态非 succeeded → 轮询
    if not prompt_text and task_id and status not in ("succeeded", "completed"):
        log(f"⏳ polling task {task_id} (status={status})...")
        poll_url = f"{url}/{task_id}" if task_id else ""
        if not poll_url:
            sys.exit(f"❌ no task id to poll, response: {json.dumps(data)[:500]}")
        for i in range(120):
            time.sleep(5)
            try:
                r2 = requests.get(poll_url, headers=headers, timeout=30)
                if r2.status_code >= 400:
                    log(f"   [{i*5}s] poll HTTP {r2.status_code}")
                    continue
                d2 = r2.json()
                t2 = d2.get("task", d2)
                st = t2.get("status", "")
                log(f"   [{i*5:4d}s] status={st}")
                if st in ("succeeded", "completed"):
                    c2 = t2.get("content", {})
                    prompt_text = c2.get("prompt", "") if isinstance(c2, dict) else ""
                    break
                if st in ("failed", "error", "canceled"):
                    sys.exit(f"❌ task {st}: {json.dumps(d2)[:500]}")
            except Exception as e:
                log(f"   [{i*5}s] poll error: {e}")

    if not prompt_text:
        sys.exit(f"❌ no prompt in response: {json.dumps(data)[:500]}")

    log(f"✅ got H3-Context-IR prompt ({len(prompt_text)} chars)")
    # stdout 只输出长描述（给 .sh 捕获）
    sys.stdout.write(prompt_text)


if __name__ == "__main__":
    main()
