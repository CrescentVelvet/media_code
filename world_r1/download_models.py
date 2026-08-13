#!/usr/bin/env python3
"""下载 World-R1 所需的全部模型权重。

Env vars:
  MODEL_FAMILY   — wan_large (默认) | wan_small | cogvideox
  WAN_MODEL_PATH — Wan2.1 基础视频模型本地目录 (默认 $MODEL_DIR/...)
  HF_HOME        — HF cache 根 (DA3 / Qwen3-VL 下到这里)
  HF_TOKEN       — HuggingFace token (gated 模型可能需要)
"""
import os
import sys
import time


def _download_to_dir(repo_id: str, local_dir: str, token: str | None = None,
                     repo_type: str = "model"):
    """用 snapshot_download 下到指定目录 (local_dir 模式, 不走 cache)。"""
    from huggingface_hub import snapshot_download
    os.makedirs(local_dir, exist_ok=True)
    print(f"📦 downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token or None,
        repo_type=repo_type,
    )
    print(f"✅ saved: {local_dir}")


def _download_to_cache(repo_id: str, cache_dir: str, token: str | None = None,
                      repo_type: str = "model"):
    """用 snapshot_download 下到 HF cache (from_pretrained("repo_id") 会找这里)。"""
    from huggingface_hub import snapshot_download
    print(f"📦 downloading {repo_id} -> HF cache ({cache_dir})")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        token=token or None,
        repo_type=repo_type,
    )
    print(f"✅ cached: {repo_id}")


def main():
    model_family = os.environ.get("MODEL_FAMILY", "wan_large")
    model_dir = os.environ.get("MODEL_DIR", "../../model")
    wan_model_path = os.environ.get("WAN_MODEL_PATH", "")
    hf_home = os.environ.get("HF_HOME", os.path.join(model_dir, "huggingface"))
    hf_token = os.environ.get("HF_TOKEN", "")
    hub_cache = os.path.join(hf_home, "hub")

    os.makedirs(hub_cache, exist_ok=True)

    # 1. 基础视频模型
    base_models = {
        "wan_large": ("Wan-AI/Wan2.1-T2V-14B-Diffusers", "Wan2.1-T2V-14B-Diffusers"),
        "wan_small": ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "Wan2.1-T2V-1.3B-Diffusers"),
        "cogvideox": ("THUDM/CogVideoX1.5-5B", "CogVideoX1.5-5B"),
    }
    if model_family not in base_models:
        sys.exit(f"❌ MODEL_FAMILY={model_family} 未知; 可选: {list(base_models)}")

    repo_id, dir_name = base_models[model_family]
    if not wan_model_path:
        wan_model_path = os.path.join(model_dir, dir_name)

    if os.path.exists(os.path.join(wan_model_path, "model_index.json")) or \
       os.path.exists(os.path.join(wan_model_path, "config.json")):
        print(f"✅ base model already present: {wan_model_path}")
    else:
        _download_to_dir(repo_id, wan_model_path, hf_token)

    # 2. DA3-GIANT (3D reward 重建模型; 代码用 from_pretrained("depth-anything/DA3-GIANT"))
    da3_repo = "depth-anything/DA3-GIANT"
    da3_cache = os.path.join(hub_cache, "models--depth-anything--DA3-GIANT")
    if os.path.exists(da3_cache):
        print(f"✅ DA3-GIANT already cached: {da3_cache}")
    else:
        _download_to_cache(da3_repo, hub_cache, hf_token)

    # 3. Qwen3-VL-4B-Instruct (3D reward scorer + general reward; 代码用 from_pretrained)
    qwen_repo = "Qwen/Qwen3-VL-4B-Instruct"
    qwen_cache = os.path.join(hub_cache, "models--Qwen--Qwen3-VL-4B-Instruct")
    if os.path.exists(qwen_cache):
        print(f"✅ Qwen3-VL-4B-Instruct already cached: {qwen_cache}")
    else:
        _download_to_cache(qwen_repo, hub_cache, hf_token)

    print("")
    print("🎉 All models downloaded.")
    print(f"  🏋️ base model:  {wan_model_path}")
    print(f"  📐 DA3-GIANT:   {da3_cache}")
    print(f"  🤖 Qwen3-VL:    {qwen_cache}")
    print("")
    print("⚠️ LPIPS (alex net) + HPSv2 权重在 reward server 首次启动时自动下载。")
    print("   若代理拦了自动下载, 见 README.md 排错。")


if __name__ == "__main__":
    main()
