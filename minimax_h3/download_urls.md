# MiniMax-H3 离线下载清单
# ====================================================================
# 网络诊断结论（本机 WSL）：
#   - huggingface.co 直连: 不通 (HTTP 000)
#   - hf-mirror.com 镜像: 可达 (307 重定向到 us.aws.cdn.hf.co)
#   - CDN 单连接: ~2MB/s；aria2c 16 连接: ~2-3MB/s（CDN 限速）
#   - IDM/迅雷 多线程可能更快（Windows 网络栈，绕过 WSL NAT）
# 权重 ~120GB（bf16 原版，06a 加载时 int8 量化到 ~62GB CPU RAM）
#
# 下载工具建议：IDM / 迅雷（多线程）。批量导入本文件中未注释的 URL。
# 保存到：/mnt/d/wheel/minimaxh3_ms/ 下，按下面的相对路径建子目录
#   例如 transformer/diffusion_pytorch_model-00001-of-00014.safetensors
#        存到 /mnt/d/wheel/minimaxh3_ms/transformer/diffusion_pytorch_model-00001-of-00014.safetensors
#
# 格式说明：
#   未注释的行 = 待下载的 URL（IDM/迅雷批量导入这些）
#   # [done] = 已下载完，URL 注释掉了
#
# ⚠️ 06a/06c 用顶层 diffusers 格式权重（diffusion_pytorch_model-*.safetensors）
#   不是 FL2VA/ 子目录里的原始格式（model-*.safetensors）
# ====================================================================

# ====================================================================
## transformer/ (16 files, 66.3GB — ALL DOWNLOADED)
# ====================================================================
# [done] 1KB    transformer/config.json
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/config.json -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00001-of-00014.safetensors -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00002-of-00014.safetensors -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00003-of-00014.safetensors -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00004-of-00014.safetensors -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00005-of-00014.safetensors -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00006-of-00014.safetensors -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00007-of-00014.safetensors -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00008-of-00014.safetensors -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00009-of-00014.safetensors -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00010-of-00014.safetensors -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00011-of-00014.safetensors -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00012-of-00014.safetensors -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00013-of-00014.safetensors -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00014-of-00014.safetensors -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model.safetensors.index.json -->

# ====================================================================
## text_encoder/ (23 files, 66.7GB — 6 done, 17 to download)
# ====================================================================
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/chat_template.json -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/config.json -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/merges.txt -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00001-of-00014.safetensors -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00002-of-00014.safetensors -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00003-of-00014.safetensors -->
# --- 以下 11 个 shards 待下载（各 ~4.88GB，最后一个 3.27GB）---
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00004-of-00014.safetensors
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00005-of-00014.safetensors
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00006-of-00014.safetensors
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00007-of-00014.safetensors
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00008-of-00014.safetensors
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00009-of-00014.safetensors
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00010-of-00014.safetensors
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00011-of-00014.safetensors
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00012-of-00014.safetensors
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00013-of-00014.safetensors
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00014-of-00014.safetensors
# --- 以下小文件待下载 ---
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model.safetensors.index.json
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/preprocessor_config.json
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/tokenizer.json
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/tokenizer_config.json
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/video_preprocessor_config.json
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/vocab.json

# ====================================================================
## vae/ (5 files, 10.4GB — 2 done, 3 to download)
# ====================================================================
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/vae/config.json -->
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/vae/diffusion_pytorch_model.safetensors.index.json -->
# --- 以下 3 个 shards 待下载 ---
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/vae/diffusion_pytorch_model-00001-of-00003.safetensors
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/vae/diffusion_pytorch_model-00002-of-00003.safetensors
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/vae/diffusion_pytorch_model-00003-of-00003.safetensors

# ====================================================================
## audio_vae/ (2 files, 0.6GB — 2 to download)
# ====================================================================
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/audio_vae/config.json
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/audio_vae/diffusion_pytorch_model.safetensors

# ====================================================================
## processor/ (7 files, 0.0GB — 7 to download)
# ====================================================================
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/processor/chat_template.json
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/processor/merges.txt
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/processor/preprocessor_config.json
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/processor/tokenizer.json
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/processor/tokenizer_config.json
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/processor/video_preprocessor_config.json
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/processor/vocab.json

# ====================================================================
## tokenizer/ (4 files, 0.0GB — 4 to download)
# ====================================================================
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/tokenizer/merges.txt
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/tokenizer/tokenizer.json
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/tokenizer/tokenizer_config.json
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/tokenizer/vocab.json

# ====================================================================
## scheduler/ (1 file, 0.0GB — 1 to download)
# ====================================================================
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/scheduler/scheduler_config.json

# ====================================================================
## audio_scheduler/ (1 file, 0.0GB — 1 to download)
# ====================================================================
https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/audio_scheduler/scheduler_config.json

# ====================================================================
## (root) (1 file — ALL DOWNLOADED)
# ====================================================================
# <!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/model_index.json -->

# ====================================================================
## 汇总: 60 files, 144.1GB total
## 已下载: 25 files, 81.0GB
## 待下载: 35 files, 63.1GB（以上未注释的 URL）
# ====================================================================

# 下载完成后，把文件按相对路径放到 /mnt/d/wheel/minimaxh3_ms/
# 然后运行 06c 常驻服务：
#   GPU=0 MODEL_PATH=/mnt/d/wheel/minimaxh3_ms \
#     DEVICE=cuda:0 MAX_PIXELS=133120 OUTPUT_DIR=~/output/minimaxh3_rotate_results/results_int8 \
#     bash /mnt/c/code/media_code/minimax_h3/06c_int8_serve.sh
#
# 或 06a 一次性推理：
#   conda activate minimax_h3
#   GPU=0 MODEL_PATH=/mnt/d/wheel/minimaxh3_ms \
#     DEVICE=cuda:0 MAX_PIXELS=133120 FPS=24 NUM_FRAMES=124 SEED=42 \
#     TASK=fl2va FIRST_FRAME=/mnt/d/your_image.jpg PROMPT="..." \
#     OUTPUT_DIR=~/output/minimaxh3_rotate_results/results_int8 \
#     bash /mnt/c/code/media_code/minimax_h3/06a_diffusers_inference.sh
