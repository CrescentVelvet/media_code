# MiniMax-H3 离线下载清单
# ====================================================================
# 网络诊断结论（本机 WSL）：
#   - huggingface.co 直连: 不通 (HTTP 000)
#   - hf-mirror.com 镜像: 可达 (307 重定向到 us.aws.cdn.hf.co)
#   - CDN 单连接: ~2MB/s；aria2c 16 连接: ~2-3MB/s（CDN 限速）
#   - IDM/迅雷 多线程可能更快（Windows 网络栈，绕过 WSL NAT）
# 权重 ~120GB（bf16 原版，06a 加载时 int8 量化到 ~62GB CPU RAM）
#
# 下载工具建议：IDM / 迅雷（多线程）。批量导入本文件的 URL。
# 保存到：~/model/MiniMax-H3/ 下，按下面的相对路径建子目录
#   例如 transformer/diffusion_pytorch_model-00001-of-00014.safetensors
#        存到 ~/model/MiniMax-H3/transformer/diffusion_pytorch_model-00001-of-00014.safetensors
#
# ⚠️ 06a 用顶层 diffusers 格式权重（diffusion_pytorch_model-*.safetensors）
#   不是 FL2VA/ 子目录里的原始格式（model-*.safetensors）
# ====================================================================

# ====================================================================
## transformer/ (16 files, 66.3GB, ALL DOWNLOADED)
# ====================================================================
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/config.json -->
#   [done] 1KB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00001-of-00014.safetensors -->
#   [done] 4.83GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00002-of-00014.safetensors -->
#   [done] 4.70GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00003-of-00014.safetensors -->
#   [done] 4.93GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00004-of-00014.safetensors -->
#   [done] 4.57GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00005-of-00014.safetensors -->
#   [done] 4.70GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00006-of-00014.safetensors -->
#   [done] 4.93GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00007-of-00014.safetensors -->
#   [done] 4.57GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00008-of-00014.safetensors -->
#   [done] 4.70GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00009-of-00014.safetensors -->
#   [done] 4.93GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00010-of-00014.safetensors -->
#   [done] 4.57GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00011-of-00014.safetensors -->
#   [done] 4.70GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00012-of-00014.safetensors -->
#   [done] 4.93GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00013-of-00014.safetensors -->
#   [done] 4.57GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model-00014-of-00014.safetensors -->
#   [done] 4.64GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/transformer/diffusion_pytorch_model.safetensors.index.json -->
#   [done] 64KB

# ====================================================================
## text_encoder/ (23 files, 66.7GB, 17 to download)
# ====================================================================
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/chat_template.json -->
#   [done] 5KB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/config.json -->
#   [done] 1KB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/merges.txt -->
#   [done] 2MB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00001-of-00014.safetensors -->
#   [done] 4.93GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00002-of-00014.safetensors -->
#   [done] 4.88GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00003-of-00014.safetensors -->
#   [done] 4.88GB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00004-of-00014.safetensors -->
#   ↑ 4.88GB，大文件建议 IDM/迅雷多线程
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00005-of-00014.safetensors -->
#   ↑ 4.88GB，大文件建议 IDM/迅雷多线程
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00006-of-00014.safetensors -->
#   ↑ 4.88GB，大文件建议 IDM/迅雷多线程
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00007-of-00014.safetensors -->
#   ↑ 4.88GB，大文件建议 IDM/迅雷多线程
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00008-of-00014.safetensors -->
#   ↑ 4.88GB，大文件建议 IDM/迅雷多线程
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00009-of-00014.safetensors -->
#   ↑ 4.88GB，大文件建议 IDM/迅雷多线程
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00010-of-00014.safetensors -->
#   ↑ 4.88GB，大文件建议 IDM/迅雷多线程
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00011-of-00014.safetensors -->
#   ↑ 4.88GB，大文件建议 IDM/迅雷多线程
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00012-of-00014.safetensors -->
#   ↑ 4.88GB，大文件建议 IDM/迅雷多线程
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00013-of-00014.safetensors -->
#   ↑ 4.88GB，大文件建议 IDM/迅雷多线程
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model-00014-of-00014.safetensors -->
#   ↑ 3.27GB，大文件建议 IDM/迅雷多线程
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/model.safetensors.index.json -->
#   ↑ 98KB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/preprocessor_config.json -->
#   ↑ 0KB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/tokenizer.json -->
#   ↑ 7MB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/tokenizer_config.json -->
#   ↑ 11KB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/video_preprocessor_config.json -->
#   ↑ 0KB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/text_encoder/vocab.json -->
#   ↑ 3MB

# ====================================================================
## vae/ (5 files, 10.4GB, 3 to download)
# ====================================================================
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/vae/config.json -->
#   [done] 2KB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/vae/diffusion_pytorch_model-00001-of-00003.safetensors -->
#   ↑ 5.06GB，大文件建议 IDM/迅雷多线程
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/vae/diffusion_pytorch_model-00002-of-00003.safetensors -->
#   ↑ 4.96GB，大文件建议 IDM/迅雷多线程
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/vae/diffusion_pytorch_model-00003-of-00003.safetensors -->
#   ↑ 399MB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/vae/diffusion_pytorch_model.safetensors.index.json -->
#   [done] 74KB

# ====================================================================
## audio_vae/ (2 files, 0.6GB, 2 to download)
# ====================================================================
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/audio_vae/config.json -->
#   ↑ 2KB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/audio_vae/diffusion_pytorch_model.safetensors -->
#   ↑ 605MB

# ====================================================================
## processor/ (7 files, 0.0GB, 7 to download)
# ====================================================================
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/processor/chat_template.json -->
#   ↑ 5KB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/processor/merges.txt -->
#   ↑ 2MB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/processor/preprocessor_config.json -->
#   ↑ 0KB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/processor/tokenizer.json -->
#   ↑ 7MB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/processor/tokenizer_config.json -->
#   ↑ 11KB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/processor/video_preprocessor_config.json -->
#   ↑ 0KB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/processor/vocab.json -->
#   ↑ 3MB

# ====================================================================
## tokenizer/ (4 files, 0.0GB, 4 to download)
# ====================================================================
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/tokenizer/merges.txt -->
#   ↑ 2MB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/tokenizer/tokenizer.json -->
#   ↑ 7MB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/tokenizer/tokenizer_config.json -->
#   ↑ 11KB
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/tokenizer/vocab.json -->
#   ↑ 3MB

# ====================================================================
## scheduler/ (1 files, 0.0GB, 1 to download)
# ====================================================================
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/scheduler/scheduler_config.json -->
#   ↑ 0KB

# ====================================================================
## audio_scheduler/ (1 files, 0.0GB, 1 to download)
# ====================================================================
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/audio_scheduler/scheduler_config.json -->
#   ↑ 0KB

# ====================================================================
## (root)/ (1 files, 0.0GB, ALL DOWNLOADED)
# ====================================================================
<!-- https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/model_index.json -->
#   [done] 3KB

# ====================================================================
## 汇总: 60 files, 144.1GB total
## 待下载: 35 files, 63.1GB
# ====================================================================

# 下载完成后，把文件按相对路径放到 ~/model/MiniMax-H3/
# 然后运行 06a：
#   conda activate minimax_h3
#   GPU=0 MODEL_PATH=~/model/MiniMax-H3 \
#     DEVICE=cuda:0 MAX_PIXELS=133120 FPS=24 NUM_FRAMES=124 SEED=42 \
#     TASK=fl2va FIRST_FRAME=/mnt/d/your_image.jpg PROMPT="..." \
#     OUTPUT_DIR=~/output/minimaxh3_rotate_results/results_int8 \
#     bash /mnt/c/code/media_code/minimax_h3/06a_diffusers_inference.sh
