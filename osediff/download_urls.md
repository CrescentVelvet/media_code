# OSEDiff 离线下载清单
# ============================================================================
# 网络诊断结论（本机 WSL，单连接测速 3MB）：
#   - download.pytorch.org (torch wheel): 103 KB/s  ← 慢但可用，IDM 多线程可加速
#   - pypi.tuna.tsinghua.edu.cn (清华 PyPI): 362 KB/s ← 中（依赖 wheel 在线装够用）
#   - hf-mirror.com/Manojb/... (SD2.1 safetensors): 111 KB/s ← 跳美国 CDN，慢
#   - hf-mirror.com/spaces/xinyu1205/... (RAM):  68 KB/s  ← 慢
#   - modelscope.cn/AI-ModelScope/... (SD2.1 .bin): 202 KB/s ← 国内 CDN 较快，
#       但缺 model_index.json + 是 .bin 而非 .safetensors，diffusers 不能直接加载，
#       故 SD2.1 仍走 hf-mirror（safetensors 标准格式）。
#   - codeload.github.com (OSEDiff repo clone): 5-10 MB/s ← 快，00a 自动 clone
# 所以 torch wheel(~2.2GB) + SD2.1(~5GB) + RAM(~1.7GB) 都需手动下载到 D:\wheel\，
# 脚本再用本地拷贝离线加载（01 会自动检测 D:\wheel\osediff_ms\）。
#
# 下载工具建议：IDM / 迅雷（多线程，可突破单连接限速）。批量导入本文件的 URL。
# 全部下完后跑：bash osediff/00a_setup_env.sh && bash osediff/01_download_models.sh
# ============================================================================

# ============================================================================
## A. pip wheel — torch 2.0.1 + torchvision 0.15.2（cu118，共 ~2.2GB）
## 保存到：D:\wheel\        （cu118 wheel 自包含 CUDA，无需 nvidia-* 依赖包）
## 00a 会自动检测 D:\wheel\torch-*.whl 本地安装。
# ============================================================================
<!-- https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl -->
<!-- https://download.pytorch.org/whl/cu118/torchvision-0.15.2%2Bcu118-cp310-cp310-linux_x86_64.whl -->

# ============================================================================
## B. SD2.1-Base 权重（hf-mirror，safetensors fp32，共 ~5.1GB）
## 保存到：D:\wheel\osediff_ms\sd21_base\ 下，按下面的相对路径建子目录
## （01 会从 D:\wheel\osediff_ms\sd21_base\ 拷到 $MODEL_DIR/sd21_base/）
##   例如 unet/diffusion_pytorch_model.safetensors 存到
##       D:\wheel\osediff_ms\sd21_base\unet\diffusion_pytorch_model.safetensors
# ============================================================================
# 根目录
<!-- https://hf-mirror.com/Manojb/stable-diffusion-2-1-base/resolve/main/model_index.json -->
#   ↑ 存到 D:\wheel\osediff_ms\sd21_base\model_index.json
# scheduler/
<!-- https://hf-mirror.com/Manojb/stable-diffusion-2-1-base/resolve/main/scheduler/scheduler_config.json -->
# tokenizer/（4 个小文件都存到 D:\wheel\osediff_ms\sd21_base\tokenizer\）
<!-- https://hf-mirror.com/Manojb/stable-diffusion-2-1-base/resolve/main/tokenizer/merges.txt -->
<!-- https://hf-mirror.com/Manojb/stable-diffusion-2-1-base/resolve/main/tokenizer/special_tokens_map.json -->
<!-- https://hf-mirror.com/Manojb/stable-diffusion-2-1-base/resolve/main/tokenizer/tokenizer_config.json -->
<!-- https://hf-mirror.com/Manojb/stable-diffusion-2-1-base/resolve/main/tokenizer/vocab.json -->
# text_encoder/（config.json 613B + model.safetensors ~1.4GB，最大文件之一）
<!-- https://hf-mirror.com/Manojb/stable-diffusion-2-1-base/resolve/main/text_encoder/config.json -->
<!-- https://hf-mirror.com/Manojb/stable-diffusion-2-1-base/resolve/main/text_encoder/model.safetensors -->
# unet/（config.json 911B + diffusion_pytorch_model.safetensors ~3.4GB，最大文件）
<!-- https://hf-mirror.com/Manojb/stable-diffusion-2-1-base/resolve/main/unet/config.json -->
<!-- https://hf-mirror.com/Manojb/stable-diffusion-2-1-base/resolve/main/unet/diffusion_pytorch_model.safetensors -->
# vae/（config.json 553B + diffusion_pytorch_model.safetensors ~335MB）
<!-- https://hf-mirror.com/Manojb/stable-diffusion-2-1-base/resolve/main/vae/config.json -->
<!-- https://hf-mirror.com/Manojb/stable-diffusion-2-1-base/resolve/main/vae/diffusion_pytorch_model.safetensors -->

# ============================================================================
## C. RAM 权重（hf-mirror spaces，~1.7GB）
## 注意：RAM 在 HF 的 spaces 仓库（不是 models），路径含 /spaces/
## 保存到：D:\wheel\osediff_ms\ram_swin_large_14m.pth
## （01 会从 D:\wheel\osediff_ms\ 拷到 $MODEL_DIR/ram_swin_large_14m.pth）
# ============================================================================
<!-- https://hf-mirror.com/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth -->

# ============================================================================
## D. 无需手动下载（00a clone OSEDiff repo 后自带）
##   - osediff.pkl (20MB)      ← SR 主权重，在 preset/models/osediff.pkl
##   - osediff_face.pkl (18MB) ← 人脸修复权重，在 preset/models/osediff_face.pkl
##   - DAPE.pth (~7MB)          ← RAM fine-tuned，在 preset/models/DAPE.pth
##      若推理报 DAPE 维度不匹配，再从 Google Drive 下完整版（见 README 排错）。
# ============================================================================

# ============================================================================
## E. xformers==0.0.20（~200MB，00a 在线装，一般不用手动下）
## 00a 会先试 pip install xformers==0.0.20（PyPI 清华 362KB/s，200MB 约 9 分钟）；
## 预编译 wheel 匹配 torch 2.0.1+cu118。若在线装失败（wheel 不匹配），
## 00a 自动 conda 装 cuda-nvcc + gxx 源码编译。仍失败才需手动：
##   pip download xformers==0.0.20 --dest D:\wheel\xformers\ --no-deps
## 然后重跑 00a。
# ============================================================================
