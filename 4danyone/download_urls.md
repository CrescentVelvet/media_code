# 4DAnyone 离线下载清单
# ============================================================================
# 网络诊断结论（本机 WSL）：
#   - files.pythonhosted.org (PyPI wheel): 8 KB/s   ← 极慢
#   - pypi.tuna.tsinghua.edu.cn (清华):   100 KB/s ← 慢
#   - download.pytorch.org / huggingface.co / hf-mirror.com: 不通
#   - modelscope.cn: 230 KB/s              ← 可用（本机唯一能用的源）
#   - codeload.github.com: 5-10 MB/s       ← 快（代码已用此下载）
# 所以 torch wheel(~3GB) + 模型权重(~10-15GB) 都需手动下载到 D:\wheel\
# 脚本再用 --find-links / 本地拷贝 离线安装。
#
# 下载工具建议：IDM / 迅雷（多线程，可突破单连接限速）。批量导入本文件的 URL。

# ============================================================================
## A. pip wheel — torch 2.8 + torchvision 0.23 + nvidia-cu12 依赖 + triton（共 ~3GB）
## 保存到：D:\wheel\        （文件名即 wheel 名，互不冲突）
# ============================================================================
<!-- https://mirrors.aliyun.com/pypi/packages/5a/63/4fdc45a0304536e75a5e1b1bbfb1b56dd0e2743c48ee83ca729f7ce44162/torch-2.8.0-cp311-cp311-manylinux_2_28_x86_64.whl -->
<!-- https://mirrors.aliyun.com/pypi/packages/79/9c/fcb09aff941c8147d9e6aa6c8f67412a05622b0c750bcf796be4c85a58d4/torchvision-0.23.0-cp311-cp311-manylinux_2_28_x86_64.whl -->
<!-- https://mirrors.aliyun.com/pypi/packages/7d/39/43325b3b651d50187e591eefa22e236b2981afcebaefd4f2fc0ea99df191/triton-3.4.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl -->
<!-- https://mirrors.aliyun.com/pypi/packages/05/6b/32f747947df2da6994e999492ab306a903659555dddc0fbdeb9d71f75e52/nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl -->
<!-- https://mirrors.aliyun.com/pypi/packages/0d/9b/a997b638fcd068ad6e4d53b8551a7d30fe8b404d6f1804abf1df69838932/nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl -->
<!-- https://mirrors.aliyun.com/pypi/packages/f8/02/2adcaa145158bf1a8295d83591d22e4103dbfd821bcaf6f3f53151ca4ffa/nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl -->
<!-- https://mirrors.aliyun.com/pypi/packages/ba/51/e123d997aa098c61d029f76663dedbfb9bc8dcf8c60cbd6adbe42f76d049/nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl -->
<!-- https://mirrors.aliyun.com/pypi/packages/dc/61/e24b560ab2e2eaeb3c839129175fb330dfcfc29e5203196e5541a4c44682/nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl -->
<!-- https://mirrors.aliyun.com/pypi/packages/1f/13/ee4e00f30e676b66ae65b4f08cb5bcbb8392c03f54f2d5413ea99a5d1c80/nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl -->
<!-- https://mirrors.aliyun.com/pypi/packages/fb/aa/6584b56dc84ebe9cf93226a5cde4d99080c8e90ab40f0c27bda7a0f29aa1/nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl -->
<!-- https://mirrors.aliyun.com/pypi/packages/85/48/9a13d2975803e8cf2777d5ed57b87a0b6ca2cc795f9a4f59796a910bfb80/nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl -->
<!-- https://mirrors.aliyun.com/pypi/packages/c2/f5/e1854cb2f2bcd4280c44736c93550cc300ff4b8c95ebe370d0aa7d2b473d/nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl -->
<!-- https://mirrors.aliyun.com/pypi/packages/56/79/12978b96bd44274fe38b5dde5cfb660b1d114f70a65ef962bcbbed99b549/nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl -->
<!-- https://mirrors.aliyun.com/pypi/packages/5c/5b/4e4fff7bad39adf89f735f2bc87248c81db71205b62bcc0d5ca5b606b3c3/nvidia_nccl_cu12-2.27.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl -->
<!-- https://mirrors.aliyun.com/pypi/packages/a2/eb/86626c1bbc2edb86323022371c39aa48df6fd8b0a1647bc274577f72e90b/nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl -->
<!-- https://mirrors.aliyun.com/pypi/packages/f6/74/86a07f1d0f42998ca31312f998bd3b9a7eff7f52378f4f270c8679c77fb9/nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl -->
<!-- https://mirrors.aliyun.com/pypi/packages/bb/fe/1bcba1dfbfb8d01be8d93f07bfc502c93fa23afa6fd5ab3fc7c1df71038a/nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl -->

# ============================================================================
## B. 4DAnyone 模型权重（modelscope，~10-15GB）
## 保存到：D:\wheel\4danyone_ms\ 下，按下面的相对路径建子目录
## （安装时脚本会从 D:\wheel\4danyone_ms\ 拷到 $MODEL_DIR）
##   例如 model.safetensors 存到 D:\wheel\4danyone_ms\4danyone\model.safetensors
# ============================================================================
# 4danyone/ (主模型 + 文本编码器 + VAE + tokenizer)
<!-- https://modelscope.cn/models/AntResearch/4DAnyone/resolve/master/4danyone/model.safetensors -->
#   ↑ 存到 D:\wheel\4danyone_ms\4danyone\model.safetensors
<!-- https://modelscope.cn/models/AntResearch/4DAnyone/resolve/master/4danyone/smplx_to_goliath70.pt -->
<!-- https://modelscope.cn/models/AntResearch/4DAnyone/resolve/master/4danyone/Wan2.2_VAE.pth -->
<!-- https://modelscope.cn/models/AntResearch/4DAnyone/resolve/master/4danyone/models_t5_umt5-xxl-enc-bf16.pth -->
#   ↑ ~5GB，最大文件，建议用 IDM/迅雷多线程下
<!-- https://modelscope.cn/models/AntResearch/4DAnyone/resolve/master/4danyone/umt5-xxl/special_tokens_map.json -->
<!-- https://modelscope.cn/models/AntResearch/4DAnyone/resolve/master/4danyone/umt5-xxl/spiece.model -->
<!-- https://modelscope.cn/models/AntResearch/4DAnyone/resolve/master/4danyone/umt5-xxl/tokenizer.json -->
<!-- https://modelscope.cn/models/AntResearch/4DAnyone/resolve/master/4danyone/umt5-xxl/tokenizer_config.json -->
#   ↑ 这4个 tokenizer 文件都存到 D:\wheel\4danyone_ms\4danyone\umt5-xxl\
# gvhmr/ (运动恢复: GVHMR + HMR2 + ViTPose + YOLOv8x)
<!-- https://modelscope.cn/models/AntResearch/4DAnyone/resolve/master/gvhmr/gvhmr_siga24_release.ckpt -->
<!-- https://modelscope.cn/models/AntResearch/4DAnyone/resolve/master/gvhmr/epoch=10-step=25000.ckpt -->
<!-- https://modelscope.cn/models/AntResearch/4DAnyone/resolve/master/gvhmr/vitpose-h-multi-coco.pth -->
<!-- https://modelscope.cn/models/AntResearch/4DAnyone/resolve/master/gvhmr/yolov8x.pt -->
#   ↑ 都存到 D:\wheel\4danyone_ms\gvhmr\
# perceptual/ (VGG-19 感知模型)
<!-- https://modelscope.cn/models/AntResearch/4DAnyone/resolve/master/perceptual/imagenet-vgg-verydeep-19-conv.safetensors -->
#   ↑ 存到 D:\wheel\4danyone_ms\perceptual\

# ============================================================================
## C. BiRefNet 前景分割模型（~900MB，不在上面的 modelscope repo 里）
<!-- https://modelscope.cn/models/AI-ModelScope/ZhengPeng7-BiRefNet/file/view/master/BiRefNet_config.py -->
<!-- https://modelscope.cn/models/AI-ModelScope/ZhengPeng7-BiRefNet/file/view/master/birefnet.py -->
<!-- https://modelscope.cn/models/AI-ModelScope/ZhengPeng7-BiRefNet/file/view/master/config.json -->
<!-- https://modelscope.cn/models/AI-ModelScope/ZhengPeng7-BiRefNet/file/view/master/model.safetensors -->
#   ↑ 这 4 个文件都存到 D:\wheel\birefnet\
# ============================================================================

# ============================================================================
## D. SMPL-X 人体模型（单独授权，~50MB，不在 HF/modelscope 上）
## 1. 注册账号 + 接受许可：https://smpl-x.is.tue.mpg.de/
## 2. 下载 models_smplx_v1_1.zip
## 3. 存到 D:\wheel\  （任意位置，安装时用 SMPLX_ARCHIVE 指向它）
# ============================================================================
