# VGGT-Human 动态掩码离线下载清单
# ============================================================================
# 动态掩码（ENABLE_DYNAMIC_MASK=1）需 SAM2 + GroundingDINO；
# MLP 动态感知（ENABLE_MLP_DYNAMIC=1）需 DINOv2。
# 当前 WSL 环境三者皆缺，以下载到 D:\wheel\vggt_human_ms\ 为准，脚本直接从 D 盘读。
#
# 下载工具建议：IDM / 迅雷（多线程）。批量导入本文件的 URL。
# 全部下完后在 proxy.env 确认路径，然后开 ENABLE_DYNAMIC_MASK=1 跑 04。
# ============================================================================

# ============================================================================
## A. SAM2.1 Hiera Large（动态掩码分割器，~156MB）
## 保存到：D:\wheel\vggt_human_ms\sam2\sam2.1_hiera_large.pt
## proxy.env: export SAM2_MODEL_PATH="/mnt/d/wheel/vggt_human_ms/sam2"
# ============================================================================
<!-- https://hf-mirror.com/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt -->

# ============================================================================
## B. GroundingDINO-Tiny（动态掩码检测器，~700MB）
## 保存到：D:\wheel\vggt_human_ms\grounding-dino-tiny\ 下，所有文件放同一目录
## proxy.env: export GROUNDING_DINO_ID="/mnt/d/wheel/vggt_human_ms/grounding-dino-tiny"
##（transformers from_pretrained 会从该目录加载，设了就不用联网下载）
# ============================================================================
# 模型权重（最大文件）
<!-- https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/main/model.safetensors -->
# 配置文件（都很小）
<!-- https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/main/config.json -->
<!-- https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/main/preprocessor_config.json -->
<!-- https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/main/tokenizer_config.json -->
<!-- https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/main/tokenizer.json -->
<!-- https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/main/special_tokens_map.json -->
<!-- https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/main/vocab.txt -->

# ============================================================================
## C. DINOv2 ViT-S/14 reg（MLP 动态感知特征提取，~85MB）
## 保存到：D:\wheel\vggt_human_ms\dinov2\dinov2_vits14_reg4_pretrain.pth
## proxy.env: export DINO_MODEL_PATH="/mnt/d/wheel/vggt_human_ms/dinov2/dinov2_vits14_reg4_pretrain.pth"
# ============================================================================
<!-- https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth -->

# ============================================================================
## D. sam2 Python 包（需 clone + pip install，不是下载文件）
## WSL 内执行（提供 SAM2 模型代码 + config yaml）：
##   git clone https://github.com/facebookresearch/segment-anything-2.git ~/repos/segment-anything-2
##   cd ~/repos/segment-anything-2 && pip install -e .
## config 文件 sam2.1_hiera_large.yaml 随仓库附带，在 sam2/configs/sam2.1/ 下。
## proxy.env: export SAM2_DIR="$HOME/repos/segment-anything-2"
# ============================================================================

# ============================================================================
## E. 无需手动下载（transformers 自动从 HF 缓存加载）
##   - GroundingDINO: 若 proxy.env 未设 GROUNDING_DINO_ID 指向本地目录，
##     transformers 会从 hf-mirror.com 自动下载到 ~/.cache/huggingface/。
##     设了本地目录就跳过联网。
##   - SAM (fallback): 若 SAM2 加载失败，代码 fallback 用 facebook/sam-vit-base
##     （transformers 自动下载）。一般不需要，SAM2 就够了。
# ============================================================================

# ============================================================================
## F. SwinIR 去噪 checkpoint（DENOISER=swinir，~15MB）
## 保存到：D:\wheel\vggt_human_ms\SwinIR\005_colorDN_DFOWM_s64w8_SwinIR-M_x1.pth
## proxy.env: export SWINIR_CKPT="/mnt/d/wheel/vggt_human_ms/SwinIR/005_colorDN_DFOWM_s64w8_SwinIR-M_x1.pth"
## SwinIR 代码仓已通过 00a 的 GitHub 代理 clone 到 ~/repos/SwinIR（无需手动下）
# ============================================================================
<!-- https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFOWM_s64w8_SwinIR-M_x1.pth -->

# ============================================================================
## G. DiffBIR 去噪 checkpoint（DENOISER=diffbir，~1.3GB）
## 保存到：D:\wheel\vggt_human_ms\DiffBIR\v1_general.pth
## proxy.env: export DIFFBIR_CKPT="/mnt/d/wheel/vggt_human_ms/DiffBIR/v1_general.pth"
## DiffBIR 代码仓已通过 00a 的 GitHub 代理 clone 到 ~/repos/DiffBIR（无需手动下）
## 注意：DiffBIR 完整推理还需 SD2.1 base model（较复杂），建议先用 DENOISER=swinir
# ============================================================================
<!-- https://hf-mirror.com/lxq007/DiffBIR-v2/resolve/main/v1_general.pth -->

# ============================================================================
## H. DENOISER=none（不需要任何 checkpoint）
## 跳过去噪，只渲染新视角 + AdaIN 颜色校正。虚拟相机仍加入增强 COLMAP 场景。
## 适合快速测试或不需要去噪的场景。
# ============================================================================
