# download_urls.md — 动态掩码（ENABLE_DYNAMIC_MASK）+ DINOv2（ENABLE_MLP_DYNAMIC）所需模型

> 用迅雷多线程下载，下完放到 `D:\wheel\` 对应子目录，然后在 WSL 里拷到 `~/model/`。
> HF 镜像 URL（hf-mirror.com）国内可直连；dl.fbaipublicfiles.com 可能需要代理。

## 1. SAM2.1 Hiera Large（动态掩码分割器）

文件: `sam2.1_hiera_large.pt`（~156MB）

下载地址（任选一个）:
```
# HF 镜像（推荐，国内直连）
https://hf-mirror.com/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt

# HuggingFace 原始
https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt
```

存放位置: `~/model/sam2/sam2.1_hiera_large.pt`

## 2. GroundingDINO-Tiny（动态掩码检测器）

完整模型仓库，需下载以下文件（放到同一个目录）:

```
# 模型权重（~700MB，最大文件）
https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/main/model.safetensors

# 配置文件（都很小）
https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/main/config.json
https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/main/preprocessor_config.json
https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/main/tokenizer_config.json
https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/main/tokenizer.json
https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/main/special_tokens_map.json
https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/main/vocab.txt
```

存放位置: `~/model/grounding-dino-tiny/`（所有文件放同一目录）

> 或者用 huggingface-cli 一键下载（WSL 内，需 HF_ENDPOINT 镜像）:
> ```bash
> huggingface-cli download IDEA-Research/grounding-dino-tiny --local-dir ~/model/grounding-dino-tiny
> ```

## 3. DINOv2 ViT-S/14 reg（MLP 动态感知特征提取）

文件: `dinov2_vits14_reg_03_pretrain.pth`（~85MB）

下载地址:
```
# Facebook 官方 CDN
https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg_03_pretrain.pth
```

存放位置: `~/model/dinov2/dinov2_vits14_reg.pth`

> 注意: 文件名要改（去掉 `_03_pretrain` 后缀，改为 `dinov2_vits14_reg.pth`）
> 或者代码也支持目录方式: 把 dinov2 仓 clone 到 `~/model/dinov2/`，代码会用 torch.hub source=local 加载

## 4. sam2 Python 包（需 clone + pip install）

```
# GitHub 仓库（提供 SAM2 模型代码 + config yaml）
https://github.com/facebookresearch/segment-anything-2
```

安装（WSL 内）:
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git ~/repos/segment-anything-2
cd ~/repos/segment-anything-2
pip install -e .
```

> config 文件 `sam2.1_hiera_large.yaml` 随仓库附带，在 `sam2/configs/sam2.1/` 下。

---

## 下载后配置

在 `proxy.env` 里添加（如果路径与默认值不同）:

```bash
# SAM2
export SAM2_MODEL_PATH="$HOME/model/sam2"
export SAM2_DIR="$HOME/repos/segment-anything-2"
# GroundingDINO（指向本地目录，跳过 HF 下载）
export GROUNDING_DINO_ID="$HOME/model/grounding-dino-tiny"
# DINOv2
export DINO_MODEL_PATH="$HOME/model/dinov2/dinov2_vits14_reg.pth"
```

## 从 D 盘拷到 WSL

```bash
# 假设下载到了 D:\wheel\ 下对应子目录
mkdir -p ~/model/sam2 ~/model/grounding-dino-tiny ~/model/dinov2
cp /mnt/d/wheel/sam2.1_hiera_large.pt ~/model/sam2/
cp -r /mnt/d/wheel/grounding-dino-tiny/* ~/model/grounding-dino-tiny/
cp /mnt/d/wheel/dinov2_vits14_reg_03_pretrain.pth ~/model/dinov2/dinov2_vits14_reg.pth
```

## 验证

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vggt_human
python -c "from sam2.build_sam import build_sam2; print('sam2 OK')"
python -c "from transformers import AutoModelForZeroShotObjectDetection; print('groundingdino OK')"
ls ~/model/sam2/sam2.1_hiera_large.pt ~/model/grounding-dino-tiny/model.safetensors ~/model/dinov2/dinov2_vits14_reg.pth
```
