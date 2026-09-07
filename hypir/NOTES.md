# HYPIR — 原理详解与排错手册（NOTES）

> 本文件从 README.md 迁入：流程原理、官方代码机制、暖启动机制、常见报错修法。
> 运行命令与参数速查看 [README.md](README.md)；实验设计与结果看 [EXPERIMENTS.md](EXPERIMENTS.md)。

## Pipeline（推理流程详解）

对应官方代码 `HYPIR/enhancer/base.py::enhance` + `HYPIR/enhancer/sd2.py::forward_generator` + `HYPIR/utils/common.py`。一张 LQ 图像从输入到输出经过 5 步：

1. **上采样（bicubic 插值）** — `F.interpolate(lq, scale_factor=upscale, mode="bicubic")`。先把 LQ 双三次插值放大到目标分辨率（`scale_by=factor` 时按固定倍数，`longest_side` 时按长边到固定尺寸）。这一步的输出同时存为 `ref`（参考图），供第 5 步小波融合用。若短边 ≤ `patch_size`(512)，还会把短边 resize 到至少 512，保证 VAE 有足够大的画布（此 resize 只作用于送入 VAE 的 `lq`，不改动 `ref`）。
2. **VAE 编码（分块，patch=512）** — `lq` 归一化到 `[-1,1]`、pad 到 8 的倍数（`vae_scale_factor=8`），再用 `make_tiled_fn` 分块送入 `vae.encode(...).latent_dist.sample()`。tile 在**像素空间**大小为 `patch_size`(512)、stride=`stride`(256)，下采样到**潜空间** 64（512/8）。重叠区用高斯权重平滑拼接，避免接缝。注意 `.sample()` 从 VAE 编码分布里采样，带轻微随机性——这就是推理要设 `--seed` 的原因。
3. **UNet 一步去噪（LoRA + SD2，t=200）** — 见下方「一步去噪」。加载了 LoRA 的 SD2 UNet 以 LQ 潜变量为输入、在 timestep=200 做一次前向 + 一次 DDPM 反推，得到复原的 HR 潜变量。同样分块（潜空间 64 / stride 32）。
4. **VAE 解码（分块）** — `make_tiled_fn(vae.decode(tile).sample, scale_type="up", scale=8, channel=3)` 把复原潜变量解码回像素空间（潜 64 → 像素 512 的 tile）。然后裁掉 padding、`(x+1)/2` 回到 `[0,1]`、bicubic 缩回 `ref` 的尺寸 `(h0,w0)`。
5. **小波融合** — `wavelet_reconstruction(x, ref)`：把第 4 步解码输出（`content`）与第 1 步上采样参考图（`style=ref`）做多尺度融合。见下方「小波融合」。

### 一步去噪（one-step denoising）怎么回事

标准 DDPM 生成要迭代 ~1000 步（DDIM ~50 步）逐步去噪。**HYPIR 不迭代，只走一步**，关键在于把「LQ 潜变量」直接当作部分加噪的 `x_t`：

```
z_in = z_lq * vae.scaling_factor          # LQ 潜变量当 x_t（不再加随机噪声！）
eps  = UNet_lora(z_in, t=200, text_embed)  # UNet 预测"把 x_0 变成 x_t 的噪声"
z0   = scheduler.step(eps, coeff_t=200, z_in).pred_original_sample   # 一步反推 x_0
```

- DDPM 有闭式关系：`x_0 = (x_t − √(1−ᾱ_t)·eps) / √(ᾱ_t)`。给定 `x_t`（=LQ 潜变量）和 UNet 预测的噪声 `eps`，**一次代数运算**就能解出估计的干净潜变量 `x_0`，不需要迭代。
- 为什么一步就够？因为 LoRA 是**专门为 t=200 这一步训练**的：训练时同样把 GT 的 LQ 潜变量当 `x_t`、UNet 预测 `eps`、一步反推 `x_0`，再用 `x_0` 解码出的图像与 GT HR 算 L2+LPIPS+GAN 损失（见 `HYPIR/trainer/base.py::optimize_generator`）。也就是说 LoRA 学到的就是「在 t=200 这一步、从 LQ 潜变量一步反推出 HR 潜变量」这个映射，推理时自然一步即出。
- `model_t=200` 是喂给 UNet 的时间步标签（UNet 以为自己正在 t=200 去噪）；`coeff_t=200` 是反推 `x_0` 时用的噪声调度系数所在时间步。两者在此都取 200。
- 这就是论文标题里的 "score prior"：扩散 UNet 的噪声预测（score）在单个时间步上提供了一个强先验，把退化图直接拉回干净图——快（一次前向）且借用了 SD2 在海量图像上学到的先验。

### 小波融合（wavelet reconstruction）怎么回事

解码出的 HR 图像纹理清晰，但扩散模型可能引入色偏/结构幻觉；而上采样 LQ 颜色和整体结构是可靠的，只是模糊（缺高频）。小波融合取两者之长：

- `wavelet_decomposition(img, levels=5)` 用逐级放大的高斯模糊（dilation 半径 1,2,4,8,16）把图像拆成 **低频**（`low`：颜色、光照、大尺度结构）和 **高频**（`high`：边缘、纹理、细节）。
- `wavelet_reconstruction(content=x_decoded, style=ref)` 的实际计算是：
  ```
  result = content_high + style_low
         = 解码输出的高频 + 上采样LQ的低频
  ```
  即**高频（纹理/边缘）来自扩散解码输出，低频（颜色/结构）来自上采样 LQ**。
- 效果：最终图保留 LQ 原本正确的颜色与构图（低频），同时注入扩散模型生成的锐利纹理（高频），避免色偏和结构漂移，又实现了超分。这是 SUPIR/CCSR 一脉相承的经典 trick。

> 总结一句：**上采样定颜色结构 → VAE 编码进潜空间 → LoRA-UNet 一步反推干净潜变量 → VAE 解码回像素 → 与上采样原图小波融合（取扩散的高频 + LQ 的低频）**。

## 合成退化流程（训练时 HQ→LQ，03c/04c 与 06 预览共用）

官方 `RealESRGANDataset` + `RealESRGANBatchTransform` 把 HQ 合成退化成 LQ（HYPIR 发布模型本身的训练方式；03c/04c 在线跑、06 离线预览复用同一套，参数取自 `configs/sd2_train.yaml`，无复制）。

**1) 取 HQ + 生成核**（`RealESRGANDataset.__getitem__`）
- 加载 HQ；`crop_type=none` 时 resize 到 `out_size`(512)（`random` 则裁 512 patch）。
- 随机生成 3 个核：`kernel1`/`kernel2`（模糊核：iso/aniso/generalized/plateau，或 sinc 低通，按 `kernel_prob`/`sinc_prob`）、`sinc_kernel`（最终 sinc 滤波核）。
- 可选 flip/rot 增强（06 预览/确定性时关掉）。
- 返回 `{hq, kernel1, kernel2, sinc_kernel, txt}`。

**2) 两阶段退化**（`RealESRGANBatchTransform.__call__`，对 `hq` 操作，参数全在 config 的 `batch_transform.params`）
- `GT = USMSharp(hq)`：先 USM 锐化 HQ 当 GT（匹配发布模型训练时的 GT 预处理）。
- **第一阶段**：`filter2D(hq, kernel1)` 模糊 → 随机 resize（up/down/keep，scale 取自 `resize_range`，mode 随机 area/bilinear/bicubic）→ 加噪（按 `gaussian_noise_prob` 选高斯或泊松，强度 `noise_range`/`poisson_scale_range`，可灰噪）→ JPEG（质量取自 `jpeg_range`）。
- **第二阶段**：按 `second_blur_prob` 可能再 `filter2D(kernel2)` 模糊 → 按 `stage2_scale`(≈4) 下采样 → 随机 resize2 → 加噪2 → **resize-back + sinc 滤波** 与 **JPEG2** 顺序随机（各 0.5，避免扭曲条纹）→ 若 `resize_back=true` 缩回原尺寸(512)。
- `LQ = clamp(round(out·255))/255`（量化到 8-bit）。

**3) 训练池**（`queue_size>0` 时，仅训练用）
- 把 `{GT,LQ,txt}` 入队（池容量 `queue_size`=256）；满后随机抽一个**缓存样本**返回（增 batch 内多样性），故 `queue_size` 须被 `batch_size`(每卡) 整除（`256%6` 报错 → 用 `BATCH_SIZE=8` 或改 `queue_size=252`，见排错 #11）。
- **06 预览脚本设 `queue_size=0`**：跳过池，直接返回**当前 HQ 的 LQ**（否则满 256 后返回的是别的图的缓存，看不到当前 HQ 的退化）。

> 一句话：`HQ →(USM锐化)→ GT`；`HQ →(两阶段 blur/resize/noise/jpeg/sinc)→ LQ`。训练时 LQ 现场合、每 epoch 随机刷新；06 离线跑同一套、`queue_size=0` 看当前图。

> ⚠️ **本仓 clone 的 `HYPIR/dataset/batch_transform.py` 是修改版**（只高斯模糊）：`__call__` 不走上面官方那套两阶段 blur/resize/noise/jpeg/sinc，而是简化为——
> ```python
> # 修改版 __call__（你的 clone 实际跑的）
> hq = USMSharp(batch[hq_key])                 # GT = USM 锐化后的 HQ
> kernel = random.randint(1,5)*2+1             # 3/5/7/9/11
> num = random.randint(1,5)                    # 重复 1-5 次
> lq = hq.clone()
> for _ in range(num):
>     lq = torchvision.transforms.GaussianBlur(kernel_size=kernel, sigma=(1.0,2.0))(lq)
> # 没有 resize / noise(gauss/poisson) / JPEG / sinc / 第二阶段
> ```
> 所以上面官方流程的**第一阶段后半（resize/noise/JPEG）和整个第二阶段（sinc/JPEG2/resize-back）在你环境里不发生**——LQ 只是「HQ 被 1-5 次随机高斯模糊」。`queue_size` 训练池保留（故 `256%6` 仍报错，见排错 #11）。**以你 clone 的 `batch_transform.py` 实际代码为准**；06 预览和 04c 训练复用的就是这版，看到/用到的是纯高斯模糊退化。想恢复完整官方退化就把 `batch_transform.py` 还原成 GitHub 版。

## 配对人脸微调（03b/04b）：机制与数据流

The official `03`/`04` path trains on **HQ only** and *synthesizes* LQ via
RealESRGAN degradation. To train on **real** degradation (e.g. the 360p camera
LQ + RAW-decoded HQ face pairs from `face_crop/crop_faces_paired.py`), use the
paired plug-ins — **no official file is modified**: the derived config points
`target:` at plug-in classes in `paired_face_plugin.py` (importable via
PYTHONPATH), and `train_paired.py` subclasses `SD2Trainer` to warm-start the
LoRA from the released `HYPIR_sd2.pth`.

> 中文说明：官方 `03/04` 只用 HQ、LQ 是现场用 RealESRGAN 合成退化的（盲复原配方）。
> 你用 `face_crop/crop_faces_paired.py` 建的是「真实 360p 相机 LQ + RAW 解码 HQ」配对，
> 想直接喂真实退化，就用这套配对插件。**不改任何官方文件**：填好的配置把 `target:`
> 指向 `paired_face_plugin.py` 里的插件类（靠 `04b_train_paired.sh` 设的 `PYTHONPATH`
> 导入），`train_paired.py` 再子类化 `SD2Trainer` 实现从发布权重暖启动。
>
> 核心思想：7k 张配对从零训 LoRA 不够（发布模型是 bs1024 大数据训的），所以默认
> **暖启动**——把发布的 `HYPIR_sd2.pth` 当 LoRA 初始化，再在你的人脸数据上继续练，
> 等于把通用复原模型「适配」到人脸域。数据流见下图，训练循环(一步去噪+L2/LPIPS/GAN)
> 与官方完全一致，只是 LQ 来自真实配对而非合成。

```
HQ folder (.../hq/<stem>_faceN.png)  ┐  paired by filename
LQ folder (.../lq/<stem>_faceN.png)  ┘
   └─ build_paired_dataset.py     -> parquet(hq_path, lq_path, prompt)
        └─ PairedFaceDataset        : load HQ+LQ, resize/paired-crop to 512,
                                      same flip/rot -> {hq, lq, txt}
        └─ PairedFaceBatchTransform : USM-sharpen HQ, rename -> {GT, LQ, txt}
        └─ FineTuneSD2Trainer       : warm-start LoRA from HYPIR_sd2.pth, then
                                      the unchanged one-step + L2/LPIPS/GAN loop
```

> 给第一次用的人（一句话版）：把「模糊的 360p LQ 人脸」和「清晰的 HQ 人脸」**成对**喂给模型，
> 让它学会把模糊脸还原成清晰脸。不是从零训练，而是在官方已发布的 `HYPIR_sd2.pth` 上
> **继续练（暖启动）**，让它专精你的人脸数据——所以 7 千张就够、收敛也快。
> 前置：已用 `face_crop/crop_faces_paired.py` 建好配对数据集并上传到默认路径。

## 暖启动机制

**背景**：官方 GitHub 的 `HYPIR/trainer/sd2.py` 里 `SD2Trainer.init_generator` 只做 `init_lora_weights="gaussian"`（随机初始化），**完全不加载 LoRA 权重** —— 即官方默认是从零训，没有暖启动。

**改动**：本仓库 clone 的 `HYPIR/HYPIR/trainer/sd2.py` 被改过，在 `init_generator` 里加了加载逻辑（`grep -n "weight_path\|torch.load" sd2.py` 可见第 57-58 行）：
```python
print(f"Load model weights from {self.config.weight_path}")
state_dict = torch.load(self.config.weight_path, map_location="cpu", weights_only=False)
# ... load_state_dict ...
```
**这是暖启动能生效的唯一原因**。没有这两行，下面的链路全断。

**暖启动链路**（04c / 04b 都走这条）：
```
config.lora_weight_path = .../HYPIR_sd2.pth        # 04c/04b 在填好的 config 里设
  ↓ FineTuneSD2Trainer.init_generator (paired_face_plugin.py)
config.weight_path = config.lora_weight_path        # 官方 config 无 weight_path 字段 → hasattr=False → 赋值
  ↓ super().init_generator()  (你改过的 sd2.py)
torch.load(config.weight_path) → load_state_dict    # 真正把发布 LoRA 载入 UNet
  → 暖启动成功（checkpoint-* 是暖启动来的，不是从零）
```

**勿改错清单**：
- 🔴 **别把 clone 里的 `sd2.py` 还原成官方版**。一旦丢了第 57-58 行的 `torch.load`，`FineTuneSD2Trainer` 设的 `weight_path` 无人读 → 暖启动静默失效，变回从零训（不会报错，但 checkpoint 质量退化，且很难察觉）。
- 🔴 **别删 `FineTuneSD2Trainer` 里的 `config.weight_path = lora_wp` 映射**。否则 `sd2.py` 读 `config.weight_path` 时缺键 → `Missing key weight_path` 直接崩。
- 🟡 改过的 `sd2.py` 是**无条件** `torch.load(config.weight_path)`（grep 未见 `if` guard）。所以 `lora_weight_path` 留空（想从零训）会把 `weight_path=None` 传进去 → `torch.load(None)` 崩。**04c / 04b 务必设 `lora_weight_path`**；从零训用 `04_train.sh`（官方 `train.py`），但跑前先 `grep` 确认 `sd2.py` 对缺 `weight_path` 的处理，否则同样崩。

**定期自检**（确认暖启动仍生效）：
```bash
grep -n "weight_path\|torch.load" $HYPIR_DIR/HYPIR/trainer/sd2.py
# 应见 init_generator 里那两行 torch.load(self.config.weight_path)；没有 = 暖启动已坏
```

## 可能遇到的问题

公司代理做 HTTPS 中间人解密，下面按流水线阶段列出常见报错与修法（命令在服务器上、conda 环境已激活时执行）。

**1. clone/pull 本仓或官方仓报错**
- 报 `SSL certificate problem` / 认证：公开仓免认证，加 `-c http.sslVerify=false`（`run_all.sh` 已对官方仓做 `sslVerify=false` 兜底）。
- 报 `Failed to connect to github.com port 443`（连不上，**非 SSL**）：git 没走代理。设全局代理（带认证的把用户名密码写进 URL，**密码特殊字符必须 URL 编码**，否则 git 解析错/连不上）：
  ```bash
  git config --global https.proxy http://USER:PASS@proxyhk.huawei.com:8080
  git config --global http.proxy  http://USER:PASS@proxyhk.huawei.com:8080   # 顺手也设 http
  # 取消：git config --global --unset https.proxy ; git config --global --unset http.proxy
  ```
  ⚠️ 密码特殊字符 URL 编码：`*`→`%2A`、`+`→`%2B`、`@`→`%40`、`:`→`%3A`、`#`→`%23`、`&`→`%26`、`=`→`%3D`。例：密码 `p*ss+word` 写成 `p%2Ass%2Bword`。
  > 这和 `_env.sh` 里的 `http_proxy`/`https_proxy` 环境变量是**两套**：环境变量给 curl/hf/pip 用，`git config http.proxy` 给 git 本身用；两个都设最稳。
- 报 `Failed to connect to proxyhk.huawei.com port 8080: No route to host`（代理都连不上 → git/pip/hf 全挂）：根因是 **docker 网桥网段和代理 IP 冲突**——`docker-compose` 建 network 时分的子网（如 `172.18.0.0/16`，网桥 `br-407a71493298`）把 `proxyhk.huawei.com` 解析到的 `172.18.100.92` 包进去了，内核把去代理的流量送进 docker 网桥而非物理网卡。排查 + 修：
  ```bash
  # 1) 查代理 IP + 看它落进哪个网桥网段（命中即冲突）：
  getent hosts proxyhk.huawei.com        # 例: 172.18.100.92
  ip route | grep 172.18                # 命中 172.18.0.0/16 dev br-xxxxx → 冲突
  # 2) 加一条主机路由，把代理 IP 强制走物理网卡（ens1f0 换成你的网卡，网关用默认网关）：
  ip route show default                  # 取默认网关，例 10.x.x.1
  sudo ip route add 172.18.100.92 via 10.x.x.1 dev ens1f0
  # 3) 验证：应能连代理了
  curl -x http://USER:PASS@proxyhk.huawei.com:8080 https://github.com -I   # 200/302 即通
  ```
  持久化（重启不丢）：写到 `/etc/network/if-up.d/` 脚本或 `nmcli`；治本是在 `/etc/docker/daemon.json` 配 `default-address-pools` 给 docker 分不与代理冲突的子网（如 `10.250.0.0/16`），再 `systemctl restart docker`。
  > 同机其他容器/虚拟网桥也可能占 `172.17.0.0/16`、`172.19.0.0/16` 等；只要代理 IP 落进任一 docker 网段就中招。

**2. `pip install -r requirements.txt` 报 `SSL:CERTIFICATE_VERIFY_FAILED` / 超时**
```bash
pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org download.pytorch.org"
INSTALL_DEPS=1 bash hypir/00_setup_env.sh
```
仍超时（大文件 torch==2.6.0）：加 `--timeout 600 --retries 10`，或先单独装 torch：`pip install --timeout 600 --retries 10 torch==2.6.0` 再重跑。不想动现有 torch 就 `SKIP_TORCH=1 INSTALL_DEPS=1 bash hypir/00_setup_env.sh`。

**3. `hf download` 报 `CAS service error : ReqwestMiddleware`**
HF 的 Xet/Rust 通道不认代理。`_env.sh` 已设 `HF_HUB_DISABLE_XET=1`；仍报则彻底卸载：
```bash
pip uninstall -y hf_xet
bash hypir/01_download_models.sh
```

**4. `hf download` 报 `SSLCertVerificationError`**
代理根 CA 不在系统证书包。先一次性建包，`_env.sh` 会自动用 `~/.ca-bundle.crt`：
```bash
bash hypir/setup_ca_bundle.sh    # 抓代理证书链 -> ~/.ca-bundle.crt，并自检
bash hypir/01_download_models.sh
```
- 自检 `[OK]` → 直接重跑 `01`。
- 自检 `[FAIL]` → 把公司根 CA 追加到 `~/.ca-bundle.crt` 后重跑（公司根 CA 常见于 `/usr/local/share/ca-certificates/`，脚本已自动并入）。
- 仍报 SSL（CDN 端点用了不同的 MITM 证书）→ `01` 自动回退到禁用 SSL 校验的下载器（`_hf_download.py`）；或直接 `HF_DISABLE_SSL=1 bash hypir/01_download_models.sh`。

**5. `hf download` 报 `repository not found for url .../stable-diffusion-2-1-base`**
原始 `stabilityai/stable-diffusion-2-1-base` 已从 HuggingFace 下架。脚本默认改用公开镜像 `Manojb/stable-diffusion-2-1-base`（完整 diffusers 格式，非 gated，无需 token）。若你仍指向旧的 `stabilityai/...`，会报此错；改成默认源即可：
```bash
unset HF_BASE_REPO   # 用默认 Manojb/stable-diffusion-2-1-base
bash hypir/01_download_models.sh
```
若你换用的镜像确为 gated 仓库（HF 对未认证账号返回 "not found" 实为 401），则需：1) 在该仓库页面接受许可证；2) 建 read token；3) `HF_TOKEN=<token> bash hypir/01_download_models.sh`（脚本会自动把 token 透传给 `hf download` 和 SSL 兜底下载器）。

**6. 推理 找不到diffusion_pytorch_model.bin**
是HYPIR/HYPIR/enhancer/sd2.py里强制要求bin格式，但下载到模型是safetensor格式，修改use_safetensors=True即可。

**7. 推理 OOM（显存不足）**
降 `PATCH_SIZE`（512→256，并相应降 `STRIDE` 到 128），或降 `UPSCALE`。免费 T4 可跑默认 512 patch（见官方 colab）。仍紧张时设 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**8. 训练时 `open_clip` / `lpips` 下载权重失败**
- 判别器 `ImageOpenCLIPConvNext` 在初始化时下载 `convnext_xxlarge`（laion2b_s34b_b82k_augreg_soup，open_clip 走 HuggingFace）。`_env.sh` 的 CA bundle + `HF_HUB_DISABLE_XET` 通常能覆盖；仍失败时手动放到 open_clip 缓存或 `HF_DISABLE_SSL=1` 后重试。
- **首次成功下载后，设 `HF_HUB_OFFLINE=1` 强制走本地缓存**（写进 `proxy.env` 或运行命令前），避免每次训练反复下载、代理下反复失败。前提是 `~/.cache/huggingface/` 里已有 convnext 权重（先联网成功跑一次再开离线）。
- `lpips.LPIPS(net="vgg")` 从作者 URL 下载 VGG 权重（走 `torch.hub`）。代理下若失败：先 `bash hypir/setup_ca_bundle.sh`，或预先把 `vgg.pth` 放进 `~/.cache/torch/hub/checkpoints/`。

**9. 训练报 `assert image.height == self.out_size`**
你的 GT 不是 512×512 且 `crop_type=none`。要么用 `03_build_dataset.sh` 的 `CROP=1` 预切 512 patch，要么 `CROP_TYPE=random bash hypir/04_train.sh`。

**10. 训练多卡 `accelerate launch` 只用一卡**
没配 accelerate 多进程。先 `accelerate config`（选 multi-GPU），再 `N_TRAIN_GPU=8 bash hypir/04_train.sh`（脚本会加 `--num_processes`）。单卡可忽略。

**11. 训练报 `queue_size` 不能被 `batch_size` 整除（合成退化路径 04c/04 才有）**
官方 `sd2_train.yaml` 的 `RealESRGANBatchTransform` 有个训练池 `queue_size=256`，要求是 `batch_size`(每卡) 的倍数。`256%6≠0` → 报错。修法二选一：
- 用 `BATCH_SIZE=8`（`256%8=0`，常用命令 #5 即如此；显存够就这个）；
- 或保 `BATCH_SIZE=6`、改 `queue_size=252`(=6×42)：在 04c/04_train.sh 的 heredoc 加 `cfg.data_config.train.batch_transform.params.queue_size=252`。
> 注：配对路径 04b 用 `PairedFaceBatchTransform`（无 queue_size），不受此限制。

**12. 跑 `.sh` 报 `syntax error near unexpected token ('`（CRLF 行尾）**
脚本被 CRLF 行尾污染（Windows→服务器用 scp/rsync/zip/复制等非 git 方式同步时带过去）。bash 遇到每行末尾的 `\r` 把引号上下文搞坏，于是 `echo "...(Baidu manual step)..."` 里的 `(` 被当成未引用的子 shell 起始符 → 解析报错（常报在某条 echo 行，如 `03d` 第 105 行）。本仓 `.gitattributes` 强制 LF，但只有 `git checkout/pull` 才会在检出时落 LF，非 git 传输方式不会自动转。
```bash
file hypir/03d_build_beauty_dataset.sh           # 出现 "CRLF line terminators" 即中招
grep -c $'\r' hypir/03d_build_beauty_dataset.sh  # 非 0 即有 \r
# 修法（任选）：
sed -i 's/\r$//' hypir/03d_build_beauty_dataset.sh                    # 剥掉 \r
find hypir retouchformer -name '*.sh' -exec sed -i 's/\r$//' {} +    # 一次性修所有 .sh
dos2unix hypir/03d_build_beauty_dataset.sh                           # 有 dos2unix 的话
git checkout -- hypir/03d_build_beauty_dataset.sh                    # git 同步的：.gitattributes 还原 LF
```
`.py` 不受影响（Python 词法器能吃 CRLF），但想统一也可一并 `sed`。预防：用 `git pull` 同步（git 按 `.gitattributes` 落 LF）；用 scp/rsync 的话传完跑一下上面的 `sed`；或 Windows 上 `git config --global core.autocrlf false` 再提交/同步。

**13. 跑 03d 报 `OSError: image file is truncated (N bytes not processed)`**
源夹里有损坏/截断图（下载不完整、传输中断等）。`build_beauty_dataset.py` 已把整张图处理包进 try/except——遇损坏图会打印 `! failed (skipped): ...` 并删掉本图半成品（不中断、不产生半对），汇总行有 `skipped=N` 计数，故**直接忽略即可**，03d 会自动跳过继续。想一次性清掉源夹里的坏图（之后重跑就没的跳了）：
```bash
INPUT_DIR=../HYPIR/dataset/guojia_datas_20260708 \
  python hypir/scan_corrupt_images.py                 # 只列
INPUT_DIR=../HYPIR/dataset/guojia_datas_20260708 DELETE=1 \
  python hypir/scan_corrupt_images.py                 # 列 + 原地删
```
扫描器和 build 脚本用同一套 `Image.open().convert("RGB").load()` 判定，一致；故意没设 `LOAD_TRUNCATED_IMAGES`（要它抛、好检测坏图）。

> 通用：`proxy.env`（代理凭证）在仓内 gitignored，`~/.ca-bundle.crt` 在家目录，都不入库；切勿把凭证写进脚本。
