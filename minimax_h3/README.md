# MiniMax-H3 runner

在 Ubuntu + NVIDIA 服务器上复现 [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)（MiniMax 的全模态视频+音频生成系统）。本目录只含编排脚本——官方代码自动 clone、权重从 HuggingFace 下载、推理走 **SGLang Diffusion**（官方推荐的三种框架之一）。

> ⚠️ **能本地复现的只有 H3-Base（768p）**。完整 H3 系统三模块：
> - **H3-Context-IR**（多模态指令预处理）——非开源，只能调 MiniMax API；
> - **H3-Base**（生成 768p 视频+原生立体声）——✅ 开源，本目录复现的就是它；
> - **H3-Regenerate-2K**（把 768p 重生成到 2K）——非开源，只能调 API。
>
> 即：本目录跑出来的是 **768p / 24fps / 立体声** 的视频。要 2K 需走「Full 2K Workflow」调 API（见文末）。

## 常用命令

> 假设已进入容器、`conda activate minimax_h3`、`cd media_code`。首次跑前先做下方「首次准备」。
> 用 diffusers 直接跑，**无需起服务**。要多卡并行更快见下方「Serving」段。

```bash
# ── H3-Context-IR API 转长 prompt（短 prompt -> API -> 长描述，打印 + 写文件，不自动跑 06）──
# H3-Context-IR 是 MiniMax 的 hosted API（非开源），把短 prompt + 可选首帧图转成
# 结构化长描述（官方推荐，效果比短 prompt 好）。需要 MiniMax API token。
# ⚠️ FIRST_FRAME 必须是 http URL（API 不读本地路径；本地图先上传公网）
# 脚本打印长描述 + 写 /tmp/h3_context_ir_prompt.txt + 给建议的 06 命令，你手动跑 06
MINIMAX_API_KEY=xxx \
PROMPT="视频中的人物保持绝对静止，一动不动，相机围绕画面中心水平旋转一圈 360°" \
FIRST_FRAME=https://raw.githubusercontent.com/CrescentVelvet/media_code/main/minimax_h3/examples/681533632532078.jpg \
FIRST_FRAME=https://raw.githubusercontent.com/CrescentVelvet/media_code/main/minimax_h3/examples/687115479202079.jpg \
bash minimax_h3/08_context_ir.sh

# ── int8 量化推理（06a，单卡常驻，比 06 两卡 offload 快）──
# int8 weight-only 量化：transformer+text_encoder 各 ~31GB，单卡 80GB 常驻
# 关键模块不量化（效果基本无损），需要 torchao（06a 自动装）
GPU=0 \
MODEL_PATH=../../model/MiniMax-H3 \
DEVICE=cuda:0 \
MAX_PIXELS=133120 \
FPS=24 \
NUM_FRAMES=124 \
SEED=42 \
TASK=fl2va \
FIRST_FRAME=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_3a8b3cc746304f49b9e3275e36aa9374/image/01000000.jpg \
PROMPT=$'integrated_multimodal_description: [Shot 1] Cinematic medium-wide shot. The subject shown in the first frame stands perfectly centered in frame, stock-still and frozen in place — absolutely no body movement, no sway, locked in a static pose. The camera is mounted rigidly on a perfectly horizontal circular ring track at fixed height, gliding along the track in a smooth, perfectly level, constant-speed 360-degree orbit around the subject. The camera moves purely horizontally — zero vertical bob, zero pitch, zero tilt, zero wobble — as if on a precision-engineered dolly on a level rail. As the camera travels, the front, three-quarter, side, back, and opposite three-quarter views of the stationary subject are revealed in sequence, each surface razor-sharp and evenly lit. The background slides laterally behind the subject in a steady, continuous parallax that confirms the perfectly circular, perfectly level path. [Shot 2] At 00:06.000, the camera completes the full 360-degree revolution and stops exactly at the starting front-facing angle, the subject still perfectly frozen in its original pose.\noverall_soundscape: A near-silent, steady room tone with only a faint, constant ambient hum. No footsteps, no movement sounds, no handling noise — the camera glides silently as if motorized on a track.\nnon_diegetic_music: A single sustained ambient synth drone, unchanging and continuous throughout, providing a calm, motionless backdrop that mirrors the perfectly steady rotation.' \
OUTPUT_DIR=../../output/minimaxh3_rotate_results/results_int8 \
OUTPUT_NAME=rotate_360.mp4 \
bash minimax_h3/06a_diffusers_inference.sh

# ── FlashVSR 4× 视频超分（07，把 06a int8 低分辨率输出超分到高清）──
# FlashVSR（CVPR 2026 one-step diffusion VSR）把 06a 的低分辨率（如 MAX_PIXELS=133120
# 出来 ~486×273）4× 超分到 ~1080p，并把 MiniMax 原生立体声 mux 回 SR 视频。
# 用 minimax_h3 env（复用，不建独立 env）；首次需 INSTALL_DEPS=1 装 Block-Sparse-Attention + diffsynth。
GPU=0 \
MUX_AUDIO=0 \
INPUT=../../output/minimaxh3_rotate_results/results_int8/rotate_360.mp4 \
FLASHVSR_MODEL_DIR=../../model/FlashVSR \
RESULTS_DIR=../../output/minimaxh3_rotate_results/results_sr \
OUTPUT_NAME=rotate_360_sr.mp4 \
bash minimax_h3/07_flashvsr_sr.sh

# ── 360° 旋转视频（H3-Context-IR 格式长 prompt，效果比短 prompt 好）──
# 图生视频（两卡分拆：text_encoder 放 cuda:1，rest（transformer/vae）放 cuda:0）（传入一张图作首帧，绕主体旋转一圈）
GPU=0,1 \
MODEL_PATH=../../model/MiniMax-H3 \
TRANSFORMER_DEVICE=cuda:0 \
TEXT_ENCODER_DEVICE=cuda:1 \
MAX_PIXELS=393216 \
FPS=24 \
NUM_FRAMES=124 \
SEED=42 \
TASK=fl2va \
FIRST_FRAME=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_3a8b3cc746304f49b9e3275e36aa9374/image/01000000.jpg \
PROMPT="视频中的人物保持绝对静止，一动不动，相机围绕画面中心水平旋转一圈 360°" \
OUTPUT_DIR=../../output/minimaxh3_rotate_results/results \
OUTPUT_NAME=rotate_360.mp4 \
bash minimax_h3/06_diffusers_inference.sh

# 图生视频（优化prompt）
GPU=0,1 \
MODEL_PATH=../../model/MiniMax-H3 \
TRANSFORMER_DEVICE=cuda:0 \
TEXT_ENCODER_DEVICE=cuda:1 \
MAX_PIXELS=393216 \
FPS=24 \
NUM_FRAMES=124 \
SEED=42 \
FIRST_FRAME=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_3a8b3cc746304f49b9e3275e36aa9374/image/01000000.jpg \
PROMPT=$'integrated_multimodal_description: [Shot 1] Cinematic medium-wide shot. The subject shown in the first frame stands perfectly centered in frame, stock-still and frozen in place — absolutely no body movement, no sway, locked in a static pose. The camera is mounted rigidly on a perfectly horizontal circular ring track at fixed height, gliding along the track in a smooth, perfectly level, constant-speed 360-degree orbit around the subject. The camera moves purely horizontally — zero vertical bob, zero pitch, zero tilt, zero wobble — as if on a precision-engineered dolly on a level rail. As the camera travels, the front, three-quarter, side, back, and opposite three-quarter views of the stationary subject are revealed in sequence, each surface razor-sharp and evenly lit. The background slides laterally behind the subject in a steady, continuous parallax that confirms the perfectly circular, perfectly level path. [Shot 2] At 00:06.000, the camera completes the full 360-degree revolution and stops exactly at the starting front-facing angle, the subject still perfectly frozen in its original pose.\noverall_soundscape: A near-silent, steady room tone with only a faint, constant ambient hum. No footsteps, no movement sounds, no handling noise — the camera glides silently as if motorized on a track.\nnon_diegetic_music: A single sustained ambient synth drone, unchanging and continuous throughout, providing a calm, motionless backdrop that mirrors the perfectly steady rotation.' \
OUTPUT_DIR=../../output/minimaxh3_rotate_results/results \
OUTPUT_NAME=rotate_360.mp4 \
bash minimax_h3/06_diffusers_inference.sh
```


- 结果：视频 → `OUTPUT_DIR/<name>.mp4`（768p 24fps 含原生立体声）。
- 默认 124 帧（~5s@24fps）；改帧数 `NUM_FRAMES=124`（需满足 17*n+5），换种子 `SEED=42`。
- 自定义 prompt 文件：`PROMPT_FILE=/your/prompt.txt bash ...`（H3-Context-IR 格式长描述效果最好）。
- SR：默认 4× full pipeline（最高质量）；低显存 / 长视频用 `PIPELINE=tiny_long`（需 `TCDecoder.ckpt`）。

## 首次准备
```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env      # 填 http_proxy / https_proxy（公司代理用）
conda create -n minimax_h3 --clone doll -y && conda activate minimax_h3
# 克隆现有 doll env（本地复制不走 conda 通道，绕开 TUNA 镜像 403 + 代理 SSL；
# sglang 支持 py3.10/3.11，doll 是哪个都能用）。
# doll 不在/想新建（会走 conda 通道，公司代理下易 403/SSL，修法见下方「可能遇到的问题」第 2 条）：
#   conda create -n minimax_h3 python=3.11 -y
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 先装 CUDA torch
INSTALL_DEPS=1 bash minimax_h3/00_setup_env.sh       # 装 sglang[all]（含 diffusion 支持）
# 下 MiniMaxAI/MiniMax-H3 权重快照（默认只下 FL2VA/；要 Ref2VA 加 DOWNLOAD_REF2VA=1）
HF_DISABLE_SSL=1 MODEL_DIR=../../model/MiniMax-H3 \
  bash minimax_h3/01_download_models.sh
```
⚠️ SGLang 自带 torch/flashinfer/cuda kernel，版本 pin 与本仓其他算法冲突，务必用专用 env（`CONDA_ENV=minimax_h3`），别装进共享 env。
⚠️ MiniMax-H3 在 HF 上是 **MiniMax H3 Community License**，可能 gated：下不动/报 401 时去 https://huggingface.co/MiniMaxAI/MiniMax-H3 接受协议、建 read token，再 `HF_TOKEN=<token> bash minimax_h3/01_download_models.sh`。

### FlashVSR 依赖（07 复用 minimax_h3 env，不建独立 env）
07 用 [OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR)，**直接复用 minimax_h3 env**（不 pin torch/transformers/numpy 版本——env 的 cu124 torch + transformers 5.x 能跑：VSR 运行时加载预计算 prompt tensor，不实例化 transformers 模型；diffsynth prompters 只用 `AutoTokenizer`）。首次准备：
```bash
# 已 conda activate minimax_h3（同 06/06a）
INSTALL_DEPS=1 bash minimax_h3/07_flashvsr_sr.sh    # clone FlashVSR + BSA 仓 + 装依赖（编译 BSA CUDA kernel 要 nvcc + 足够内存）
# 权重（v1.1 推荐）下到 model/FlashVSR/：
#   https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1
#   diffusion_pytorch_model_streaming_dmd.safetensors / Wan2.1_VAE.pth / LQ_proj_in.ckpt / TCDecoder.ckpt
```
> ⚠️ Block-Sparse-Attention 编译吃内存（多 ninja 并发易 OOM）；A100/A800（Ampere）加速最佳，H200 可跑但加速受限，其他卡（RTX 40/50、H800）兼容性未知（见 [BSA README](https://github.com/mit-han-lab/Block-Sparse-Attention)）。
> 权重放 `model/FlashVSR/`（`FLASHVSR_MODEL_DIR` 默认指向它）；脚本自动 clone 官方仓到 `../FlashVSR`、BSA 仓到 `../Block-Sparse-Attention`。
> 若 `from diffsynth import ModelManager` 在本 env 的 transformers 上 import 失败（5.x 改了 `PreTrainedConfig`/移除 `HybridCache`，理论上 diffsynth 没用到，但不保险），兜底建独立 env：`conda create -n flashvsr python=3.11 -y && conda activate flashvsr && pip install torch --index-url https://download.pytorch.org/whl/cu124 && CONDA_ENV=flashvsr INSTALL_DEPS_USE_PINS=1 bash minimax_h3/07_flashvsr_sr.sh`（`INSTALL_DEPS_USE_PINS=1` 才装 FlashVSR 官方 requirements 的 torch 2.6/transformers 4.46.2 pin）。

---

以下为详细参考（系统原理 / 各脚本参数 / 排错 / 目录布局）。

## 系统原理（H3-Base 在干什么）
MiniMax-H3 是全模态生成系统：吃 **文本 + 可选图片/视频/音频参考**，吐 **视频 + 原生立体声音频**（768p / 24fps / 32kHz stereo / 4–15s）。本地能跑的 H3-Base 架构：

1. **H3-Encoder**——用 Qwen3-VL-32B 全量预训练权重，取第 50 层 hidden state 喂给 Transformer。文本/图像都经它编码（音频不经）。
2. **H3-VisualVAE**——时间因果视频自编码器，空间压缩 16×、时间 4×、24 通道（f16t4d24）；潜变量再 patchify 成 `1×2×2`（有效空间下采样 32×、时间 4×）。解码器是 ViT-based。
3. **H3-AudioVAE**——左右声道各自编/解码，32kHz → 40Hz 潜序列（立体声）。
4. **H3-Omni-Transformer**——33B 单流 dense Transformer（~13B 在 AdaLN 分支，推理可预计算缓存故不用加载）。无模态专属结构（模态参数只在输入/输出/AdaLN）；用 3D MM-RoPE 联合表示 `(t,h,w)`。
5. 两个任务专用 checkpoint：**FL2VA**（首末帧 / 纯文）、**Ref2VA**（多模态参考）。都是 CFG-distilled 权重。

> 一句话：参考素材 →(各自 VAE/Encoder 编码)→ 拼成统一 packed 多模态序列 →(MM-RoPE)→ Omni-Transformer 联合预测视频+音频潜变量 →(各自 VAE 解码)→ 视频 + 立体声。

## Pipeline（推理流程详解）
对应 SGLang Diffusion server。一次生成从请求到 mp4 走 4 步：

1. **提交任务**——`POST /v1/videos`，body 含 `task`(t2va/fl2va/ref2va)、`prompt`(H3-Context-IR 格式的长描述)、`conditions`(参考素材列表)、`target`(short_edge/aspect_ratio/duration_seconds)、`seed` + 采样参数。服务返回 `{"id":"<video_id>"}`。
2. **编码参考**——服务把 `conditions` 里的图/视频/音频各自过 VAE/Encoder 编进潜空间，和文本拼成 packed 序列。
   - **FL2VA**：`conditions` 是首帧(`role=keyframe, frame_index=0`)和/或末帧(`frame_index=-1`)；无图即纯 T2VA。
   - **Ref2VA**：`conditions` 是参考素材，`role=reference`，按模态独立编号(`<Picture N>`/`<Audio N>`/`<Video N>`)。注意视频参考自带音轨（其音轨算 `<Audio 1>`，独立音频文件算 `<Audio 2>`，依此类推）。
3. **扩散去噪**——Omni-Transformer 以 packed 序列为条件，联合预测视频潜变量 + 音频潜变量（CFG-distilled，默认 50 步，`flow_shift=12` / `audio_flow_shift=3`）。
4. **解码 + 取片**——视频潜变量过 VisualVAE 解码、音频潜变量过 AudioVAE 解码（立体声），mux 成 mp4。`GET /v1/videos/<id>` 轮询 `status`；`completed` 后 `GET /v1/videos/<id>/content` 下载二进制 mp4。

### Prompt 长什么样（为什么这么长）
官方 768p 样例的 prompt 是 **H3-Context-IR 的输出格式**（因为 H3-Context-IR 非开源，官方把它的输出直接给出来让你照抄复现）。结构：
- `integrated_multimodal_description:` 逐镜头的画面+运镜+动作描述（带时间码）；
- `overall_soundscape:` 环境音/拟音描述；
- `non_diegetic_music:` 配乐描述；
- Ref2VA 还有 `subject_definitions:` / `summary:` / `retention_analysis:` / `detailed_description:`（参考素材如何用、保留哪些）。

> 想自己写短 prompt 也能跑（`03_generate.sh` 的 `PROMPT="..."`），但效果不如 H3-Context-IR 格式的长描述——官方建议接 Context-IR API 或照「Prompting Guidance」自建预处理。本目录的三个样例直接用官方给的 Context-IR 输出，可严格复现。

## Serving (02 — SGLang 起 H3-Base 服务)
`02_serve.sh` 调 `sglang serve`：把 33B Transformer + Qwen3-VL-32B 编码器分片到多卡，起一个 HTTP 服务。并行度全可配，下表是 SGLang cookbook 的 **verified 配方**映射到 A100：

| 硬件 | 配方 | 命令 |
|---|---|---|
| **4× A100 80GB**（8 卡选 4 卡） | 容量（最稳，推荐先试） | `GPU=0,1,2,3 NUM_GPUS=4 ULYSSES_DEGREE=4 USE_FSDP=1 MODEL_PATH=../../model/MiniMax-H3 bash minimax_h3/02_serve.sh` |
| 4× A100 80GB | 最快（TP2 + Ulysses2） | `GPU=0,1,2,3 NUM_GPUS=4 TP_SIZE=2 ULYSSES_DEGREE=2 MODEL_PATH=../../model/MiniMax-H3 bash ...` |
| 2× RTX 5090 32GB | 逐层 offload（慢） | `GPU=0,1 EXTRA_SGLANG_FLAGS="--performance-mode memory --layerwise-offload-components dit,text_encoder,vae --dit-layerwise-resident-layers 20" bash ...` |

- A100 是 Ampere，不在官方 verified 列表，但 4× A100 80GB 单卡显存 80GB 足够覆盖 80GB 配方。
- 默认 `NUM_GPUS=4 ULYSSES_DEGREE=4`（resident）在 80GB 卡上可能 OOM——A100 务必备好 `USE_FSDP=1` 或 `TP_SIZE=2 ULYSSES_DEGREE=2` 兜底。
- `MODEL_VARIANT=fl2va`(默认, :30010) / `ref2va`(:30011)；`PORT` 可覆盖。
- `BG=1` 后台起 + 轮询 `/health` 等就绪（加载 33B 要几分钟）；`BG=0`(默认) 前台跑看日志。
- `GPU=0,1,2,3` 选物理卡号（8 卡服务器选其他 4 张就改这里，`NUM_GPUS` 必须和卡数一致）。
- `MODEL_PATH` 指向 HF 权重快照根目录（含 `model_index.json`）。
- `EXTRA_SGLANG_FLAGS` 透传任意 SGLang 参数（如 `--quantization fp8`、offload 选项）。

## Generate (03 — 发请求 + 轮询 + 下载)
`03_generate.sh` 调 `generate.py`：根据 `TASK` 拼请求 body、提交、轮询、下载，打印耗时。任务→conditions 映射（每条都要显式带 `GPU` `SERVER_URL` 输入路径 `OUTPUT_DIR`/`OUTPUT_NAME`）：
```bash
# T2VA（无 conditions）
GPU=0,1,2,3 SERVER_URL=http://localhost:30010 \
  TASK=t2va PROMPT="..." DURATION=10 ASPECT_RATIO=16:9 SEED=0 \
  OUTPUT_DIR=../../output/minimaxh3_rotate_results/results/t2va OUTPUT_NAME=t2va.mp4 \
  bash minimax_h3/03_generate.sh
# I2VA 首帧（FL2VA 权重）
GPU=0,1,2,3 SERVER_URL=http://localhost:30010 \
  TASK=fl2va FIRST_FRAME=/data/first.png DURATION=8 \
  OUTPUT_DIR=../../output/minimaxh3_rotate_results/results/fl2va OUTPUT_NAME=fl2va.mp4 \
  bash minimax_h3/03_generate.sh
# FL2VA 首末帧
GPU=0,1,2,3 SERVER_URL=http://localhost:30010 \
  TASK=fl2va FIRST_FRAME=/data/first.png LAST_FRAME=/data/last.png DURATION=8 \
  OUTPUT_DIR=../../output/minimaxh3_rotate_results/results/fl2va OUTPUT_NAME=fl2va_fl.mp4 \
  bash minimax_h3/03_generate.sh
# Ref2VA 参考图+音频（Ref2VA 权重，服务在 :30011）
GPU=0,1,2,3 SERVER_URL=http://localhost:30011 \
  TASK=ref2va REF_IMAGES=/data/subject.png REF_AUDIOS=/data/voice.mp3 \
  PROMPT="Use <Picture 1> as the subject and <Audio 1> as the voice." \
  OUTPUT_DIR=../../output/minimaxh3_rotate_results/results/ref2va OUTPUT_NAME=ref2va.mp4 \
  bash minimax_h3/03_generate.sh
# Ref2VA 多参料（逗号分隔；视频可带 start_time_seconds）
GPU=0,1,2,3 SERVER_URL=http://localhost:30011 \
  TASK=ref2va REF_IMAGES=/data/a.png,/data/b.png REF_VIDEOS=/data/v1.mp4 REF_VIDEO_STARTS=0,0 \
  PROMPT="Combine <Picture 1>, <Picture 2>, <Video 1> ..." \
  OUTPUT_DIR=../../output/minimaxh3_rotate_results/results/ref2va OUTPUT_NAME=ref2va_multi.mp4 \
  bash minimax_h3/03_generate.sh
```
- 本地路径自动转 `file://`（服务进程要能读该路径）；http(s) URL 原样传（官方样例用 CDN URL）。
- Ref2VA 的 condition **顺序**会影响模态编号。官方 ref2va 样例顺序是「视频先、音频后」（视频自带音轨算 `<Audio 1>`、音频文件算 `<Audio 2>`），故用 `CONDITIONS_FILE=examples/ref2va_conditions.json` 传 verbatim JSON 数组，绕开 env 构造的固定顺序，保证严格复现。
- `NUM_INFERENCE_STEPS=50 FLOW_SHIFT=12.0 AUDIO_FLOW_SHIFT=3.0`（SGLang cookbook 默认）；`NUM_OUTPUTS=N` 每次出 N 段。
- `POLL_INTERVAL=10` `TIMEOUT_MINS=30` 控制轮询。

## 复现官方三个 768p 样例
`examples/` 下三个脚本各跑一个官方样例（prompt 直接抄自 `scripts/readme/reproducible-768p-*-request.sh`，参考素材用官方 CDN URL，输入地址在脚本内写死；调用时仍要显式传 `GPU` `SERVER_URL` `OUTPUT_DIR`）：
```bash
# 前置：FL2VA 服务先起好（见「常用命令」起服务 1)，端口 :30010
GPU=0,1,2,3 SERVER_URL=http://localhost:30010 \
  OUTPUT_DIR=../../output/minimaxh3_rotate_results/results/t2va \
  bash minimax_h3/examples/run_t2va.sh     # 文生视频，10s 16:9，星舰舰长 -> t2va.mp4
GPU=0,1,2,3 SERVER_URL=http://localhost:30010 \
  OUTPUT_DIR=../../output/minimaxh3_rotate_results/results/fl2va \
  bash minimax_h3/examples/run_fl2va.sh    # 首帧生视频，8s auto，拉面/家庭 -> fl2va.mp4
# 前置：Ref2VA 服务先起好（见「常用命令」起服务 2)，端口 :30011
GPU=0,1,2,3 SERVER_URL=http://localhost:30011 \
  OUTPUT_DIR=../../output/minimaxh3_rotate_results/results/ref2va \
  bash minimax_h3/examples/run_ref2va.sh   # 参考视频+音频，5s auto，粉西装男 -> ref2va.mp4
```
对照官方结果：仓库 `assets/t2va.mp4` / `fl2va.mp4` / `ref2va.mp4`（在 https://github.com/MiniMax-AI/MiniMax-H3 的 assets 下）。

## 生成 360° 旋转视频

参考 `wan22_rotate` 的思路（选正面图+分割 → Wan2.2+LoRA 旋转），但 **MiniMax-H3 不需要 LoRA**——prompt 直接写旋转指令即可（如 `"相机围绕主体水平旋转一圈 360°，主体保持静止"`）。

用 `06_diffusers_inference.sh`，prompt 写旋转描述，输入图可以是**原始拍摄图 / 人体分割白底图 / 任意主体图**（MiniMax-H3 能理解各种输入，不像 Wan2.2 需要 LoRA 训练 + 白底分割图）。命令见上方「常用命令」段。

- 默认 124 帧 / 24fps / 768p。`NUM_FRAMES=121` 改帧数（需满足 17*n+5），`SEED=42` 换种子。
- 与 `wan22_rotate` 的区别：MiniMax-H3 不用 LoRA/分割，prompt 即旋转指令；但旋转一致性不如专门训练的 LoRA（可能旋转中途主体形变）。要精确旋转接 `wan22_rotate/02`，要快速出片用 06。

## FlashVSR 视频超分 (07)
`07_flashvsr_sr.sh` 调 `07_flashvsr_sr.py`：把 **06a int8 量化出的低分辨率视频**（`MAX_PIXELS=133120` ~486×273 等）用 [FlashVSR](https://github.com/OpenImagingLab/FlashVSR)（CVPR 2026，one-step diffusion streaming VSR）4× 超分到高清，并把 MiniMax 原生立体声 mux 回 SR 视频。

> ⚠️ FlashVSR 是为 **4× 超分** 设计的，官方强烈建议 4× 设置。输入 LR × 4 = 输出 SR（如 486×273 → 1944×1092 ≈ 1080p）。06a 用低 `MAX_PIXELS` 出低分辨率再 07 超分，比直接 06 出 768p 更快/省显存且画质可控。

### 原理（07 在干什么）
1. **读视频 + 补帧**：逐帧读 06a 输出，取首帧算 LR 尺寸 → 4× BICUBIC 上采样 → 中心裁剪到 128 倍数（FlashVSR VAE 要求）。模型要求帧数 `F=8n+1` 且末尾 ~4 帧无直接 LQ 输入（runoff），故**向上补到 `F=8n+1≥N+PAD_FRONT+PAD_BACK`**（后垫 ≥4 吸收 runoff，前垫给因果状态预热），生成后裁 `output[PAD_FRONT:PAD_FRONT+N]` 还原 N 帧。
   > ⚠️ **修复了官方 `infer_flashvsr_*.py` 的缺帧 bug**：官方向下截断到 8n+1 且不裁输出——`N≡2,3,4 mod 8` 时丢真帧（MiniMax 124 帧出 121 帧），`N≡5,6,7,0 mod 8` 时多出冻结尾帧。本脚本改为向上补 + 裁剪，**所有 N 都输出恰好 N 帧**。
2. **超分**：one-step diffusion（DMD 蒸馏，单步去噪）+ **Locality-Constrained Sparse Attention**（LCSA，靠 Block-Sparse-Attention kernel）剪冗余注意力；`topk_ratio = sparse_ratio * 768*1280 / (th*tw)` 随分辨率缩放保持算力恒定；可选 **tiny conditional decoder**（TCDecoder）省显存。`color_fix=True` 校正色彩偏移。
3. **拼音频**：SR 输出是纯视频（imageio），`MUX_AUDIO=1`（默认）用 ffmpeg 把原 LR 视频的音频流（MiniMax 原生立体声）mux 回 SR 视频。

### 两种管线
| `PIPELINE` | 类 | 额外权重 | 用途 |
|---|---|---|---|
| `full`（默认） | `FlashVSRFullPipeline` | `Wan2.1_VAE.pth` | 最高质量，短视频（MiniMax ~5s/124 帧）推荐 |
| `tiny_long` | `FlashVSRTinyLongPipeline` | `TCDecoder.ckpt` | 低显存 / 长视频流式，frame 留 CPU |

> 低显存还能 `TILED=1`（full 专用，分块省 VRAM 但更慢）。`SPARSE_RATIO=1.5` 更快、`2.0` 更稳；`LOCAL_RANGE=9` 更锐、`11` 更稳（官方推荐）。

### 命令
```bash
# 用 minimax_h3 env（conda activate minimax_h3，同 06/06a；见上方「FlashVSR 依赖」）
# 默认 full pipeline，4× 超分 06a int8 输出
GPU=0 \
INPUT=../../output/minimaxh3_rotate_results/results_int8/rotate_360.mp4 \
FLASHVSR_MODEL_DIR=../../model/FlashVSR \
RESULTS_DIR=../../output/minimaxh3_rotate_results/results_sr \
OUTPUT_NAME=rotate_360_sr.mp4 \
bash minimax_h3/07_flashvsr_sr.sh

# 低显存 / 长视频 -> tiny_long 流式管线
GPU=0 PIPELINE=tiny_long \
INPUT=../../output/minimaxh3_rotate_results/results_int8/rotate_360.mp4 \
FLASHVSR_MODEL_DIR=../../model/FlashVSR \
RESULTS_DIR=../../output/minimaxh3_rotate_results/results_sr \
bash minimax_h3/07_flashvsr_sr.sh

# 不保留音频（纯视频输出）
GPU=0 MUX_AUDIO=0 INPUT=../../output/minimaxh3_rotate_results/results_int8/rotate_360.mp4 \
FLASHVSR_MODEL_DIR=../../model/FlashVSR \
bash minimax_h3/07_flashvsr_sr.sh
```
- 输入也可是图片帧目录（按文件名自然排序，`.png/.jpg`），非视频时 fps 默认 30。
- 首次 `INSTALL_DEPS=1`：自动 clone FlashVSR 仓（`../FlashVSR`，含 `diffsynth` + `examples/WanVSR/utils`）+ BSA 仓（`../Block-Sparse-Attention`）+ 编译 BSA CUDA kernel + `pip install -e` FlashVSR（`--no-deps`，不拉 torch 2.6/transformers 4.46.2/numpy 1.26.4 pin 降级 env）+ 版本无关运行时依赖。
- 权重默认读 `model/FlashVSR/`（`FLASHVSR_MODEL_DIR` 覆盖）；full 检查 `diffusion_pytorch_model_streaming_dmd.safetensors` / `Wan2.1_VAE.pth` / `LQ_proj_in.ckpt`，tiny_long 把 `Wan2.1_VAE.pth` 换成 `TCDecoder.ckpt`。

## Full 2K Workflow（调 MiniMax API，非开源部分）
要 2K 输出需把本地 768p 结果喂回 MiniMax API 的 H3-Regenerate-2K，并用 H3-Context-IR API 预处理自由 prompt。本目录不含这部分（要 API Token），步骤：
1. 本地起 H3-Base 服务（本目录 02）；
2. 调 `POST /video-generation-v2-h3-context-ir`（MiniMax API）把你的自由 prompt 转成 H3-Context-IR 格式长描述；
3. 用长描述调本地服务出 768p；
4. 把 768p 调 `POST /video-generation-v2-regeneration`（API）重生成到 2K。
API 文档：Global `platform.minimax.io` / CN `platform.minimaxi.com`（`/video-generation-v2-create`、`-h3-context-ir`、`-regeneration`）。

## 可能遇到的问题

公司代理做 HTTPS 中间人解密，按阶段列常见报错与修法（命令在服务器上、conda 环境已激活时执行）。

**1. clone/pull 本仓或官方仓报错**
- `SSL certificate problem`：公开仓加 `-c http.sslVerify=false`。
- `Failed to connect to github.com port 443`（连不上，非 SSL）：git 没走代理。设全局代理（密码特殊字符必须 URL 编码）：
  ```bash
  git config --global https.proxy http://USER:PASS@proxyhk.huawei.com:8080
  git config --global http.proxy  http://USER:PASS@proxyhk.huawei.com:8080
  ```
- `No route to host` 连代理都不通：多为 docker 网桥网段和代理 IP 冲突。查 `getent hosts <proxy>` + `ip route | grep <网段>`，加主机路由 `sudo ip route add <代理IP> via <默认网关> dev <物理网卡>`，或改 `/etc/docker/daemon.json` 的 `default-address-pools` 给 docker 分不冲突的子网。

**2. `conda create` 报 `HTTP 403` / `SSL: CERTIFICATE_VERIFY_FAILED`**
两道坎：① `~/.condarc` 把 `custom_channels` / `default_channels` 指向清华 TUNA 镜像（`mirrors.tuna.tsinghua.edu.cn/anaconda/...`），该镜像偶发限流/抽风返 403；② 公司代理 TLS 拦截，conda 不信代理根 CA 报 SSL。`conda config --show channels` 只显逻辑通道名（`conda-forge` / `defaults`），镜像重定向藏在 `custom_channels` / `default_channels` 里，用 `conda config --show-sources` 才看得到。⚠️ 关键：conda **不读** `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` 环境变量（和 pip 不同！），SSL 必须走 `conda config --set ssl_verify`，导出环境变量对 conda 无效。三档修法（从简到繁）：
- **克隆现有 env（最省事，不走 conda 通道，无 403/SSL）**——本机已有 `doll`(python 3.11) 就直接克隆，sglang[all] 自带 torch/flashinfer，只借 doll 的 python 起壳：
  ```bash
  conda create -n minimax_h3 --clone doll -y && conda activate minimax_h3
  ```
- **新建 + 绕开镜像 + 用 CA 包验 SSL**（doll 不在时）——先建 CA 包，再让 conda 用它（不是导出环境变量，是 `ssl_verify`），最后 `--override-channels` 直连官方源：
  ```bash
  bash minimax_h3/setup_ca_bundle.sh                      # -> ~/.ca-bundle.crt（含代理根 CA，自检 [OK]）
  conda config --set ssl_verify "$HOME/.ca-bundle.crt"     # conda 不读 REQUESTS_CA_BUNDLE，必须设这条
  conda create -n minimax_h3 python=3.11 -y --override-channels \
    -c https://conda.anaconda.org/conda-forge -c https://repo.anaconda.com/pkgs/main
  ```
  官方源也被代理挡时，把两个 `-c` 换成 BFSU 同源镜像（`https://mirrors.bfsu.edu.cn/anaconda/cloud/conda-forge`、`https://mirrors.bfsu.edu.cn/anaconda/pkgs/main`）。建完想恢复默认：`conda config --set ssl_verify true`。
- **永久修 `~/.condarc`**（影响所有 conda 命令）——把 TUNA 域名（`mirrors.tuna.tsinghua.edu.cn/anaconda`）整体替换成 BFSU（`mirrors.bfsu.edu.cn/anaconda`，同源更稳），或直接删掉 `custom_channels` / `default_channels` 两段回退到 conda 官方源；SSL 一劳永逸设 `conda config --set ssl_verify "$HOME/.ca-bundle.crt"`。

**3. `pip install sglang[all]` 报 SSL/超时**
```bash
pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org download.pytorch.org"
INSTALL_DEPS=1 bash minimax_h3/00_setup_env.sh
```
torch 大文件超时：先单独 `pip install --timeout 600 --retries 10 torch --index-url https://download.pytorch.org/whl/cu124`，再 `INSTALL_DEPS=1`。
> flashinfer（SGLang 默认注意力后端依赖）走自建下载源，代理下易失败：先 `bash minimax_h3/setup_ca_bundle.sh` 建 CA 包；仍失败可 `pip install flashinfer -f https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python` 或换 `EXTRA_SGLANG_FLAGS="--attention-backend triton"`。

**4. `hf download` 报 `CAS service error : ReqwestMiddleware`**
HF 的 Xet/Rust 通道不认代理。`_env.sh` 已设 `HF_HUB_DISABLE_XET=1`；仍报就 `pip uninstall -y hf_xet` 后重跑 `01`。

**5. `hf download` 报 `SSLCertVerificationError`**
代理根 CA 不在系统证书包。先 `bash minimax_h3/setup_ca_bundle.sh`（抓代理证书链→`~/.ca-bundle.crt` 并自检）；自检 `[OK]` 重跑 `01`，`[FAIL]` 把公司根 CA 追加到 `~/.ca-bundle.crt`；仍不行 `HF_DISABLE_SSL=1 bash minimax_h3/01_download_models.sh`（走 SSL 免校验兜底下载器 `_hf_download.py`）。

**6. `hf download` 报 401 / `repository not found`（MiniMax-H3 gated）**
HF 上 MiniMax-H3 是 Community License，可能需接受协议。去 https://huggingface.co/MiniMaxAI/MiniMax-H3 点 Accept，建 read token，再 `HF_TOKEN=<token> bash minimax_h3/01_download_models.sh`（脚本会透传给 `hf download` 和兜底下载器）。

**7. serve 报 OOM / CUDA out of memory（4× A100 80GB）**
默认 `NUM_GPUS=4 ULYSSES_DEGREE=4`（resident）在 80GB 卡上可能不够。按顺序试：
```bash
# a) FSDP 容量配方（A100 80GB verified）
GPU=0,1,2,3 NUM_GPUS=4 ULYSSES_DEGREE=4 USE_FSDP=1 \
  MODEL_PATH=../../model/MiniMax-H3 bash minimax_h3/02_serve.sh
# b) TP2 + Ulysses2（降单卡峰值显存）
GPU=0,1,2,3 NUM_GPUS=4 TP_SIZE=2 ULYSSES_DEGREE=2 \
  MODEL_PATH=../../model/MiniMax-H3 bash minimax_h3/02_serve.sh
# c) offload（极慢但能塞下，2× RTX5090 才需要）
GPU=0,1 NUM_GPUS=2 MODEL_PATH=../../model/MiniMax-H3 \
  EXTRA_SGLANG_FLAGS="--performance-mode memory --layerwise-offload-components dit,text_encoder,vae --dit-layerwise-resident-layers 20" \
  bash minimax_h3/02_serve.sh
# 还可加 export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**8. serve 起不来：`sglang: command not found` / `import sglang` 失败**
没装 SGLang：`INSTALL_DEPS=1 bash minimax_h3/00_setup_env.sh`。装了还是 `command not found` 多半没 `conda activate minimax_h3`（脚本默认沿用当前 env）。`[diffusion]` extra 没带上时（模型类找不到）补 `pip install "sglang[diffusion]"`。
> 若报 `unrecognized arguments --model-variant fl2va --performance-mode speed`：SGLang Diffusion（`--model-variant` / `--performance-mode` 等参数）在 PyPI 的 `sglang[all]`（如 0.5.10.post1）不带——这些参数在 git main 分支。`00_setup_env.sh` 已自动处理：先 `pip install -U "sglang[all]"`，自检 `--model-variant` 失败则 `git clone` sglang 源码 + `pip install -e ... --no-deps --config-settings editable_mode=compat`（跳过 git pin 的 `torch==2.13.0`/`cuda-python>=13.0`，用 env 现有 torch cu124 装 sglang 本体；`compat` 模式避免 `KeyError: sglang.multimodal_gen`）。重跑一次：
> ```bash
> INSTALL_DEPS=1 bash minimax_h3/00_setup_env.sh
> ```
> 末尾看是否 `✅ sglang serve now supports --model-variant`。仍失败手动：`LD_LIBRARY_PATH= git clone --depth 1 https://github.com/sgl-project/sglang.git /tmp/sglang-src && SGLANG_BUILD_RUST_EXTS=none pip install -e "/tmp/sglang-src/python[diffusion]" --no-deps --config-settings editable_mode=compat`。
> 若 editable 安装报 `cargo is required to discover the Rust extension modules`：设 `SGLANG_BUILD_RUST_EXTS=none` 跳过 Rust 扩展（00 已自动设）；运行时若报 Rust 相关错误再装工具链：`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source $HOME/.cargo/env`。
> **依赖对齐**（sglang git main pin `transformers==5.12.1`）：00 自动 `-U diffusers peft "transformers==5.12.1" huggingface_hub xgrammar`。注意：
> - transformers **必须 pin `==5.12.1`**：不带版本 pip 会升到 5.15+，5.15 内置 `qwen3_asr` 与 sglang 注册冲突；4.x 缺 `PreTrainedConfig`（sglang 用了 5.x 新名 `PretrainedConfig`）。
> - `HybridCache` 在 5.x 已移除（用 `DynamicCache`），sglang 不依赖它，00 已去掉自检。
> - `huggingface_hub` 旧版缺 `is_offline_mode`，与 transformers 5.x 不兼容，必须一起升。
> - `xgrammar>=0.2.1`（sglang git main 用了 `AnyTokensFormat` 等新 API）。
> - **qwen3_asr 热修**：sglang git main 的 `sglang/srt/configs/qwen3_asr.py` 用 `AutoConfig.register("qwen3_asr", ...)` 没传 `exist_ok=True`，transformers 5.12+ 内置同名配置会报 `ValueError: 'qwen3_asr' is already used`。00 在 editable install 后自动用 python 脚本给两处 register 加 `try/except + exist_ok=True`（检测 `exist_ok` 已在文件里则跳过）。手动修：`python -c "import pathlib; p=pathlib.Path('/tmp/sglang-src/python/sglang/srt/configs/qwen3_asr.py'); s=p.read_text(); s=s.replace('AutoConfig.register(\"qwen3_asr\", Qwen3ASRConfig)', 'try:\n    AutoConfig.register(\"qwen3_asr\", Qwen3ASRConfig, exist_ok=True)\nexcept ValueError:\n    pass'); p.write_text(s)"`
> - **`requires ... which is not installed` 警告正常**：用 `--no-deps` 装 sglang git main，它 pin 的是 `torch==2.13.0`/`cuda-python>=13.0`（CUDA 13）等依赖，而当前 env 是 torch 2.x cu124（CUDA 12.4）。pip 会打印 `sglang 0.x requires torch==2.13.0, but you have torch 2.x which does not match` 之类的警告——**这是预期的，不影响 MiniMax-H3 推理**。sglang 的 diffusion serving 代码不依赖 torch 2.13 新特性，cu124 的 torch 能正常跑。
> - **g++ 太旧（C++20 JIT fallback）**：sglang JIT 融合 QKNorm+RoPE kernel 要 C++20 `<concepts>`，系统 g++ 9 不支持会 fallback 到非融合 Python 实现（功能正确但略慢）。00 在 INSTALL_DEPS 开头自动检测并 `conda install gxx_linux-64=12 python=3.10`（可选，失败不中断）。**02_serve.sh 启动 sglang serve 前会自动检测并设 `CXX`/`CC` 指向高版本 g++**（conda 的优先，其次 `g++-12`/`g++-11`），让 JIT 编译用高版本 g++。升级后日志里 JIT kernel 编译成功；不升级也能跑。
> - **JIT 编译失败导致 worker 僵死（任务永远 queue）**：如果 `ninja exited with status 1`（`<concepts>` 找不到 / `-std=c++20` 不识别），sglang worker 会僵死，任务停在 `status=queue` 不处理。根因是 CXX 没指向高版本 g++。修法：确认 `INSTALL_DEPS=1 bash minimax_h3/00_setup_env.sh` 装了 `gxx_linux-64=12`，重启服务看日志是否 `🎮 CXX=... (C++20 JIT ok)`。g++ 升级后旧失败产物可能残留，设 `CLEAR_JIT_CACHE=1 bash minimax_h3/02_serve.sh` 清理 `~/.cache/tvm-ffi/sgl_kernel_jit_*` 后重启。
> - **`only_qv` 参数不兼容旧 kernel**：`sglang-kernel==0.4.1`（A100/sm80 版本）的 `flash_attn_varlen_func` 不支持 `only_qv` 参数，sglang git main 代码传了它。00 在 editable install 后自动热修 `sglang/kernels/ops/attention/flash_attention_v3.py`，用 `inspect.signature` 动态过滤不支持的参数（检测 `inspect.signature` 已在则跳过，幂等）。

**9. `02_serve.sh` 后台模式一直 `not ready`**
看 `../MiniMax-H3/logs/serve_<variant>_<port>.log` 末尾：常见是权重路径错（`model_index.json` 缺 → 重跑 `01`）、CUDA/torch 不匹配、或多卡初始化卡住。脚本会在进程死掉时自动 `tail -n 40` 报错。健康检查超时可 `HEALTH_TIMEOUT_MINS=60 bash minimax_h3/02_serve.sh` 放宽。

**10. generate 报 `connection refused` / `submit failed HTTP 5xx`**
服务没起好或挂在别的端口。`curl -s http://localhost:30010/health` 验证；FL2VA 用 30010、Ref2VA 用 30011，`SERVER_URL` 要对上。5xx 多为请求体不合法（task/conditions 拼错）——看服务日志里的 traceback。

**11. Ref2VA 用本地参料，`file://` 路径服务读不到**
本地直装 SGLang 时 `file://` 用服务进程能读的绝对路径即可；若 SGLang 跑在 **docker** 里，容器看不到宿主机路径——需 `-v /data/minimax-h3:/data/minimax-h3:ro` 挂卷，参考素材放挂载目录下并用 `file:///data/minimax-h3/xxx`。最省事：把参料传到公网 URL，用 http URL。

**12. 跑 `.sh` 报 `syntax error near unexpected token ('`（CRLF 行尾）**
Windows→服务器用 scp/zip 等非 git 方式同步带过去。本仓 `.gitattributes` 强制 LF，但只有 `git checkout/pull` 才落 LF，非 git 传输不会转。
```bash
file minimax_h3/02_serve.sh           # 出现 "CRLF line terminators" 即中招
find minimax_h3 -name '*.sh' -exec sed -i 's/\r$//' {} +    # 一次性修所有 .sh
# 或 git checkout -- minimax_h3/
```

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | 当前 env | 专用 `minimax_h3` 推荐（SGLang pin 与其他算法冲突） |
| `GPU` | _(必填)_ | 物理卡号列表（`0,1,2,3`），8 卡选 4 卡用前 4 张；`NUM_GPUS` 要和卡数一致 |
| `MINIMAX_H3_DIR` | `../MiniMax-H3` | GitHub 参考仓（scripts/skills，serve 不依赖它） |
| `MODEL_DIR` / `MODEL_PATH` | `../../model/MiniMax-H3` | HF 权重快照（SGLang `--model-path` 指它） |
| `MINIMAX_H3_REPO` | 官方 GitHub URL | clone 源 |
| `INSTALL_DEPS` | `0` | `1` = `pip install "sglang[all]"`（首次准备手动设 `1`） |
| `HF_HUB_DISABLE_XET` | `1` | 关 HF Xet/Rust 通道（代理不友好） |
| `HF_DISABLE_SSL` | `0` | `1` = SSL 免校验下载权重 |
| `HF_TOKEN` | _(unset)_ | gated 仓才需（先去 HF 页面接受协议） |
| `DOWNLOAD_REF2VA` | `0` | `1` = 额外下 Ref2VA/（默认只下 FL2VA/） |

### Serve (02)
| var | default | note |
|---|---|---|
| `MODEL_VARIANT` | `fl2va` | `fl2va` \| `ref2va` |
| `HOST` / `PORT` | `0.0.0.0` / `30010`(fl2va) `30011`(ref2va) | |
| `NUM_GPUS` | `4` | 张数 |
| `ULYSSES_DEGREE` | `=NUM_GPUS` | Ulysses 注意力并行度 |
| `TP_SIZE` | _(unset)_ | 张量并行；A100 80GB 最快配 `2` |
| `USE_FSDP` | `0` | `1` = `--use-fsdp-inference true`（80GB 卡容量配方） |
| `PERFORMANCE_MODE` | `speed` | `speed` \| `memory`（offload 用 memory） |
| `EXTRA_SGLANG_FLAGS` | _(unset)_ | 透传任意 `sglang serve` 参数 |
| `BG` | `0` | `1` = 后台起 + 轮询 `/health` 等就绪 |
| `HEALTH_TIMEOUT_MINS` | `30` | 后台模式等就绪超时 |
| `LOG_DIR` / `LOG_FILE` | `../MiniMax-H3/logs/...` | 服务日志 |

### Generate (03)
| var | default | note |
|---|---|---|
| `SERVER_URL` | `http://localhost:$PORT` | 也可 `host:port`；Ref2VA 改 `:30011` |
| `TASK` | `t2va` | `t2va` \| `fl2va` \| `ref2va` |
| `PROMPT` / `PROMPT_FILE` | _(t2va 内置默认)_ | inline 优先；文件次之 |
| `NUM_FRAMES` | `124` | 帧数（需满足 17*n+5，124=17*4+5 ~3s@24fps） |
| `ASPECT_RATIO` | t2va=`16:9` / fl2va·ref2va=`auto` | `16:9` `9:16` `1:1` `auto` |
| `SHORT_EDGE` | `768` | 短边像素 |
| `SEED` | `0` | |
| `FIRST_FRAME` / `LAST_FRAME` | _(unset)_ | FL2VA 首末帧（路径或 URL） |
| `REF_IMAGES` / `REF_AUDIOS` / `REF_VIDEOS` | _(unset)_ | Ref2VA 参料（逗号分隔列表） |
| `REF_VIDEO_STARTS` | _(unset)_ | 视频起始秒（逗号分隔，对应 REF_VIDEOS） |
| `VIDEO_AS_AUDIO_REF` | `0` | `1` = 视频参料带音轨（`type=video_audio`） |
| `CONDITIONS_FILE` | _(unset)_ | JSON 数组，verbatim 覆盖 REF_* 构造（复现官方顺序用） |
| `NUM_INFERENCE_STEPS` / `FLOW_SHIFT` / `AUDIO_FLOW_SHIFT` | `50` / `12.0` / `3.0` | 采样参数（cookbook 默认） |
| `NUM_OUTPUTS` | `1` | 每次出几段 |
| `OUTPUT_DIR` / `OUTPUT_NAME` | `../../output/minimaxh3_rotate_results/results/<task>` / `<task>_seed<seed>.mp4` | |
| `POLL_INTERVAL` / `TIMEOUT_MINS` | `10` / `30` | 轮询间隔 / 超时 |

### FlashVSR SR (07)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | 当前 env（`minimax_h3`） | 复用 minimax_h3 env（不建独立 env）；兜底独立 env 见「FlashVSR 依赖」 |
| `INSTALL_DEPS_USE_PINS` | `0` | `1`=装 FlashVSR 官方 requirements.txt pin（torch 2.6/transformers 4.46.2/numpy 1.26.4），仅独立 env 兜底用 |
| `INPUT` | _(必填)_ | 输入 LR 视频路径或图片帧目录 |
| `FLASHVSR_MODEL_DIR` | `../../model/FlashVSR` | 权重目录（v1 / v1.1 通用） |
| `FLASHVSR_DIR` | `../FlashVSR` | 官方仓（diffsynth + `examples/WanVSR/utils`） |
| `BSA_DIR` | `../Block-Sparse-Attention` | Block-Sparse-Attention 源码（LCSA 必需） |
| `PIPELINE` | `full` | `full`（最高质量，需 `Wan2.1_VAE.pth`） \| `tiny_long`（低显存/长视频，需 `TCDecoder.ckpt`） |
| `SCALE` | `4` | 超分倍数（官方推荐 4×） |
| `SEED` | `0` | |
| `SPARSE_RATIO` | `2.0` | LCSA 稀疏度；`1.5` 更快 / `2.0` 更稳 |
| `LOCAL_RANGE` | `11` | 局部注意力范围；`9` 更锐 / `11` 更稳 |
| `TILED` | `0` | `1`=full 分块（低显存，更慢） |
| `COLOR_FIX` | `1` | `1`=校正色彩偏移 |
| `PAD_FRONT` | `0` | 前垫帧数（因果状态预热，提升首帧质量）；裁剪时一并丢弃 |
| `PAD_BACK` | `4` | 后垫帧数（吸收模型末尾 ~4 帧 runoff）；**须 ≥4** 保证末帧可靠 |
| `MUX_AUDIO` | `1` | `1`=把原 LR 视频音频（MiniMax 立体声）mux 回 SR 输出 |
| `DEVICE` | `cuda` | GPU 选卡用 `GPU=N`（映射 `CUDA_VISIBLE_DEVICES`） |
| `OUTPUT_DIR` / `OUTPUT_NAME` | `../../output/minimaxh3_rotate_results/results_sr` / `<input>_sr.mp4` | |
| `INSTALL_DEPS` | `0` | `1`=clone 仓 + 装 BSA + diffsynth + requirements |

## Outputs
- **02 serve**: 日志 `../MiniMax-H3/logs/serve_<variant>_<port>.log` + PID 文件 `serve_<variant>_<port>.pid`（BG 模式）。
- **03 generate**: 视频 `../../output/minimaxh3_rotate_results/results/<task>/<name>.mp4`（含原生立体声）。
- **06/06a diffusers**: 视频 `OUTPUT_DIR/<name>.mp4`（768p/低分辨率 24fps 含原生立体声）。
- **07 FlashVSR SR**: 视频 `../../output/minimaxh3_rotate_results/results_sr/<name>_sr.mp4`（4× 超分，mux 原音频）。

## 目录布局
```
<code-dir>/
├── media_code/                  # 本仓
│   ├── proxy.env                # 代理 + 覆盖项, gitignored
│   └── minimax_h3/             # 编排脚本(本目录)
├── MiniMax-H3/                  # GitHub 参考仓(自动 clone 到 ../MiniMax-H3；含 scripts/skills)
│   ├── results/                # 生成视频输出
│   ├── results_int8/           # 06a int8 低分辨率输出（07 的典型输入）
│   ├── results_sr/             # 07 FlashVSR 超分输出
│   └── logs/                    # 服务日志
├── FlashVSR/                   # 07 自动 clone（diffsynth + examples/WanVSR/utils）
├── Block-Sparse-Attention/     # 07 自动 clone（LCSA CUDA kernel）
└── ../../model/MiniMax-H3/      # HF 权重快照(<code-dir> 上一级, 各算法共享)
    ├── model_index.json         # 仓库级公共入口(SGLang 读)
    ├── FL2VA/                   # FL2VA 任务族(transformer/text_encoder/tokenizer/processor/visual_vae/audio_vae)
    ├── Ref2VA/                  # Ref2VA 任务族(DOWNLOAD_REF2VA=1 才下)
    └── ../FlashVSR/             # 07 权重(diffusion_pytorch_model_streaming_dmd.safetensors 等)
```
默认：参考仓 `../MiniMax-H3`、权重 `../../model/MiniMax-H3`（相对本目录）；用 `MINIMAX_H3_DIR` / `MODEL_DIR` 覆盖。SGLang 服务**只依赖 HF 权重快照**，GitHub 参考仓纯为方便看 scripts/skills。

## Notes
- 官方代码与权重遵循其自有 license（MiniMax H3 Community License，非商用——见 HF 仓库 LICENSE）。本目录只做编排，不复制官方代码。
- `.gitattributes`（仓根）强制 LF，Windows 推上来的脚本在 Ubuntu 上干净运行。
- `proxy.env`（代理凭证）gitignored，不入库；切勿把凭证写进脚本。
- 公司 TLS 拦截代理下：pip 用 `--trusted-host`；`hf`/`git` 用 CA bundle（`_env.sh` 优先 `~/.ca-bundle.crt`，由 `setup_ca_bundle.sh` 构建）。
