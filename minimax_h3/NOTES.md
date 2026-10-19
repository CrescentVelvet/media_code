# NOTES.md — MiniMax-H3 原理与排错手册

> 分工：`README.md` / `README_wsl.md` = 怎么跑；`EXPERIMENTS.md` = 实验结论与数据；本文件 = **原理机制 + 排错手册**。
> 引用形式 `file:line` 均指本机 conda env 内的 diffusers 0.40.0 源码
> （`~/miniconda3/envs/minimax_h3/lib/python3.11/site-packages/diffusers/`）。

---

## 1. 模型构成与显存账

| 组件 | 规模 | 说明 |
|---|---|---|
| 主 transformer | 33B（bf16 ≈66 GB，int8 ≈33 GB） | Omni-Transformer，14 个分片；视频 + 音频 token 走同一条 packed sequence |
| text encoder | Qwen3-VL-32B（bf16 ≈63 GB，int8 ≈31 GB） | 14 个分片；**只取第 50 层 hidden state** |
| video VAE | 较小（fp16 ≈5 GB） | 空间 16× 下采样；**默认开空间 tiling** |
| audio VAE | 0.6 GB | 音量/音色分支 |
| Turbo LoRA（可选） | ≈2 GB | 蒸馏把 50 步压到 4/8 步 |

合计 bf16 ≈130 GB、int8 ≈65 GB。**结论：24 GB 卡必须量化 + offload；80 GB 卡也放不下，同样要 offload**——
「服务器 80 GB 所以没走 offload」是常见误判，它只是 CPU 内存宽裕得多。

### 1.1 为什么 text encoder 如此占地方，却只用一个中间层

`encoders.py:35` 定义 `text_encoder_layer: int = 50`，`encoders.py:61` 说明输出是
「decoder layer 50 之后、形状 `(1, num_text_tokens, 5120)` 的 hidden state」。
即：Qwen3-VL 全部 32B 参数都要前向，但**下游只吃第 50 层**。
这也解释了 09f 的体检方式——必须核对「拿的是不是第 50 层」，而不是最后一层。

⚠️ 由此带来的一个常见误读：Qwen3-VL 存在 **massive activation**（个别维度数值上万）。
任何按「整层 std / mean-pool 余弦」做的健康检查都会被这 1–3 个维度主导：
- 实测 prompt `a` 的第 50 层有单元素 = 15552，`15552/√5120 ≈ 217.4`，而 per-token RMS = 219.4
  —— **整个 RMS 就是一个元素撑起来的**。
- 剔除 3 个 massive 通道后，prompt 才真正可分（cos 0.17 / 0.44 / 0.81）。
→ **不要用原始 std 或 mean-pool 余弦判 text encoder 是否损坏。**

---

## 2. 三种 workflow（**没有 V2V**）

| workflow | 输入 | 用途 |
|---|---|---|
| `t2va` | 纯文本 | 文生视频 + 音频 |
| `fl2va` | 首帧 / 末帧图（可只给一张）+ 文本 | 图生视频；首尾同图可做闭环 |
| `ref2va` | 多张参考图 + 文本 | 多参考合成；**需要独立 `transformer_ref` 权重，本机未下载** |

**没有 video-to-video、没有局部重绘、没有 inpainting**。
「把一段已有视频改一改」在 H3 上不成立——要逐帧修手只能用图像编辑模型（另一条技术栈）。
`10_orbit_datagen.py` 里多图 = **每图各出一条**（数据工厂语义），不是多参考合成。

---

## 3. 必须记住的数值约定

| 约定 | 值 | 出处 / 说明 |
|---|---|---|
| 帧数 | **`17n + 5`**，且 120–360 帧（5–15 s @ 24 fps） | `before_denoise.py:173,201,218,404,411`；不够会被**向上取整**，静默改你的参数 |
| latent 形状 | `(1, 24, T_lat, H/16, W/16)`，`T_lat = (frames-5)/4 + 2` 量级 | 实测 124 帧 → 37；960×544 → 34×60 |
| latent 归一化 | `raw = norm * latents_std + latents_mean`；`vae.decode` 吃 **raw** | 09a 实测：raw 版 roundtrip 完美、norm 版低对比度 |
| std 基准 | `RMS(latents_std) ≈ 1.9250` | 判「是否收敛」的参照，见 `EXPERIMENTS.md` §9 |
| NIS 与步数 | `num_inference_steps = N+1` 表示 **N 步去噪** | scheduler 的 sigma grid 含末尾 0 点；传 51 = 50 步 |
| sigma 网格 | 经 **指数 shift** 变换，非 `linspace(1,0,N)` | `scheduling_minimax_h3.py:30,65`：`σ' = sσ / (1+(s-1)σ)` |
| video shift 默认 | **12.0** | `scheduling_minimax_h3.py:34,74`；快照 `scheduler_config.json` 也写着 12.0（**无版本漂移**） |
| audio shift 默认 | **3.0** | 同文件 `:34` —— **一次请求跑两套 schedule，音视频各一套** |
| turbo 768p 配方 | `VIDEO_SHIFT=6` | 训练时 shift=6；用 12 会明显劣化 |
| LoRA alpha | **留 `auto`**，读 checkpoint `__metadata__['alpha']` | 官方 `DEFAULT_LORA_ALPHA=8`，仅 `4step_v1.0_768p` 显式 128；写死 128 在 alpha=8 的文件上放大 16× |
| 无 CFG | guidance-distilled，**没有 unconditional pass / 没有 guider** | `denoise.py:55`；所有约束必须写进**正向** prompt |
| VAE tiling | **默认开启**（与库内其他 VAE 相反） | `autoencoder_kl_minimax_h3.py:520` |

### 3.1 随机数契约（复现实验必须对齐）

`before_denoise.py:848` 的注释明确：一次请求的所有 stream 都从**同一个 generator** 抽，
顺序是「条件噪声 → 视频噪声（latent 形状）→ 音频噪声（row 布局）」，
给了 `latents` / `audio_latents` 就跳过那一次抽取、并**顺移后续抽取**。

所以想用「重放初始噪声」做诊断，必须同时对齐三件事：**device（CPU）、seed、抽取次数与形状**。
`09b` 用 `torch.Generator("cpu")`，`06c/06d` 用 `cuda:0` —— 两者复现不出同一份噪声，
但**都不影响出片质量**，只影响可诊断性。

---

## 4. 两条官方内存配方 + 本机实际能用的组合

| | 配方 A（80 GB） | 配方 B（24–32 GB） | 本机实际 |
|---|---|---|---|
| 量化 | 无（bf16） | int8 weight-only（torchao `Int8WeightOnlyConfig(version=2)`） | int8 |
| offload | `ComponentsManager.enable_auto_cpu_offload()` | 手动 `enable_group_offload(offload_type="leaf_level")` | 手动 leaf_level |
| 落点 | 06b | 09b / 06c / 06d | 06d（turbo）/ 09b（50 步） |
| 峰值 | 80 GB 级 | 实测 **15.4 GB** VRAM | 同左 |

**两条配方不可混搭**：`09e` 就是混搭产物（CM manager + 手动 offload 的语义冲突），已标注作废。

### 4.1 两个 doc 与 code 不一致的地方（实测勘误）

1. **`low_cpu_mem_usage=False` 用不了**：官方 doc 的消费卡配方写 `False`，
   但 diffusers 0.40.0 在量化加载时**硬性断言必须 True**，`False` 直接 ValueError。
   → 一律 `True`。06c/06d 原本写 True 是对的。
2. **`use_stream=True` 不建议**（09b docstring 有完整推导）：
   它会启用 ① 每次 onload 临时 `pin_memory`，② `LazyPrefetchGroupOffloadingHook`
   靠 trace 首帧执行顺序预取下一组。而 t2va 下 Qwen3-VL 的 ~150 个 `visual.*` 层**根本不执行**，
   diffusers 自己就会警告 `layers were not executed ... device-mismatch`，
   trace 出的顺序必然残缺。关掉后是「不 pin、不预取、同步 H2D」的最保守路径，
   代价只是搬运不与计算重叠（本场景早已是 IO/swap 瓶颈）。

---

## 5. ⚠️ 本机已知坑（按踩到频率排序）

### 5.1 `torch.inference_mode()` 与 int8 group offload 冲突（会崩）

现象：text encoder 第一个 Linear 回搬时报
`set storage cuda:0 vs cpu`，栈里是
`torchao return_and_correct_aliasing → _correct_storage_aliasing`。

根因：`inference_mode` 产生的 inference tensor 不允许被 aliasing 修正原地改写。

修法：**用 `torch.no_grad()`**。`09b`（无上下文）、`09f`（`no_grad`）同样配置都正常，
`06d` 源码里本来就写着这条注释。
→ **凡是 int8 + group offload 的推理脚本，禁用 `inference_mode`。**

### 5.2 `set_attention_backend()` 的后端名在 0.40 变了

`"sdpa"` 不再是合法名（0.40 用 `native` / `_native_efficient` / `flash` / `xformers` 等）。
传旧名不会报错退出，而是**该 arm 被静默跳过**——`09g` 因此空跑过一轮。
→ 探针脚本用 `ARMS=default`（不动后端），或先 `print` 一次合法名。

### 5.3 大权重下载后**必须验 sha256**，且验法有讲究

见 `EXPERIMENTS.md` §5、§9.4。三条铁律：
1. **大小对 ≠ 内容对**：本次 5 个坏片大小精确、safetensors 头合法、加载不报错。
2. **「OK」可信，「HASHBAD」可疑**：内存枯竭时 drvfs 短读会让 `sha256sum` 输出截断流哈希。
   校验脚本必须打印**实际读取字节数**，字节不满 = 读失败，不是损坏。
3. 多线程下载工具 + 断电/中断 → 高发「大小对、内容错」。

### 5.4 WSL 内存枯竭会伪装成各种故障

`MemFree 1.6 GB / 56 GB`（跌破内核水位线）时：`dd` / `sha256sum` 读 `/mnt/d` 报
`Cannot allocate memory`；模型加载会莫名变慢甚至 OOM。跑重活前先 `free -g`。
本机 56 GB RAM 对 int8 的 65 GB 权重是**紧平衡**，并发任务极易打爆——
一次 09f 体检（VmHWM 55.6 GB）就足以让并行的哈希校验全线假警报。

### 5.5 LoRA 相关

- `LORA_ALPHA` **永远留 auto**；启动日志有一行 `LoRA alpha = ... → PEFT scale = ...`，必须核对。
- 768p turbo 必须 `VIDEO_SHIFT=6`；544p v0.1 用 12。
- 3090 跑 768p（1344×768）可能在 **VAE 解码**段 OOM（transformer 段不会）→ 降 `NUM_FRAMES` 或换 544p。
- 历史「出噪点」曾被怀疑是 alpha 写死导致，**已证伪**：出噪那几次 alpha 恰好全对。

### 5.6 工具链坑（本机 WSL + Windows 混用）

- `wsl.exe ... -- bash -lc '<多行脚本>'`：管道符/换行会被 Windows 侧 shell 吃掉；
  含 `$VAR` 时变量可能在**外层**被展开成空。→ **一律写 `.sh` 文件，再 `wsl -- bash <path>`**。
- `wsl ... bash x.sh > out.txt` 的重定向必须在单引号内：
  `wsl.exe -d Ubuntu2404 -- bash -lc '... > /mnt/c/.../out.txt 2>&1'`。
- 读 WSL 内文件可直接用 UNC `\\wsl.localhost\Ubuntu2404\home\velvet\...`。
- Windows 侧 Python 通常没有 PIL/numpy；拼图、抽帧用 **WSL 的 minimax_h3 env** 跑。
- `nohup ... &` 挂后台会被会话结束带走；长任务走任务化后台（`run_in_background`）。
- `hf-mirror.com` 的 `api` 端点通、`resolve` 端点可能不通（网络会漂移，动手前先 `curl` 复测）；
  WSL 不通时用 Windows 侧迅雷下载，**下完仍要验哈希**。
- `Write` 工具单次写入约 20 KB 会 JSON 解析失败 → 大文件分块写。

---

## 6. 排错决策树（症状 → 下一步）

```
出片是马赛克 / 彩色噪点 / 雪花
├─ 1. 先看形态
│   ├─ 整齐方格 + 每格纯色  → VAE 侧（tiling 接缝/归一化）→ 09a roundtrip
│   ├─ 细密 RGB 雪花        → 去噪未收敛 → 进 2
│   └─ 低对比度/发灰        → 归一化做重了（norm 版喂 decode）→ 用 raw
├─ 2. 判「去噪有没有干活」（不要用帧间相关，见 EXPERIMENTS §9）
│   ├─ raw latent z std / 1.925 ≫ 1.2，且与重放初始噪声相关 >0.3 → 没收敛
│   └─ 两项都正常，但画面仍差 → 看 4（条件/解码）
├─ 3. 没收敛 → 二分
│   ├─ 权重完整性：32 片 sha256（内存空闲 + 打印字节数）
│   ├─ 加载路径：tiny fixture 四配方 A/B（验证 offload 机制本身）
│   ├─ 量化：小 Linear 对比 fp32/bf16/int8 相对误差（应 ~5e-3）
│   └─ 条件注入：text encoder 第 50 层（提 massive 通道后再判）
└─ 4. 条件/解码侧
    ├─ prompt 是否写进「正向」？（无 CFG，无 negative）
    ├─ 帧数是否被静默取整？（17n+5）
    ├─ shift 与 LoRA 配方是否匹配？（768p turbo=6，基座=12）
    └─ LoRA alpha / scale 是否 16× 放大？（看启动日志那行）

崩溃 / 报错
├─ `set storage cuda:0 vs cpu`        → 用了 inference_mode，改 no_grad（§5.1）
├─ `Cannot allocate memory` (drvfs)   → 内存枯竭，不是文件坏了（§5.4）
├─ VAE 解码 OOM                       → 降 NUM_FRAMES 或换 544p（§5.5）
└─ offload 后 device mismatch 警告    → 关了 use_stream（§4.1）
```

---

## 7. 故障档案：马赛克（2026-08-31 ~ 10-08）

**现象**：本机 WSL 无论 06b（bf16）/ 06c（int8）/ 06d（int8+turbo）/ 09b（官方配方），
全部出雪花或粗块；**服务器同脚本同权重正常**。

**走了弯路的四个假设（都被证伪，留作前车之鉴）**

| 假设 | 为什么当时像 | 怎么被推翻 |
|---|---|---|
| offload 路径有 bug | 只有本机坏、服务器 80 GB 也 offload 但更宽裕；外部有 SGLang「mapped courier → 彩色噪声」同类报告 | tiny fixture 四配方逐字节一致（但**只证明机制，不证明规模**） |
| int8 量化打坏 | 06c/06d 是 int8 | 06b 是纯 bf16，**也坏** |
| LoRA alpha 写死 | 历史有 16× 放大事故 | 出噪那几次 alpha 恰好全对；06c/09b 无 LoRA 也坏 |
| 版本漂移（0.40 vs 0.36.dev） | 快照 `_diffusers_version: 0.36.0.dev0`，本机 0.40.0 + 自造 index | 26 个 json 与上游逐字节一致；index 只是原样复制；shift 值相同 |

**真正的根因**：`transformer/00001–00005` 五个分片损坏 —— 大小正确、头合法、
加载不报错，但张量内容是错的。替换后 09b 同参复跑即正常。

**为什么难查（可复用的教训）**

1. **「能加载」不等于「权重对」**。safetensors 只校验结构，不校验语义；
   量化更是把错权重再压一遍，错误被「平均」得不那么显眼。
2. **故障在最早的环节，症状在最晚的环节**。权重错 → 去噪输出垃圾 → 表现为「VAE 出马赛克」，
   于是所有注意力都被吸到 VAE / offload / 量化上。
3. **排除实验要能覆盖「规模」**。tiny fixture 能证明 offload 机制没 bug，
   但 0.16 MB 撑不起分配器压力，**不能**证明真实规模下没问题——
   所以它只能「排除机制」，不能「排除嫌疑」。
4. **一个可信的正向证据 > 一堆可疑的负向证据**。「某片哈希等于上游」这种结论不可能造假；
   「某片哈希不等」在内存压力下可能是读失败。定案要靠前者。

**最终判据（可复用）**：`z std / RMS(latents_std)` + 与重放初始噪声的残差相关 + 肉眼画面。
详见 `EXPERIMENTS.md` §9。

---

## 8. 与其他项目的关系

- **本目录产物是 `vggt_human` 的输入数据工厂**（`10_orbit_datagen.py`）：
  用 FL2VA 造「人物静止 + 相机环绕」的干净序列，绕开实拍里人物走动拖垮重建的问题。
- **能力边界要先钉死**：H3 能做「生成新视频」，不能做「修复已有视频」。
  修手（`HAND_MOTION_LEDGER.md` 里的 L4/L5 问题）不属于 H3 的能力范围；
  它只能在「补视角 / 造静止数据」这条轴上帮忙。
- 与 `wan22_rotate`、`4danyone`、`flashworld` 的分工见仓根 `README.md`。
