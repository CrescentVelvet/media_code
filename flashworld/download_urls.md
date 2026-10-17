# FlashWorld 权重下载清单（迅雷用 · 已逐条校验）

**一键复制的清单文件**（纯 URL，一行一条，直接全选粘贴到迅雷「新建任务」）：

| 文件 | 内容 | 说明 |
|---|---|---|
| [`download_urls.txt`](download_urls.txt) | Wan2.2 基座 **16 条**（34.2 GB） | ModelScope，Windows 侧实测可用 |
| [`download_urls_ckpt.txt`](download_urls_ckpt.txt) | `model.ckpt` **2 条候选**（19.5 GB） | 先试 hf-mirror，不通再试 HF 直连 |

清单里**故意不含** 5 个 KB 级小文件（`text_encoder/config.json`、`transformer/config.json`、
`vae/config.json`、`scheduler/scheduler_config.json`、`tokenizer/tokenizer_config.json`）：
前者三个都叫 `config.json`，迅雷批量下载会变成 `config(1).json` 张冠李戴。
这些由 `01c_assemble_flat.py` 按完整相对路径补下，总共不到 100 KB。

---

## 1. 通道实测（2026-10-17，本机）

| 通道 | WSL 侧 | Windows 侧（迅雷所在侧） |
|---|---|---|
| `www.modelscope.cn` | ✅ ~2.3 MB/s 单线程，**支持 Range** | ✅ 可用（455 KB/s 单线程，迅雷多线程会快很多） |
| `hf-mirror.com` | ✅ ~2.3 MB/s | ❌ `CONNECT tunnel failed, response 502` |
| `huggingface.co` | ❌ RST | ⚠️ API 200，但 `resolve` 302 到 `cdn-lfs-us-1.hf.co` 后 **TLS 握手失败**（本机走 HTTP 代理，`Connection Established` 后 cdn 域名被拦） |

**结论：迅雷只能走 modelscope**（Wan 基座）。`model.ckpt` 只有 HF 有，
modelscope 搜过没有镜像 → 它继续由 WSL 后台用 hf-mirror 下（已在跑，约 2.3 h）。

## 2. 下载步骤

**① 迅雷批量下载（34.2 GB）**

复制 [`download_urls.txt`](download_urls.txt) 全部内容 → 迅雷「新建任务」→ 粘贴 →
保存目录设 `D:\wheel\flashworld_ms\thunder`（目录不存在就新建）。

> 迅雷会把 16 个文件**平铺**在该目录下，文件名就是原文件名 —— 这是可以的，
> 下一步的脚本会按文件名归位。

**② 归位 + 补小文件（在 WSL 里跑）**

```bash
cd /mnt/c/code/media_code
python flashworld/01c_assemble_flat.py --src /mnt/d/wheel/flashworld_ms/thunder
```

做了什么：按内嵌清单（21 个文件 + 精确字节数）把扁平文件**移动**到
`/mnt/d/wheel/flashworld_ms/Wan2.2-TI2V-5B-Diffusers/{vae,transformer,text_encoder,tokenizer,scheduler}/`，
尺寸不符的判为「没下完」并跳过，缺失的小文件自动从 modelscope 补下。
可重复执行（已就位的跳过）。想保留迅雷原件加 `--keep`。

**③ 挂进 HF cache**

```bash
bash flashworld/01b_link_local_weights.sh
```

**④ 校验**

```bash
CHECK_ONLY=1 bash flashworld/01_download_models.sh
```

## 3. 目录结构（01c / 01b 默认按这个找）

```
/mnt/d/wheel/flashworld_ms/                     # = $FLASHWORLD_MODEL_DIR
├── thunder/                                    # ← 迅雷保存目录（可删）
├── Wan2.2-TI2V-5B-Diffusers/                   # ← 01c 归位目标，34.2 GB
│   ├── model_index.json  configuration.json
│   ├── scheduler/scheduler_config.json
│   ├── tokenizer/{spiece.model,tokenizer.json,tokenizer_config.json,special_tokens_map.json}
│   ├── text_encoder/{config.json,model.safetensors.index.json,model-0000{1,2,3}-of-00003.safetensors}
│   ├── transformer/{config.json,diffusion_pytorch_model.safetensors.index.json,
│   │                diffusion_pytorch_model-0000{1..5}-of-00005.safetensors}
│   └── vae/{config.json,diffusion_pytorch_model.safetensors}
└── FlashWorld/model.ckpt                       # ← HF 单文件，19.5 GB
```

## 4. 完整文件清单（21 个，合计 34.20 GB）

字节数与 ModelScope API、HF API 的 `size` 字段逐个核对一致（2026-10-17）。

| 相对路径 | 字节 | 来源 |
|---|---|---|
| `configuration.json` | 74 | `download_urls.txt` |
| `model_index.json` | 499 | `download_urls.txt` |
| `scheduler/scheduler_config.json` | 820 | 01c 补下 |
| `text_encoder/config.json` | 855 | 01c 补下 |
| `text_encoder/model-00001-of-00003.safetensors` | 4,935,812,536 | `download_urls.txt` |
| `text_encoder/model-00002-of-00003.safetensors` | 4,983,103,192 | `download_urls.txt` |
| `text_encoder/model-00003-of-00003.safetensors` | 1,442,935,480 | `download_urls.txt` |
| `text_encoder/model.safetensors.index.json` | 22,476 | `download_urls.txt` |
| `tokenizer/special_tokens_map.json` | 7,079 | `download_urls.txt` |
| `tokenizer/spiece.model` | 4,548,313 | `download_urls.txt` |
| `tokenizer/tokenizer.json` | 16,837,459 | `download_urls.txt` |
| `tokenizer/tokenizer_config.json` | 61,758 | 01c 补下 |
| `transformer/config.json` | 495 | 01c 补下 |
| `transformer/diffusion_pytorch_model-00001-of-00005.safetensors` | 4,978,254,344 | `download_urls.txt` |
| `transformer/diffusion_pytorch_model-00002-of-00005.safetensors` | 4,846,784,976 | `download_urls.txt` |
| `transformer/diffusion_pytorch_model-00003-of-00005.safetensors` | 4,972,658,392 | `download_urls.txt` |
| `transformer/diffusion_pytorch_model-00004-of-00005.safetensors` | 4,846,785,080 | `download_urls.txt` |
| `transformer/diffusion_pytorch_model-00005-of-00005.safetensors` | 354,751,840 | `download_urls.txt` |
| `transformer/diffusion_pytorch_model.safetensors.index.json` | 73,297 | `download_urls.txt` |
| `vae/config.json` | 1,701 | 01c 补下 |
| `vae/diffusion_pytorch_model.safetensors` | 2,818,777,808 | `download_urls.txt` |

`assets/` `examples/` 下的 png/jpg 是文档图，不用下。

## 5. 为什么不用 huggingface_hub 直接下

`01_download_models.py` 走 `snapshot_download`，56 GB 在 hf-mirror 上单线程 ~2.3 MB/s，
且本机实测中途会 `httpx.ReadTimeout` 断掉（已发生过一次）。迅雷多线程 + 断点续传更稳。

## 6. 许可

- FlashWorld 代码 **CC BY-NC-SA 4.0**，仓库 README 明确「仅限学术研究」。
- Wan2.2 权重遵循 Wan 原仓库许可（Apache-2.0），商用请自行确认下游条款。
