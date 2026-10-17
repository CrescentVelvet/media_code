# FlashWorld 权重下载清单（迅雷/IDM 用）

单线程实测（2026-10-17，WSL）：`hf-mirror.com` ≈ 1.9 MB/s、`modelscope.cn` ≈ 2.3 MB/s
→ 合计 ~54 GB，单线程要 7 小时左右。**用迅雷多线程会快很多**，下完跑
`bash flashworld/01b_link_local_weights.sh` 挂进 HF cache 即可（软链接，不占空间）。

目录约定（01b 默认按这个找）：

```
/mnt/d/wheel/flashworld_ms/            # = $FLASHWORLD_MODEL_DIR
├── Wan2.2-TI2V-5B-Diffusers/          # 34.2 GB，保持仓库内相对路径
│   ├── model_index.json
│   ├── vae/
│   ├── transformer/
│   ├── text_encoder/
│   ├── tokenizer/
│   └── scheduler/
└── FlashWorld/
    └── model.ckpt                     # 19.5 GB
```

---

## 1. Wan2.2-TI2V-5B-Diffusers（34.2 GB，20 个文件）

镜像前缀二选一（同路径，换域名即可）：

- HF 镜像：`https://hf-mirror.com/Wan-AI/Wan2.2-TI2V-5B-Diffusers/resolve/main/`
- ModelScope：`https://www.modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers/resolve/master/`

| 相对路径 | 大小 |
|---|---|
| `model_index.json` | 0.3 KB |
| `configuration.json` | 0.3 KB |
| `scheduler/scheduler_config.json` | 0.3 KB |
| `tokenizer/special_tokens_map.json` | 0.5 KB |
| `tokenizer/tokenizer_config.json` | 0.5 KB |
| `tokenizer/spiece.model` | 4.5 MB |
| `tokenizer/tokenizer.json` | 16.8 MB |
| `text_encoder/config.json` | 0.8 KB |
| `text_encoder/model.safetensors.index.json` | 12 KB |
| `text_encoder/model-00001-of-00003.safetensors` | 4935.8 MB |
| `text_encoder/model-00002-of-00003.safetensors` | 4983.1 MB |
| `text_encoder/model-00003-of-00003.safetensors` | 1442.9 MB |
| `transformer/config.json` | 1.3 KB |
| `transformer/diffusion_pytorch_model.safetensors.index.json` | 0.1 MB |
| `transformer/diffusion_pytorch_model-00001-of-00005.safetensors` | 4978.3 MB |
| `transformer/diffusion_pytorch_model-00002-of-00005.safetensors` | 4846.8 MB |
| `transformer/diffusion_pytorch_model-00003-of-00005.safetensors` | 4972.7 MB |
| `transformer/diffusion_pytorch_model-00004-of-00005.safetensors` | 4846.8 MB |
| `transformer/diffusion_pytorch_model-00005-of-00005.safetensors` | 354.8 MB |
| `vae/config.json` | 0.3 KB |
| `vae/diffusion_pytorch_model.safetensors` | 2818.8 MB |

> `assets/` `examples/` 里的 png/jpg 是文档图，**不用下**。
> 总大小按上面的表算：文本编码器 11.4 GB + DiT 20.0 GB + VAE 2.8 GB ≈ 34.2 GB。

## 2. FlashWorld model.ckpt（19.5 GB，1 个文件）

- HF 镜像：`https://hf-mirror.com/imlixinyang/FlashWorld/resolve/main/model.ckpt`
- HF 直链（本机不通，海外可用）：`https://huggingface.co/imlixinyang/FlashWorld/resolve/main/model.ckpt`

HF 上该仓库 `usedStorage = 20,936,494,840` 字节（≈19.5 GB），只有这一个权重文件。

---

## 校验

下完跑：

```bash
CHECK_ONLY=1 bash flashworld/01_download_models.sh    # 下载后（HF cache 布局）
# 或
bash flashworld/01b_link_local_weights.sh             # 迅雷目录 → 挂进 cache 并校验
```

`01_download_models.py` 会检查 `model_index.json` / vae / transformer 分片 / text_encoder
分片是否齐全，以及 ckpt 是否 >19 GB。

## 许可

- FlashWorld 代码 **CC BY-NC-SA 4.0**，仓库 README 明确「仅限学术研究」。
- Wan2.2 权重遵循 Wan 原仓库许可（Apache-2.0），商用请自行确认下游条款。
