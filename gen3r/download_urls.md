# Gen3R 权重下载清单（迅雷/IDM 手动加速）

总计 **37.74 GB**，21 个文件。脚本自动下载（`01_download_models.sh`）在本机
约 2 MB/s（aifasthub），需要 4–6 小时；用迅雷多线程通常能快数倍。

## 用法

1. 把下面 URL 粘进迅雷/IDM 批量下载。
2. **保持相对路径**：存到 `D:\wheel\gen3r_ms\` 下，子目录结构必须与 URL 尾部一致
   （例如 `.../transformer/diffusion_pytorch_model.safetensors` →
   `D:\wheel\gen3r_ms\transformer\diffusion_pytorch_model.safetensors`）。
3. 下完执行：
   ```bash
   bash gen3r/01b_link_local_weights.sh
   ```
   脚本把 `D:\wheel\gen3r_ms`（WSL: `/mnt/d/wheel/gen3r_ms`）校验大小后**硬链接/复制**
   到 `~/model/gen3r`。

## 文件清单（按大小降序）

| 大小 | 相对路径 |
|---|---|
| 9.28 GB | `text_encoder/diffusion_pytorch_model-00002-of-00003.safetensors` |
| 9.19 GB | `text_encoder/diffusion_pytorch_model-00001-of-00003.safetensors` |
| 6.02 GB | `transformer/diffusion_pytorch_model.safetensors` |
| 4.44 GB | `vggt/model.safetensors` |
| 4.44 GB | `clip_image_encoder/diffusion_pytorch_model.safetensors` |
| 2.69 GB | `text_encoder/diffusion_pytorch_model-00003-of-00003.safetensors` |
| 1.18 GB | `geo_adapter/diffusion_pytorch_model.safetensors` |
| 0.47 GB | `wan_vae/diffusion_pytorch_model.safetensors` |
| 16 MB | `tokenizer/tokenizer.json` |
| 4.3 MB | `tokenizer/spiece.model` |
| 60 KB | `tokenizer/tokenizer_config.json` |
| 21 KB | `text_encoder/diffusion_pytorch_model.safetensors.index.json` |
| 7 KB | `tokenizer/special_tokens_map.json` |
| 828 B | `model_index.json` |
| 567 B | `transformer/config.json` |
| 450 B | `scheduler/scheduler_config.json` |
| 257 B | `geo_adapter/config.json` |
| 252 B | `text_encoder/config.json` |
| 168 B | `wan_vae/config.json` |
| 67 B | `clip_image_encoder/config.json` |
| 62 B | `vggt/config.json` |

## URL 前缀

```
https://aifasthub.com/JaceyH919/Gen3R/resolve/main/<相对路径>
```

示例（最大那个）：

```
https://aifasthub.com/JaceyH919/Gen3R/resolve/main/text_encoder/diffusion_pytorch_model-00002-of-00003.safetensors
```

> 备选端点：`https://hf-mirror.com/JaceyH919/Gen3R/resolve/main/<相对路径>`
> （2026-10-17 实测 hf-mirror 大文件基本不可用，仅作参考）
