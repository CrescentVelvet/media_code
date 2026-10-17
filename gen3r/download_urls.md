# Gen3R 权重下载清单（迅雷/IDM 手动加速）

总计 **37.74 GB**，21 个文件。脚本自动下载（`01_download_models.sh`）在本机
约 2 MB/s（aifasthub），需要 4–6 小时；用迅雷多线程通常能快数倍。

## 用法（一键复制版）

**直接打开 [`download_urls.txt`](download_urls.txt)，全选复制，粘进迅雷「新建 → 批量任务」。**
该文件是纯 URL 列表（21 行，无表头、无注释），不会混进多余字符。

1. 迅雷下载目录设为 `D:\wheel\gen3r_ms`。
2. **子目录不用管**：迅雷若把文件摊平（`text_encoder/xxx` 变成 `xxx`），
   `01b` 会按文件名兜底匹配再整理成上游要的结构。
3. 下完执行：
   ```bash
   bash gen3r/01b_link_local_weights.sh
   ```
   脚本校验大小（精确到字节）后，在 `~/model/gen3r` 建软链指向 `/mnt/d/wheel/gen3r_ms`；
   想真复制到 Linux fs 就加 `LINK_MODE=copy`。

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

完整 URL 全在 [`download_urls.txt`](download_urls.txt)（纯列表，可直接全选复制）。

> 备选端点：`https://hf-mirror.com/JaceyH919/Gen3R/resolve/main/<相对路径>`
> （2026-10-17 实测 hf-mirror 大文件基本不可用，仅作参考）
