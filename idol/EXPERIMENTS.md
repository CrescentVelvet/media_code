# Idol 复现实验记录

按日期倒序追加。这里只记**结论与踩坑**，运行命令看 README。

---

## 2026-11-02 首次复现（RTX 3090 / WSL2）

### 结论

| 项 | 结果 |
|---|---|
| 官方仓 | `speridlabs/idol-sdk`，Apache-2.0，2026-10-29 首发，核心代码仅 463 行 |
| 环境 | conda env `idol`，python 3.11 / torch 2.8.0+cu128 / transformers 4.57.6 / diffusers 0.36.0 / flash-attn 2.8.3.post2 |
| 权重 | 6.4 GB 全落 `D:\wheel\hf_home_idol`（HF cache），下载 48 min @1.5~2.3 MB/s |
| 单张耗时 | 262k 高斯 80 s（含首次 compile）/ 批量同进程后续 ≈40 s；1M 高斯 69 s |
| 峰值显存 | 9,456 MiB（1M 配置）≈ 9,452 MiB（262k 配置） |
| 产物 | 标准 3DGS PLY（262k → 18 MB，1M → 69 MB），17 属性齐全 |
| 验证 | 蛋椅 / 机甲机器人 / 甜甜圈 / 玩具火车 / Q 版熊猫，六视角投影无塌陷重影 |

**显存结论**：高斯数从 262k → 1M（4 倍），显存只变 4 MiB。瓶颈是 6.4 GB 权重常驻 + CUDA context，
不是解码后的高斯。所以 24 GB 卡上可以放心用 1,048,576。

### 依赖链的三个坑（都会让安装跑偏，值得记）

1. **`tensordict>=0.8` 会把 torch 顶掉**。它不是硬 pin，pip 解析到最新 tensordict 时，
   后者声明 `torch>=2.11.0a0`，于是 pip 开始下 torch 2.15.0（539 MB），把刚装好的
   torch 2.8.0+cu128 和本地 nvidia-cu12.8 整套 wheel 全部作废。
   → 钉 `tensordict==0.8.3` + `--no-deps`，装完复查 torch 版本。
2. **`--no-deps` 的连带缺失**。tensordict 的 `cloudpickle` 要手工补，否则报
   `No module named 'cloudpickle'` 挂在 `import idol`。
3. **`--no-index` 装不了 torch**。torch 的纯 python 依赖（filelock / sympy / numpy / pillow）
   `D:\wheel` 里没有，`--no-index` 直接 `No matching distribution found`。
   正确姿势：先联网补这些小包，再 `--no-index --find-links` 装 torch/torchvision/nvidia-*。
   反过来「只用 `--find-links` 不加 `--no-index`」也不行 —— pip 会无视本地 wheel，
   照样从 tuna 重下 888 MB torch + ~3 GB nvidia 包。

### 下载通道的三个坑

| 配置 | 实测速度 | 结论 |
|---|---|---|
| 直连 HF | 0（curl 000，连接重置） | 不可用 |
| 代理 10808 + hf_transfer=1 | 129 K / 2 min，incomplete 长期 0 字节 | **卡死** |
| 代理 10808 + hf-xet 默认开 | 0.54 MB/s | 慢 3 倍 |
| 代理 10808 + `HF_HUB_DISABLE_XET=1` + hf_transfer=0 | **1.7~2.3 MB/s** | 采用 |

### 编译器的坑

`torch.compile` 走 triton，需要 C 编译器。本机 WSL **没有系统 gcc**，`sudo` 要密码装不了
build-essential。改用 `conda install -c conda-forge gcc_linux-64 gxx_linux-64`（不需要 root），
但它只提供 `x86_64-conda-linux-gnu-gcc`，而 triton 硬找 `gcc`，报
`InductorError: Failed to find C compiler`。用 `CC`/`CXX` 环境变量指定即可。

副作用：这组 conda 包的 activate 脚本会往环境里注入 `SIZE` / `STRIP` / `AR` 等 binutils 变量名
（`SIZE=x86_64-conda-linux-gnu-size`），把业务脚本里的 `SIZE` 变量顶掉 —— `04_preview.sh` 因此
改名 `PREVIEW_SIZE`。

### 其他

- **别在脚本运行中 Edit 该脚本**。下载跑了 48 min，我中途改了 `01_download_models.sh` 的统计逻辑，
  bash 按字节偏移继续读，读到错位内容后报 `unexpected EOF while looking for matching '"'`。
  脚本本身没问题（`bash -n` 通过），是「边跑边改」的锅。
- **预览工具的渲染方式影响判读**。最初用「近覆盖远」（画家算法）画散点，1M 熊猫比 262k 还稀疏发暗 ——
  因为 1M 里 65% 的高斯 opacity<0.5，本该叠加成不透明，单一覆盖只取最后落点。
  改成 alpha 加权累积（`sum(c·a)/sum(a)`）后正常。**这不是模型缺陷，是可视化工具的缺陷**。
- 官方口径 RTX 4090 19.1 s/图 vs 本机 3090 约 40 s（纯采样）。未逐项对齐性能，
  差距符合同代卡算力比，不视为复现问题。

### 未做 / 待办

- [ ] `ATTN_BACKEND=flash_attn` vs `sdpa` 的耗时对比（flash-attn 2.8.3 已装但未验证是否真被用上）
- [ ] `--mode real` 实拍照片路径未测（目前 5 个样本全是 synthetic 渲染/生成图）
- [ ] 输出 PLY 与输入视角的一致性误差（PSNR/SSIM）未量化，只做了六视角肉眼核对
- [ ] Gradio demo（`03_run_demo.sh`）未起过
