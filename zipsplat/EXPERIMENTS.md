# ZipSplat 实验记录

## 2026-11-20 — 佳佳抱枕：官方 504px 权重推理复现

### 复现范围

- 官方 `cvg/ZipSplat` commit `73ab63f048fcc15ee54ada4ef9cdd79b925611a8`；没有修改官方实现。
- 官方 `zipsplat-da3g-504.tar`（5.59 GiB；16 Gaussians/token），SHA-256：
  `85d28887dbaee3f2116cf464ab0a15053a40b2f88bff69aa09e7b0f6fda46fd4`。
- WSL Ubuntu 24.04，RTX 3090 24GB；独立 conda env，Python 3.11.17、
  torch 2.8.0+cu128、gsplat 1.5.3。验证的是发布权重推理，不是从零训练。
- 佳佳抱枕 Remy 包：82 张 1440×1920 HEIC → 去畸变、ROI 后 1439×1919 PNG。
  16 输入为 `0,5,11,16,22,27,32,38,43,49,54,59,65,70,76,81`；
  8 留出为 `3,14,19,36,47,62,74,79`，不参与预测或 TokenBA。
- 网络实际输入 378×504；原始竖图中心裁剪由官方 `prepare_inputs` 处理。

### 实测结果

| 分支 | 高斯数 | 前馈 / TokenBA (s) | 峰值 allocated (MiB) | 输入自洽 PSNR / SSIM | 8 留出 PSNR / SSIM | 留出 alpha>0.5 |
|---|---:|---:|---:|---:|---:|---:|
| 16v 纯前馈 | 248,832 | 2.51 / — | 9,832.2 | 20.827 / 0.5088 | 19.757 / 0.4772 | 1.0000 |
| 16v 前馈 + TokenBA | 248,832 | 2.46 / 10.56 | 9,832.2 | 23.334 / 0.6453 | 19.995 / 0.4817 | 0.9998 |

- 模型加载约 23s；16v TokenBA 执行官方默认 4 轮上限、96 joint steps。
- TokenBA 对输入自洽 +2.507dB、SSIM +0.1365；对留出仅 +0.238dB、+0.0045 SSIM。
  **输入侧收益远大于留出，不能据此宣称具有普遍 NVS 增益。**
- 三视角冒烟（前馈、无 BA）也已跑通：46,656 Gaussians，0.76s 前馈，
  峰值 8,619.3MiB；首次全分辨率渲染触发 120.4s CUDA JIT 编译。
- TokenBA 输入相机：Wahba 平均旋转误差 0.273°；米制 ATE 0.0099m，但
  中心轨迹 PCA `[1.104,0.672,0.033]`，标记 `planar_degenerate=true`，ATE 只作诊断。

### 评测口径与目检

- 输入自洽使用预测内参渲染原分辨率 GT；留出 GT 相机只用 16 输入相机对齐，
  不进入模型或优化。最终留出对齐为 Wahba 全局旋转 + 输入中心估计尺度/平移。
- PLY 的 SH 为 channel-first 存储，加载时恢复 `(K,3)`；黑底、gsplat 渲染。
- 目检正/侧面输入视角和留出视角：人物、抱枕、椅子与室内结构可辨，未见全黑、
  大面积空洞或整体崩溃；但人脸、抱枕五官、衣服文字和地面纹理显著模糊，局部
  有重影。该包为人物＋抱枕＋环境，手持抱枕有微动，不是静态纯抱枕重建。
- 环绕视频只供目检，不是独立新视角证据。未测 LPIPS，未复现论文训练/基准。

### 实现修正

- 官方 `load_image` 接收路径，不能传 PIL Image；前馈日志改用 `pred["model_images"]`。
- 严格纯前馈需要独立 `BA=0` 运行以保存前馈相机；将前馈高斯与 BA 后相机组合的
  `jiajia_16v/evaluation_feedforward/` 是历史误配，**不得作为前馈自洽结果引用**。
  正确对照在 `jiajia_16v_feedforward/evaluation/`。归档保留该目录供审计，但不用其报数。
- `proxy.env` 只做 proxy/no_proxy 白名单解析，不 source 整个含密钥共享配置；
  曾因错误赋值使结果目录为空，已用显式路径完成模型、评测和归档验证。
- DrvFs 上 `cp -a` 的属主/时间戳保留会失败；改普通 `cp -r` 并逐文件长度校验。

### 产物

- `D:\output\results\zipsplat_results\jiajia_16v\`：前馈/BA PLY、相机、渲染、
  环绕视频及 `evaluation/metrics.json`。
- `D:\output\results\zipsplat_results\jiajia_16v_feedforward\`：严格前馈对照。
- `D:\output\results\zipsplat_results\jiajia_smoke_3v\`：三视角冒烟。
- 权重：`D:\wheel\zipsplat\zipsplat-da3g-504.tar`；代码与权重仅限各自许可范围，
  其中发布权重为 CC BY-NC 4.0，非商用。
