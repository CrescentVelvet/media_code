# ZipSplat 实验记录

## 2026-11-20 — 三输入同协议横评与复现汇报（补充）

### 共同协议与实现修正

- 复用 `compare_4methods/audit_31_eval_3view.py`，ZipSplat 通过 `EXTRA_JSON` 接入同一
  PLY 读取、相机对齐、渲染与指标链，不另造评分口径。两个分支都校验真实输入文件名为
  `000_5093309607791922.png`、`001_5093317707502922.png`、`002_5093325907213922.png`，
  对应原始帧 0/40/81；与历史三输入 PNG 逐张 SHA-256 一致。各 46,656 高斯。
  TokenBA 分支前馈 0.9841s、TokenBA 2.2145s。
- 原始 1439×1919 → 中心 1435×1435（x0=0,y0=235）→ 512×512；固定 GT 内参。
  三输入、32 GT、29 留出。三输入 Umeyama 初值后，仅用三输入做 Sim(3) 光度精化并冻结高斯；
  留出只用于评分。**这不是 16v 的 Wahba 协议，两组读数不可混报。**
- 发现旧 `sim3_refine.py` 光度精化固定白底、E14 最终评分黑底不一致。新增 `bg_value`
  （默认 1.0 保持旧调用），共同评测显式传 0.0，DA3-GS / SA / ZipSplat 全部重跑黑底。
  因此旧 E13 白底 10.0658/11.2513、E14 黑底 12.8009/13.4261 不用于本次排行榜。
- `03b_eval_3view_compare.sh` 以 `RESULT_NAME` 分方法保存，避免覆盖结果。
  ZipSplat / DA3-GS 为无位姿输入；历史 SplatAnything v3 为 **pose 模式**。

### 29 留出实测（统一黑底，单场景、非普适排名）

| 方法 | PSNR (dB) | SSIM | 锐度比 | alpha>0.5 覆盖率 |
|---|---:|---:|---:|---:|
| DA3-GS | 12.9516 | 0.3315 | 0.0676 | 0.7939 |
| SplatAnything v3（有位姿） | 13.4221 | 0.3805 | 0.3078 | 0.8262 |
| ZipSplat 纯前馈（无位姿） | 17.6399 | 0.4558 | 0.1149 | 1.0000 |
| ZipSplat + TokenBA（无位姿） | 17.2682 | 0.4729 | 0.2953 | 0.99994 |

- 本组中 ZipSplat 覆盖近完整，但前馈锐度保留偏低；TokenBA 提升 SSIM 与锐度比，
  PSNR 却较纯前馈 −0.3717dB，**不能称全面提升**。与 16v 的 +0.238dB 不矛盾：协议不同。
- 同帧目检可见主体结构与室内布局成立，但人物脸、抱枕五官、衣服文字、地面纹理仍明显偏软；
  覆盖率不是细节恢复证明。人物与手持抱枕微动仍是混杂因素。

### 汇报与归档

- PPT：`D:\ppt\20261120-ZipSplat复现与对比.pptx`（11 页）；源码在
  `D:\ppt\_src\20261120-ZipSplat复现与对比\`，页面均使用 `assets/...` 相对引用。
  第 6 页内嵌 16v TokenBA 环绕 MP4（504×504、96 帧、24fps、4s）；视频仅供目检。
- 新增产物：`jiajia_3v_compare/`、`jiajia_3v_compare_feedforward/`、
  `eval_3view_r512_black/`，均归档在 `D:\output\results\zipsplat_results\`。
  同帧证据在 `eval_3view_r512_black/renders/`；DA3-GS 读数在 `DA3-GS_metrics.json`，
  SA 读数在 `metrics.json`，两文件也各附 ZipSplat 两分支。

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
