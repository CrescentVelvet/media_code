# Pi3X 实验记录

> 按日期倒序追加。命令见 README.md / README_wsl.md。

## 2026-11-20（追加）渲染全局对齐 bug 与 PPT 结论撤回

- **旧第 5 页「与实拍逐帧对齐」结论错误，撤回。** `05_render_orbit.py` 初版直接用
  相机中心 Umeyama 的 `R/t` 变换点云，虽然同页/评测已指出该旋转在平面轨迹上退化，
  渲染却没有同步改用 Wahba——导致旧投影比正确旋转差 **27.81°**，帧 000 相机中心偏
  **0.372m**，人物偏左下、右上大片出画。不是主体点云整体差，也不能写成“全部对应”。
- 修正：`R = wahba(Rp,Rg)`，尺度仍取中心 Umeyama 的 `s`，固定 R/s 后取相机中心均值
  平移 `t=mean(G-s*P@R.T)`；ATE RMSE **0.379→0.0107m**。修正后的帧 000/041 人物与
  家具构图基本对应，但仍有点云散点/孔洞，不能声称像素级或逐帧严格对齐。
- 初版 `preview_proj_vs_gt.png` 也是同一错误对齐口径，**不得再作质量证据**；以修正后的
  `poster_reproject.png` / `reproject.mp4` 为准。
- 环绕另修正取景：轨道弧心≠主体，主体质心作注视点；轨道半径 1.13m 会过近裁头，
  改为独立 `ORBIT_DIST=2.2m`、ELEV=5°。环绕首帧只是取 GT0 方位，不声称构图与 GT 相同。
- 同步替换 PPT 第 1/5/6/7 页相关素材与表述，重出 `orbit_nocond.mp4`、`reproject.mp4`。

## 2026-11-20 Pi3X 复现 + 佳佳抱枕（82 帧 Remy 包）GT 对账

**结论：复现成功，前馈位姿质量超预期。** 82 帧 1439×1919（去畸变后）一次前向出位姿+点云，
与采集包 ARKit GT 对账：旋转几乎零误差（Wahba 对齐后 mean 0.28°），相邻帧相对位姿
6.3mm/0.16°，恢复内参 fx 差 0.66%；纯图像模式输出**近似 metric**（Sim(3) scale 0.969）。

| 指标 | nocond（纯图像） | cond（注入 GT 内参+位姿） |
| --- | --- | --- |
| Sim(3) scale (pred→m) | 0.9687 | 1.0210 |
| 旋转误差 (Wahba, deg) mean/med/max | 0.28/0.27/0.67 | 0.16/0.14/0.53 |
| RPE 相邻帧 trans (m) mean/max | 0.0063/0.0164 | 0.0052/0.0122 |
| RPE 相邻帧 rot (deg) mean/max | 0.16/0.57 | 0.15/0.51 |
| fx 相对误差 mean | 0.66% | 0.64% |
| fy 相对误差 mean | 1.20% | 0.13% |
| 点云（conf>0.1+边缘过滤） | 19.4M 点（95.0% 保留） | 19.4M 点（95.2% 保留） |
| 前向耗时（82 帧 @434×574） | 985s（与 infinisplat 训练抢卡） | ~6min（独占 3090） |

- **ATE（中心对齐后 RMSE 0.38/0.43m）不可靠**：本包相机中心轨迹近平面（PCA 三个方向
  2.54/0.64/**0.03** m），中心 Sim(3) 在面外方向退化，残差是常值全局旋转（逐帧 27.8°±0.1，
  方向一致）——这不是跟踪误差。Wahba 对齐（与中心解耦）后旋转误差 0.28°，证明位姿朝向
  预测本身是对的。**评测协议教训：平面轨迹上报 ATE 必须同时报 PCA 退化标志，旋转误差用
  Wahba 口径（已固化进 eval_gt.py）。**
- 点云几何尺度合理性（仅作粗检）：对齐到 GT 尺度后投回 GT 第 0 帧，41% 点在视锥内、
  中位深度 1.63m（人手持手机距抱枕约 1-2m ✓）、bbox ~10×5×11m（房间尺度 ✓）。
  ⚠️ 当时 `preview_proj_vs_gt.png` 使用错误的中心对齐旋转，**目检“位置吻合”结论撤回**；
  正确投影见本文件顶部追加记录。
- 条件注入（Pi3X 招牌能力）各项指标小幅全面变好，幅度不大——说明纯图像模式本身已接近
  GT 水平，条件注入的边际收益在本包上有限。
- 输入分辨率：Pi3X loader 把 1439×1919 压到 434×574（PIXEL_LIMIT=255000 官方默认），
  在此低分辨率下达到上述精度；加大 PIXEL_LIMIT 有望更准但更吃显存（未测）。

### 环境/基建坑（已固化进脚本）

- `pi3x` env = CPython 3.10 + torch 2.5.1+cu124（本地 `/mnt/d/wheel` wheel --no-deps）+
  torchvision 0.20.1+cu124（--no-deps，pytorch index）+ nvidia-* 钉版（**tuna pypi**，
  download.pytorch.org 的 nvidia wheel 实测卡 0 字节）+ numpy 1.26.4 等。
- `nvidia-cufile-cu12==1.9.1.3`（torch 2.5.1 元数据要求）tuna 上不存在（最低 1.11），
  仅 GPUDirect 文件 IO 用，前馈推理不需要 → 可选安装。
- torch 本地 wheel --no-deps 装完缺 sympy/networkx/jinja2，需补；sympy 钉 1.13.1。
- 权重下载：HF 走代理 ~0.1MB/s 不可用；**hf-mirror.com 直连 2.1MB/s 且支持 Range →
  6 段并行 ~5.7MB/s**（必须绕开 proxy.env 代理）；5,440,325,620 B / 5.07 GiB。
- 无 CUDA 编译版 RoPE2D → 走 PyTorch 慢速 fallback（官方自带警告），前向因此偏慢；
  如需提速可后续编译官方 rope CUDA 扩展【待办】。
- Pi3X loader 按文件名 sorted 读取 → 02 输出 %05d.png 保证顺序与 conditions.npz 一致；
  条件注入时 intrinsics 会被 loader 按 resize 比例自动缩放，无需手动换算。
- load_state_dict(strict=False)：nocond 模式 unexpected=436（多模态分支权重，预期行为）。

### 产物

- WSL：`~/output/pi3x_results/jiajia_baozhen/`；D 盘：`/mnt/d/output/results/pi3x_results/jiajia_baozhen/`
- `images/`（82 张去畸变 PNG）、`conditions.npz`（GT）、`{nocond,cond}/predictions.npz +
  dense_cloud.ply + poses.json + intrinsics.json + eval_gt.json`、`preview_proj_vs_gt.png`
- 权重：`/mnt/d/wheel/pi3x_ms/model.safetensors`（软链 `~/model/pi3x/`）
