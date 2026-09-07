# vlm_bridge — 无视觉主模型的图像识别桥接（GLM-5.3 → Qwen3.8-Max API）

写代码用的主模型（GLM-5.3）**没有视觉能力**。遇到需要"看图"的场景（检查渲染结果、
读截图里的报错、对比两帧差异、确认分割 mask 是否干净），AI 直接调用本目录的
`ask_vlm.py`：把图片 base64 发给 Qwen3.8-Max（火山网关，OpenAI 兼容接口），
拿回文本描述。**全程无需用户切换模型。**

配置来源：`~/.workbuddy/models.json` 中 `supportsImages: true` 的条目（key / url /
model），已填进 `vlm.env`（gitignored，不入库）。

## 与 qwen3vl/ 的区别
`qwen3vl/` 是本地权重批量推理（transformers + GPU + 下载权重，图生文 captioning），
面向数据集批量处理。`vlm_bridge/` 是 **API 单次即时问答**，零依赖（纯标准库）、
秒级响应，面向"AI 干活过程中临时看一张图"。

## 常用命令
```bash
# 0) 首次：连通性自检（会自动从 example 生成 vlm.env）
bash vlm_bridge/00_check_env.sh

# 1) 单图问答（问题写具体，见下方「提问技巧」）
python vlm_bridge/ask_vlm.py minimax_h3/examples/681533632532078.jpg "这张图里有什么？画面内容是什么？"

# 2) 双图对比（最多 4 张）
python vlm_bridge/ask_vlm.py a.jpg b.png "这两张图有什么区别？"

# 3) 问题里带引号 / 换行时，用管道传入
echo '图里的报错信息原文是什么？逐字给出' | python vlm_bridge/ask_vlm.py screenshot.png

# 4) 覆盖配置（不弹 guidance 后缀）
VLM_MODEL=qwen3.8-max python vlm_bridge/ask_vlm.py --raw img.jpg "自定义完整 prompt"
```

## Config（vlm_bridge/vlm.env，从 vlm.env.example 复制）
| var | default | note |
|---|---|---|
| VLM_API_URL | 火山网关 chat/completions | OpenAI 兼容端点 |
| VLM_API_KEY | （models.json 里的 key） | 轮换后从 models.json 同步 |
| VLM_MODEL | qwen3.8-max | 视觉主力模型 |
| VLM_FALLBACK_MODEL | doubao-seed-2.1-pro | 主模型报错时自动重试一次 |
| VLM_MAX_TOKENS | 1024 | 单次回答上限 |
| VLM_HTTP_PROXY | （空） | 需要代理时填公司代理地址 |

## 提问技巧（AI 桥接调用时同样适用）
- **问题要具体**：「渲染质量如何」不如「人脸区域是否模糊？边缘有无锯齿/伪影？」
- **读文字**：直接说「逐字给出图中所有文字/报错信息」
- **对比**：明确对比维度（清晰度 / 颜色 / 物体位置 / 伪影）
- 脚本默认自动附加 guidance（要求具体、可验证、不确定就说），`--raw` 关闭

## 可能遇到的问题
1. **401 / 403**：key 失效——去 `~/.workbuddy/models.json` 拿最新 key 更新 `vlm.env`
2. **连不上 API**：公司网络需代理时，在 `vlm.env` 设 `VLM_HTTP_PROXY`
3. **模型下线**：`VLM_FALLBACK_MODEL` 会自动兜底重试；两个都挂就换 models.json 里
   其它 `supportsImages: true` 的条目
4. **图太大**：API 对 base64 体积有上限（一般 ~10MB），超大图先缩分辨率再问

## 目录布局
```
vlm_bridge/
├── vlm.env.example   # 配置模板（入库）
├── vlm.env           # 实际配置（gitignored）
├── ask_vlm.py        # 桥接脚本（纯标准库，零依赖）
├── 00_check_env.sh   # 连通性自检
└── README.md         # 本文件
```
